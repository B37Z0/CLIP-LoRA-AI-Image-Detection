"""
GPU / environment feasibility test. Entirely AI-generated.

Run this on your local machine (Ubuntu 24.04, ROCm 7.2.4, `ml` conda env),
not in a sandbox. It checks:
  1. PyTorch sees your ROCm GPU
  2. peft and transformers import and run
  3. CLIP ViT-B/32 and ViT-L/14 (via transformers.CLIPVisionModelWithProjection)
     load, and LoRA attaches to their attention q_proj/v_proj layers
  4. Rough VRAM usage + step time for a forward/backward pass at bf16

Uses transformers rather than open_clip: open_clip's ViT implementation
relies on torch.nn.MultiheadAttention, which reads its out_proj weight/bias
directly for a fused functional call instead of invoking out_proj(x) as a
normal module forward — so PEFT's LoRA hook never actually runs, even
though the params show up as "trainable". transformers' CLIPAttention calls
q_proj/k_proj/v_proj/out_proj as genuine Python-level module calls, so LoRA
connects correctly. This is also what CLIP+LoRA papers in this space
(MoLE, HyperDet, RINE) are actually built on.

Install first (inside your `ml` conda env) if missing:
    pip install peft transformers
"""

import time
import torch

# ROCm's flash/memory-efficient SDPA backends are still marked experimental
# (see the warnings from your last run). Force the math backend for this
# feasibility test so a backend quirk doesn't masquerade as a real bug.
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)


def section(title):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def check_device():
    section("1. Device check")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA/ROCm available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"Total VRAM: {props.total_memory / 1e9:.1f} GB")
    else:
        print("WARNING: no GPU detected. Check ROCR_VISIBLE_DEVICES and your ROCm install.")
    return torch.cuda.is_available()


def check_imports():
    section("2. Library imports")
    ok = True
    for lib in ["peft", "transformers"]:
        try:
            __import__(lib)
            print(f"  {lib}: OK")
        except ImportError as e:
            print(f"  {lib}: MISSING ({e})")
            ok = False
    return ok


def test_clip_lora(model_id, device, r=8, alpha=8):
    """Load a CLIP vision tower via transformers (whose CLIPAttention calls
    q_proj/k_proj/v_proj/out_proj as real module forward passes, unlike
    open_clip's nn.MultiheadAttention), attach LoRA, and run one
    forward+backward pass at bf16. Reports time and peak VRAM.

    NOTE: we switched from open_clip to transformers here. open_clip's ViT
    uses torch.nn.MultiheadAttention internally, which reads out_proj.weight/
    .bias directly for a fused functional call rather than invoking
    out_proj(x) as a normal forward pass — so PEFT's LoRA hook on that
    module never actually runs. transformers' CLIPAttention calls each
    projection as a genuine Python-level module call, so LoRA connects
    correctly. This is also what the CLIP+LoRA papers (MoLE, HyperDet,
    RINE) are actually built on.
    """
    from transformers import CLIPVisionModelWithProjection
    from peft import LoraConfig, get_peft_model

    section(f"3. Testing {model_id} with LoRA r={r}")

    t0 = time.time()
    model = CLIPVisionModelWithProjection.from_pretrained(model_id)
    model = model.to(device)
    print(f"  Model load time: {time.time() - t0:.1f}s")

    target_modules = []
    for name, module in model.named_modules():
        if name.endswith(("q_proj", "v_proj")) and isinstance(module, torch.nn.Linear):
            target_modules.append(name)
    if not target_modules:
        print("  Could not auto-find q_proj/v_proj Linear layers — inspect "
              "model.named_modules() manually.")
        return

    print(f"  Target modules found: {len(target_modules)} total "
          f"(sample: {target_modules[:2]})")

    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
    )
    try:
        model = get_peft_model(model, lora_config)
    except Exception as e:
        print(f"  LoRA attach FAILED: {e}")
        return

    model.train()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.3f}%)")

    dummy_input = torch.randn(8, 3, 224, 224, device=device)

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()
    t0 = time.time()
    try:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(pixel_values=dummy_input).image_embeds
            loss = out.float().pow(2).mean()

        if not loss.requires_grad:
            print("  SANITY CHECK FAILED: loss.requires_grad is False — LoRA still "
                  "disconnected from the forward path.")
            return

        loss.backward()
        torch.cuda.synchronize()
    except Exception as e:
        print(f"  Forward/backward FAILED: {e}")
        return

    step_time = time.time() - t0
    peak_mem = torch.cuda.max_memory_allocated(device) / 1e9
    print(f"  Forward+backward step time: {step_time:.3f}s")
    print(f"  Peak VRAM used: {peak_mem:.2f} GB")

    del model, dummy_input, out, loss
    torch.cuda.empty_cache()


def main():
    has_gpu = check_device()
    imports_ok = check_imports()

    if not (has_gpu and imports_ok):
        print("\nStopping here — fix the above before testing model loading.")
        return

    device = "cuda"

    # Start small, then try the larger model. If ViT-L/14 fails or is too
    # slow/memory-heavy, that's useful information for scoping down.
    test_clip_lora("openai/clip-vit-base-patch32", device)
    test_clip_lora("openai/clip-vit-large-patch14", device)

    section("Done")
    print("If both models loaded, attached LoRA, and ran a step without error,")
    print("your environment is ready for real training. Compare peak VRAM against")
    print("your 16GB budget, remembering real training adds optimizer state,")
    print("larger batch sizes, and the frequency-branch CNN on top.")


if __name__ == "__main__":
    main()