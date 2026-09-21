"""
CLIP-ViT + LoRA detector for the AI-generated-image-detector.

LoRA targets q_proj/v_proj (to start) inside vision_model.encoder.layers, 
which are real nn.Linear modules in transformers' CLIPAttension. 
- open_clip's nn.MultiheadAttention bypasses LoRA's forward hook so it
  was not used.

Ohja et al. 2023:
- A frozen CLIP-ViT with only a linear-probe generalized better than 
  fine-tuning. This should also be tested as a baseline.
- Compare this to the minimal LoRA baseline depending on whether it 
  suffers a generalization drop to see if LoRA itself would be causing
  a drop or if it happens regardless...

Ablations:
- The frequency branch expects raw pixel values in [0,1] RGB so the 
  normalization needs to be inverted for the CNN input.

project_lora_null_space relies on pretty shaky name matching but if 
it works it works.
"""

import torch
import torch.nn as nn

# OpenAI CLIP standard normalization stats (to invert CLIPImageProcessor)
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


class CLIPLoRADetector(nn.Module):
    def __init__(self, model_id="openai/clip-vit-large-patch14", r=8, lora_alpha=8,
                 use_lora=True, use_freq_branch=False, freq_dim=128, lora_dropout=0.0,
                 null_space=False, null_space_k=32):
        super().__init__()
        from transformers import CLIPVisionModelWithProjection  # deferred import

        backbone = CLIPVisionModelWithProjection.from_pretrained(model_id)

        self.null_space = null_space and use_lora
        null_bases = {}

        if use_lora:
            from peft import LoraConfig, get_peft_model
            target_modules = [
                name for name, module in backbone.named_modules()
                if name.endswith(("q_proj", "v_proj")) and isinstance(module, nn.Linear)
            ]

            # LoRA-Null: for each target layer, compute the top-k right 
            # singular vectors of the frozen weights (before wrapping).
            # This represents the subspace CLIP relies on the most for 
            # its original representations (generalization). Project the 
            # LoRA A matrix to be orthogonal to this subspace (span) after
            # every optimizer step so adaption can't mess with the original
            # representation. 
            if self.null_space:
                for name, module in backbone.named_modules():
                    if name in target_modules:
                        _, _, Vh = torch.linalg.svd(module.weight.data, full_matrices=False)
                        k = min(null_space_k, Vh.shape[0]) # cap k at max input features
                        null_bases[name] = Vh[:k, :].clone() # extract top-k right singular vectors

            lora_config = LoraConfig(
                r=r, lora_alpha=lora_alpha, target_modules=target_modules,
                lora_dropout=lora_dropout, bias="none",
            )
            self.backbone = get_peft_model(backbone, lora_config)

            if self.null_space:
                # Register null bases as non-trainable buffers.
                self._null_basis_names = list(null_bases.keys())
                for name, v_top in null_bases.items():
                    self.register_buffer("nullbasis_" + name.replace(".", "_"), v_top)
        else:
            # Frozen linear-probe - no adaptation / only classifer head.
            for p in backbone.parameters():
                p.requires_grad = False
            self.backbone = backbone

        self.use_freq_branch = use_freq_branch
        # image_embeds dimension: 768 for ViT-L/14, 512 for ViT-B/32
        embed_dim = self.backbone.config.projection_dim 

        if use_freq_branch:
            from baseline_cnn import FrequencyBranch
            self.frequency_branch = FrequencyBranch(out_dim=freq_dim)
            self.register_buffer("clip_mean", CLIP_MEAN)
            self.register_buffer("clip_std", CLIP_STD)
            classifier_in = embed_dim + freq_dim
        else:
            classifier_in = embed_dim

        self.classifier = nn.Linear(classifier_in, 2)

    def project_lora_null_space(self):
        """
        Call after each optimizer.step() if null_space=True. Projects
        each LoRA A matrix's rows to be orthogonal to the frozen weight's
        top-k singular directions - eliminating any drift back into the
        protected subspace.
        """
        if not self.null_space:
            return
        for name, module in self.backbone.named_modules():
            matches = [l_name for l_name in self._null_basis_names if name.endswith(l_name)]
            # Skip modules w/o LoRA weights or that are not in the null space.
            if not matches or not hasattr(module, "lora_A"):
                continue
            v_top = getattr(self, "nullbasis_" + matches[0].replace(".", "_"))
            A = module.lora_A["default"].weight # (r, in_features)
            with torch.no_grad():
                component = (A @ v_top.T) @ v_top  # A's projection onto V_top's span
                A -= component

    def forward(self, pixel_values):
        image_embeds = self.backbone(pixel_values=pixel_values).image_embeds

        if self.use_freq_branch:
            # Invert CLIP normalization to recover [0,1] RGB approx.
            # clamp() to guard against small OOR values...
            raw_rgb = (pixel_values * self.clip_std + self.clip_mean).clamp(0, 1)
            freq_features = self.frequency_branch(raw_rgb)
            combined = torch.cat([image_embeds, freq_features], dim=1)
            return self.classifier(combined)

        return self.classifier(image_embeds)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    configs = [
        ("LoRA", dict(use_lora=True, use_freq_branch=False)),
        ("frozen linear-probe", dict(use_lora=False, use_freq_branch=False)),
        ("LoRA + frequency branch", dict(use_lora=True, use_freq_branch=True)),
        ("LoRA-Null", dict(use_lora=True, use_freq_branch=False, null_space=True)),
        ("LoRA-Null + frequency branch", dict(use_lora=True, use_freq_branch=True, null_space=True)),
    ]

    for label, kwargs in configs:
        model = CLIPLoRADetector(**kwargs).to(device)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"[{label}] Trainable params: {trainable:,} / {total:,} "
              f"({100 * trainable / total:.3f}%)")

        dummy = torch.rand(4, 3, 224, 224, device=device)  # rand: [0,1]-ish range
        with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=torch.bfloat16):
            logits = model(dummy)
            loss = logits.float().pow(2).mean()

        assert loss.requires_grad, f"[{label}] loss disconnected from trainable params"
        loss.backward()

        if kwargs.get("null_space"):
            # Simulate one optimizer step to verify the LoRA-Null mechanism.
            # Must confirm the projection actually drives all the A matrices' 
            # components along the protected subspace back to ~0 (it must run too, obviously).
            with torch.no_grad():
                for p in model.parameters():
                    if p.requires_grad and p.grad is not None:
                        p -= 0.1 * p.grad

            max_component_before = 0.0
            for name, module in model.backbone.named_modules():
                if hasattr(module, "lora_A"):
                    matches = [n for n in model._null_basis_names if name.endswith(n)]
                    if matches:
                        v_top = getattr(model, "nullbasis_" + matches[0].replace(".", "_"))
                        A = module.lora_A["default"].weight
                        component = (A @ v_top.T) @ v_top
                        max_component_before = max(max_component_before, component.abs().max().item())

            model.project_lora_null_space()

            max_component_after = 0.0
            for name, module in model.backbone.named_modules():
                if hasattr(module, "lora_A"):
                    matches = [n for n in model._null_basis_names if name.endswith(n)]
                    if matches:
                        v_top = getattr(model, "nullbasis_" + matches[0].replace(".", "_"))
                        A = module.lora_A["default"].weight
                        component = (A @ v_top.T) @ v_top
                        max_component_after = max(max_component_after, component.abs().max().item())

            print(f"[{label}] Max protected-subspace component before projection: "
                  f"{max_component_before:.6f}, after: {max_component_after:.6f}")
            assert max_component_after < 1e-5, (
                f"[{label}] protected component was not zeroed out by the projection, "
                f"LoRA-Null constraint is not actually being enforced"
            )

        print(f"[{label}] Forward + backward pass successful. logits.shape={logits.shape}\n")
    