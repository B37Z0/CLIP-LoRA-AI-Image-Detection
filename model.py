"""
CLIP-ViT + LoRA detector for the AI-generated-image-detection project.

LoRA targets q_proj/v_proj inside vision_model.encoder.layers — these are
real nn.Linear modules in transformers' CLIPAttention (unlike open_clip's
nn.MultiheadAttention, which bypasses LoRA's forward hook; see the GPU
feasibility test for why that matters).

Ohja et al. 2023:
- A frozen CLIP-ViT with only a linear-probe generalized better than 
  fine-tuning. This should be tested as the most important baseline.
- Compare to all LoRA-based configurations to see if naive adaptation
  specifically suffers generalization drops.

Ablations:
- The frequency branch expects raw pixel values in [0,1] RGB so the 
  normalization needs to be inverted for the CNN input.
"""

import torch
import torch.nn as nn

# OpenAI CLIP standard normalization stats (to invert CLIPImageProcessor)
# back to ~[0,1] RGB for the frequency branch (expects raw pixel values).
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def compute_lora_null_init(backbone, target_modules, calibration_pixel_values, r, protect_k, device):
    """
    Computes the LoRA-Null initialization (Tang et al. 2025). 
    Per-layer, for each target module:
    
    1.  Capture input activations X_pre from the representative calibration
        data (forward hook on frozen backbone). Paper stores X_pre as 
        (in_features -> d_in, num_samples -> B*L); here activations are stored as
        the transpose X_pre^T (num_samples, in_features) - the mapping is modified.
    
    2.  SVD(X_pre) = U Σ V^T. U_null = r columns of U with the smallest 
        singular values ("left null space of X_pre" - U_null^T @ X_pre ≈ 0).
        
        For the transposed activations:
        SVD(X_pre^T) = (U Σ V^T)^T = V Σ U^T, so U' = V and V'^T = U^T -
        the ROWS of V'^T (i.e. Vh from torch.linalg.svd) are the COLUMNS
        of the paper's U. Singular values from torch.linalg.svd are descending,
        so the smallest singular values are the last rows of Vh: 
        U_null = Vh[-r:; :].T should equal the paper's U_null, with shape
        (in_features, r).

    3.  Project frozen weights W0 onto the null space: 
        W_proj = W0 @ U_null @ U_null^T for shape (out_features, in_features);
        we also observe rank(W_proj) <= rank(U_null) = r.

    4.  Initialize B and A using the SVD(W_proj) = U'Σ'(V')^T, where U' and V'
        are orthogonal matrices of left and right singular vectors, respectively, 
        and Σ' is a diagonal matrix of singular values. Half of the singular values
        in Σ' are handed to each of A and B:
            B = U'[:, :r] @ sqrt(Σ'[:r])   (out_features, r)
            A = sqrt(Σ'[:r]) @ V'^T[:r, :] (r, in_features)
        so that B @ A reconstructs W_proj up to the truncated rank in W_proj (<= r).

    5.  The frozen residual is set to W0' = W0 - scaling*(B @ A) (scaling 
        from LoRA via PEFT - just set to 1 by default). Ultimately, the initial 
        sum W0' + scaling*(B @ A) = W0, an alternative to LoRA's 0-init of B to
        start the model with the same behavior as the pretrained weights.
            - Verified in __main__ below.

    Returns (init_data, protect_bases):
        init_data[name] = (A_init, B_init, W0_original)
        protect_bases[name] = (protect_k, in_features) 
            - Top-k "most-used" activation directions (principal subspace of X_pre)
              used only for the train-time gradient projection ablation - not part
              of LoRA-Null. 
    """
    activations = {name: [] for name in target_modules}
    original_weights = {}
    hooks = []

    def make_hook(name):
        def hook(module, inputs, output):
            # inputs[0] shape = (batch, seq_len, in_features). Flatten the batch
            # and sequence dims so each token position is one calibration
            # sample - matching the "B*L samples" in the paper.
            activations[name].append(inputs[0].detach().reshape(-1, inputs[0].shape[-1]))
        return hook

    for name, module in backbone.named_modules():
        if name in target_modules:
            hooks.append(module.register_forward_hook(make_hook(name)))
            original_weights[name] = module.weight.data.clone()

    backbone.eval()
    with torch.no_grad():
        backbone(pixel_values=calibration_pixel_values.to(device))

    for h in hooks:
        h.remove()

    init_data, protect_bases = {}, {}
    for name, acts in activations.items():
        act_matrix = torch.cat(acts, dim=0) # (num_tokens, in_features) = X_pre^T
        _, _, Vh = torch.linalg.svd(act_matrix, full_matrices=False)

        k = min(protect_k, Vh.shape[0])
        protect_bases[name] = Vh[:k, :].clone() # top-k: principal subspace of X_pre

        r_eff = min(r, Vh.shape[0])
        u_null = Vh[-r_eff:, :].T.clone() # (in_features, r) - U_null

        W0 = original_weights[name] # (out_features, in_features)
        w_proj = (W0 @ u_null) @ u_null.T # project W0 onto null space

        U_svd, S_svd, Vh_svd = torch.linalg.svd(w_proj, full_matrices=False)
        sqrt_s = S_svd[:r_eff].sqrt()
        B_init = U_svd[:, :r_eff] * sqrt_s.unsqueeze(0) # (out_features, r)
        A_init = sqrt_s.unsqueeze(1) * Vh_svd[:r_eff, :] # (r, in_features)

        init_data[name] = (A_init, B_init, W0)

    return init_data, protect_bases


def apply_lora_null_init(peft_backbone, init_data):
    """
    Writes joint (A, B) init into each PEFT LoRA layer and adjust the 
    frozen weight to W0' = W0 - B @ A. 
    """
    for name, module in peft_backbone.named_modules():
        matches = [l_name for l_name in init_data if name.endswith(l_name)]
        # Skip modules w/o LoRA weights or that are not in the null space.
        if not matches or not hasattr(module, "lora_A"):
            continue
        A_init, B_init, W0 = init_data[matches[0]]
        scaling = module.scaling["default"]

        with torch.no_grad():
            module.lora_A["default"].weight.data = A_init.clone()
            module.lora_B["default"].weight.data = B_init.clone()
            module.base_layer.weight.data = W0 - scaling * (B_init @ A_init)


def sample_calibration_batch(dataset, n=64, device="cuda"):
    """
    Pulls n real-image samples from TinyGenImageDataset for LoRA-Null
    calibration. get_generator gets and only decodes the n images with
    real labels. Real images are used to roughly approximate the CLIP
    original distribution - sadly I don't have access to the original
    dataset used for pretraining.
    """
    real_indices = []
    for i in range(len(dataset)):
        if dataset.get_generator(i) == "Real":
            real_indices.append(i)
            if len(real_indices) >= n:
                break
    images = torch.stack([dataset[i][0] for i in real_indices])
    return images.to(device)


class CLIPLoRADetector(nn.Module):
    def __init__(self, model_id="openai/clip-vit-large-patch14", r=8, lora_alpha=8,
                 use_lora=True, use_freq_branch=False, freq_dim=128, lora_dropout=0.0,
                 null_space_init=False, null_space_grad_protect=False, protect_k=64,
                 calibration_pixel_values=None):
        super().__init__()
        from transformers import CLIPVisionModelWithProjection

        backbone = CLIPVisionModelWithProjection.from_pretrained(model_id)

        self.null_space_grad_protect = null_space_grad_protect and use_lora
        protect_bases = {}

        if use_lora:
            from peft import LoraConfig, get_peft_model
            target_modules = [
                name for name, module in backbone.named_modules()
                if name.endswith(("q_proj", "v_proj")) and isinstance(module, nn.Linear)
            ]

            init_data = {}
            if null_space_init or null_space_grad_protect:
                assert calibration_pixel_values is not None, (
                    "null_space_init/null_space_grad_protect requires calibration_pixel_values \
                    (batch of real images) to compute activations"
                )
                device = calibration_pixel_values.device
                backbone = backbone.to(device)
                init_data, protect_bases = compute_lora_null_init(
                    backbone, target_modules, calibration_pixel_values, r, protect_k, device
                )

            lora_config = LoraConfig(
                r=r, lora_alpha=lora_alpha, target_modules=target_modules,
                lora_dropout=lora_dropout, bias="none",
                init_lora_weights=not null_space_init, # skip PEFT default init
            )
            self.backbone = get_peft_model(backbone, lora_config)

            if null_space_init:
                # LoRA-Null (Tang et al. 2025): init B, A from the null-space 
                # projection of W0 and adjust the frozen weights to the residuals
                # so the initial model output exactly matches the pretrained.
                apply_lora_null_init(self.backbone, init_data)

            if self.null_space_grad_protect:
                self._protect_basis_names = list(protect_bases.keys())
                for name, v_top in protect_bases.items():
                    self.register_buffer("protectbasis_" + name.replace(".", "_"), v_top)
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

        self.classifier = nn.Linear(classifier_in, 2)  # 2 classes: real, fake

    def project_lora_null_space_gradient(self):
        """
        Ablation ON TOP OF LoRA-Null init: 
        Zero the LoRA A matrices' gradient component along the layers'
        top-k most used activation directions before optimizer.step().
        Not only do A's rows start out in the null space of X_pre; A's 
        rows are repeatedly projected to the null space throughout 
        training, protecting that subspace. Far more radical than LoRA-Null.
        - Should be done before optimizer.step() to keep the AdamW optimizer
          internal state consistent with the weights.
        """
        if not self.null_space_grad_protect:
            return
        for name, module in self.backbone.named_modules():
            matches = [orig for orig in self._protect_basis_names if name.endswith(orig)]
            if not matches or not hasattr(module, "lora_A"):
                continue
            v_top = getattr(self, "protectbasis_" + matches[0].replace(".", "_"))
            A = module.lora_A["default"].weight
            if A.grad is not None:
                with torch.no_grad():
                    component = (A.grad @ v_top.T) @ v_top
                    A.grad -= component

    def forward(self, pixel_values):
        image_embeds = self.backbone(pixel_values=pixel_values).image_embeds

        if self.use_freq_branch:
            # Invert CLIP normalization to recover [0,1] RGB approx.
            # clamp() to guard against small out of range values.
            raw_rgb = (pixel_values * self.clip_std + self.clip_mean).clamp(0, 1)
            freq_features = self.frequency_branch(raw_rgb)
            combined = torch.cat([image_embeds, freq_features], dim=1)
            return self.classifier(combined)

        return self.classifier(image_embeds)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Placeholder calibration batch to verify LoRA-Null and grad_protect
    # mechanisms. Check the hook-capture, SVD, init, and projection.
    torch.manual_seed(0)
    calibration = torch.rand(32, 3, 224, 224, device=device)

    configs = [
        ("LoRA", dict(use_lora=True, use_freq_branch=False)),
        ("frozen linear-probe", dict(use_lora=False, use_freq_branch=False)),
        ("LoRA + frequency branch", dict(use_lora=True, use_freq_branch=True)),
        ("LoRA-Null", dict(
            use_lora=True, use_freq_branch=False,
            null_space_init=True, calibration_pixel_values=calibration)),
        ("LoRA-Null + grad-protect", dict(
            use_lora=True, use_freq_branch=False,
            null_space_init=True, null_space_grad_protect=True,
            calibration_pixel_values=calibration)),
        ("LoRA-Null + frequency branch", dict(
            use_lora=True, use_freq_branch=True,
            null_space_init=True, calibration_pixel_values=calibration)),
        ("LoRA-Null + frequency branch + grad-protect", dict(
            use_lora=True, use_freq_branch=True,
            null_space_init=True, null_space_grad_protect=True,
            calibration_pixel_values=calibration)),
    ]

    for label, kwargs in configs:
        model = CLIPLoRADetector(**kwargs).to(device)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"[{label}] Trainable params: {trainable:,} / {total:,} "
              f"({100 * trainable / total:.3f}%)")

        if kwargs.get("null_space_init"):
            # Simulate optimizer step to verify LoRA-Null mechanism.
            # The output at initialization must match the untouched
            # pretrained backbone. Compared in fp32 to be strict.
            ref_backbone = CLIPLoRADetector(use_lora=False).to(device)
            probe_input = torch.rand(4, 3, 224, 224, device=device)
            with torch.no_grad():
                ref_embeds = ref_backbone.backbone(pixel_values=probe_input).image_embeds
                lora_null_embeds = model.backbone(pixel_values=probe_input).image_embeds
            max_diff = (ref_embeds - lora_null_embeds).abs().max().item()
            print(f"[{label}] Max |output diff| against untouched pretrained backbone: "
                  f"{max_diff:.8f} (should be ~0 if weights are preserved)")
            assert max_diff < 1e-3, (
                f"[{label}] LoRA-Null init does NOT preserve the pretrained model's "
                f"output - something is wrong :("
            )

            # Secondary check; A's rows should be in the null space of X_pre.
            # Verify a near-zero component along the protected subspace to 
            # ensure the SVD split and B, A construction is correct.
            if hasattr(model, "_protect_basis_names"):
                max_init_component = 0.0
                for name, module in model.backbone.named_modules():
                    if hasattr(module, "lora_A"):
                        matches = [n for n in model._protect_basis_names if name.endswith(n)]
                        if matches:
                            v_top = getattr(model, "protectbasis_" + matches[0].replace(".", "_"))
                            A = module.lora_A["default"].weight
                            component = (A @ v_top.T) @ v_top
                            max_init_component = max(max_init_component, component.abs().max().item())
                print(f"[{label}] Max init component along protected subspace: "
                      f"{max_init_component:.8f} (should be ~0 if A constructed correctly)")
                assert max_init_component < 1e-3, f"[{label}] A's rows are NOT null-space-orthogonal"

        dummy_input = torch.rand(4, 3, 224, 224, device=device)
        with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=torch.bfloat16):
            logits = model(dummy_input)
            loss = logits.float().pow(2).mean()

        assert loss.requires_grad, f"[{label}] loss disconnected from trainable params"
        loss.backward()

        if kwargs.get("null_space_grad_protect"):
            # Also verify the protected subspace-orthogonal projection
            # before the optimizer.step().
            name0, module0 = next(
                (n, m) for n, m in model.backbone.named_modules() if hasattr(m, "lora_A")
            )
            matches = [n for n in model._protect_basis_names if name0.endswith(n)]
            v_top = getattr(model, "protectbasis_" + matches[0].replace(".", "_"))
            A = module0.lora_A["default"].weight
            grad_component_before = ((A.grad @ v_top.T) @ v_top).abs().max().item()

            model.project_lora_null_space_gradient()

            grad_component_after = ((A.grad @ v_top.T) @ v_top).abs().max().item()
            print(f"[{label}] Gradient component along protected subspace - before: "
                  f"{grad_component_before:.6f}, after: {grad_component_after:.6f}")
            assert grad_component_after < 1e-5, (
                f"[{label}] gradient projection did not zero the protected component"
            )

        print(f"[{label}] Forward + backward pass successful. logits.shape={logits.shape}\n")

"""
[LoRA] Trainable params: 787,970 / 304,754,178 (0.259%)
[LoRA] Forward + backward pass successful. logits.shape=torch.Size([4, 2])

[frozen linear-probe] Trainable params: 1,538 / 303,967,746 (0.001%)
[frozen linear-probe] Forward + backward pass successful. logits.shape=torch.Size([4, 2])

[LoRA + frequency branch] Trainable params: 824,898 / 304,791,106 (0.271%)
[LoRA + frequency branch] Forward + backward pass successful. logits.shape=torch.Size([4, 2])

[LoRA-Null] Trainable params: 787,970 / 304,754,178 (0.259%)
[LoRA-Null] Max |output diff| against untouched pretrained backbone: 0.00000954 (should be ~0 if weights are preserved)
[LoRA-Null] Forward + backward pass successful. logits.shape=torch.Size([4, 2])

[LoRA-Null + grad-protect] Trainable params: 787,970 / 304,754,178 (0.259%)
[LoRA-Null + grad-protect] Max |output diff| against untouched pretrained backbone: 0.00001049 (should be ~0 if weights are preserved)
[LoRA-Null + grad-protect] Max init component along protected subspace: 0.00000204 (should be ~0 if A constructed correctly)
[LoRA-Null + grad-protect] Gradient component along protected subspace - before: 0.013021, after: 0.000001
[LoRA-Null + grad-protect] Forward + backward pass successful. logits.shape=torch.Size([4, 2])

[LoRA-Null + frequency branch] Trainable params: 824,898 / 304,791,106 (0.271%)
[LoRA-Null + frequency branch] Max |output diff| against untouched pretrained backbone: 0.00000858 (should be ~0 if weights are preserved)
[LoRA-Null + frequency branch] Forward + backward pass successful. logits.shape=torch.Size([4, 2])

[LoRA-Null + frequency branch + grad-protect] Trainable params: 824,898 / 304,791,106 (0.271%)
[LoRA-Null + frequency branch + grad-protect] Max |output diff| against untouched pretrained backbone: 0.00000763 (should be ~0 if weights are preserved)
[LoRA-Null + frequency branch + grad-protect] Max init component along protected subspace: 0.00000204 (should be ~0 if A constructed correctly)
[LoRA-Null + frequency branch + grad-protect] Gradient component along protected subspace - before: 0.002898, after: 0.000000
[LoRA-Null + frequency branch + grad-protect] Forward + backward pass successful. logits.shape=torch.Size([4, 2])
"""