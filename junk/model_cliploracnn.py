"""
CLIP-ViT + LoRA AI-generated-image-detector.

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
"""

import torch
import torch.nn as nn

# OpenAI CLIP standard normalization stats (to invert CLIPImageProcessor)
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


class CLIPLoRADetector(nn.Module):
    def __init__(self, model_id="openai/clip-vit-large-patch14", r=4, lora_alpha=8,
                 use_lora=True, use_freq_branch=False, freq_dim=128):
        super().__init__()
        from transformers import CLIPVisionModelWithProjection  # deferred import
 
        backbone = CLIPVisionModelWithProjection.from_pretrained(model_id)
 
        if use_lora:
            from peft import LoraConfig, get_peft_model
            target_modules = [
                name for name, module in backbone.named_modules()
                if name.endswith(("q_proj", "v_proj")) and isinstance(module, nn.Linear)
            ]
            lora_config = LoraConfig(
                r=r, lora_alpha=lora_alpha, target_modules=target_modules,
                lora_dropout=0.1, bias="none",
            )
            self.backbone = get_peft_model(backbone, lora_config)
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
    ]
 
    for label, kwargs in configs:
        model = CLIPLoRADetector(**kwargs).to(device)
 
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"[{label}] Trainable params: {trainable:,} / {total:,} "
              f"({100 * trainable / total:.3f}%)")
 
        dummy = torch.rand(4, 3, 224, 224, device=device)
        with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=torch.bfloat16):
            logits = model(dummy)
            loss = logits.float().pow(2).mean()
 
        assert loss.requires_grad, f"[{label}] loss disconnected from trainable params"
        loss.backward()
        print(f"[{label}] Forward + backward pass successful. logits.shape={logits.shape}\n")

"""
[LoRA] Trainable params: 787,970 / 304,754,178 (0.259%)
[LoRA] Forward + backward pass successful. logits.shape=torch.Size([4, 2])

[frozen linear-probe] Trainable params: 1,538 / 303,967,746 (0.001%)
[frozen linear-probe] Forward + backward pass successful. logits.shape=torch.Size([4, 2])

[LoRA + frequency branch] Trainable params: 824,898 / 304,791,106 (0.271%)
[LoRA + frequency branch] Forward + backward pass successful. logits.shape=torch.Size([4, 2])
"""