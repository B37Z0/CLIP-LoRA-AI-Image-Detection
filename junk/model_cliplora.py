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
"""

import torch
import torch.nn as nn


class CLIPLoRADetector(nn.Module):
    def __init__(self, model_id="openai/clip-vit-large-patch14", r=8, lora_alpha=8, use_lora=True):
        super().__init__()
        from transformers import CLIPVisionModelWithProjection
        backbone = CLIPVisionModelWithProjection.from_pretrained(model_id)
 
        if use_lora:
            from peft import LoraConfig, get_peft_model
            target_modules = [
                name for name, module in backbone.named_modules()
                if name.endswith(("q_proj", "v_proj")) and isinstance(module, nn.Linear)
            ]
            lora_config = LoraConfig(
                r=r, lora_alpha=lora_alpha, target_modules=target_modules,
                lora_dropout=0.0, bias="none",
            )
            self.backbone = get_peft_model(backbone, lora_config)
        else:
            for p in backbone.parameters():
                p.requires_grad = False
            self.backbone = backbone
 
        # image_embeds dimension: 768 for ViT-L/14, 512 for ViT-B/32
        embed_dim = self.backbone.config.projection_dim
        self.classifier = nn.Linear(embed_dim, 2) # binary classifier head
 
    def forward(self, pixel_values):
        image_embeds = self.backbone(pixel_values=pixel_values).image_embeds
        return self.classifier(image_embeds)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPLoRADetector().to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.3f}%)")

    dummy_input = torch.randn(4, 3, 224, 224, device=device)
    with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=torch.bfloat16):
        logits = model(dummy_input)
        loss = logits.float().pow(2).mean()

    assert loss.requires_grad, "loss disconnected from LoRA params; check target_modules"
    loss.backward()
    print(f"Forward + backward pass successful. logits.shape={logits.shape}")

"""
Trainable params: 787,970 / 304,754,178 (0.259%)
Forward + backward pass successful. logits.shape=torch.Size([4, 2])
"""