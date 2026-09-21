# AI-Generated Image Detection with CLIP-ViTs

Detecting AI-generated images by adapting a pretrained CLIP vision transformer, with a focus on *generalization to unseen generators* 

> **Status:** active work-in-progress. Results below are from a curated subset (Tiny-GenImage); scaling to the full GenImage dataset is an eventual next step (see [Limitations & Next Steps](#limitations--next-steps)).

## Motivation

Recent AI-generated image (AIGI) detection literature (Ojha et al. 2023; others investigated in Mahara & Rishe 2025) shows that classifiers built on **pretrained multi-modal vision-language models (VLMs)** - adapted only lightly via linear probe or LoRA (and variants), rather than trained from scratch, generalize far better across generative models including GANs and diffusion models than other methods. This project builds and empirically tests that architecture family, specifically investigating how naive adaptation trades away generalization.

## Task & Evaluation

The task is simple binary classification: real images vs. AI-generated images. Training holds out two generators (**Midjourney**, **VQDM**) entirely from the training data, and evaluates generalization across them specifically. At the moment, because the Tiny-GenImage held-out test split is skewed with ~78% real images, results are reported as **balanced accuracy** (average of real-image recall and each held-out generator's fake-image recall) per generator - not raw accuracy which would be misleading for the imbalanced split.

**Dataset:** [Tiny-GenImage](https://huggingface.co/datasets/TheKernel01/Tiny-GenImage) (28k train / 7k val, 8 generators + real images); in the future the full [GenImage](https://arxiv.org/abs/2306.08571) benchmark.

## Architectures

| Model | Description |
|---|---|
| **Dual-stream CNN** (`baseline_cnn.py`) | Semi-reproduction baseline (Yousaf et al. 2022 pattern): ResNet18 spatial stream + a frequency stream (YCbCr conversion -> per-channel DFT real/imaginary + single-level Haar DWT, each through a separate small CNN), then fused via concatenation. |
| **Frozen CLIP linear probe** (`model.py`) | Ojha et al. 2023 baseline: frozen CLIP ViT-L/14 backbone, single linear classification head. No finetuning/adaptation. |
| **CLIP + LoRA** (`model.py`) | CLIP ViT-L/14 with LoRA adapters on attention `q_proj`/`v_proj` layers, via `transformers.CLIPVisionModelWithProjection` + `peft`. |
| **CLIP + LoRA-Null** (`model.py`) | LoRA with an added constraint: after every optimizer step, each adapter's `A` matrix is projected to be orthogonal to the frozen weight's top-32 singular directions (computed via SVD), preventing adaptation from messing with the input-side features that the pretrained weights rely on most for generalization. |
| **+ Frequency branch** (ablation on any CLIP variant) | Fuses the same YCbCr DFT+DWT frequency branch from the CNN baseline into the CLIP model's classifier head, testing whether frequency information adds anything on top of CLIP's semantic features (drawing a parallel to the dual-stream CNN, CLIP image processing is fundamentally rooted in the spatial domain). |

A few implementation notes:
- LoRA is built on `transformers.CLIPVisionModelWithProjection`, not `open_clip` - `open_clip`'s ViT uses `torch.nn.MultiheadAttention`, which reads its `out_proj` weight directly for a fused functional call, rather than invoking it as a normal forward pass (silently and entirely bypassing LoRA's hook).
- The dual-stream CNN baseline is a *largely simplified* reproduction of Yousaf et al.: DFT and DWT features are processed by separate small CNNs rather than stacked into one 18-channel input to a single ResNet-50, and streams are fused via feature concatenation rather than the paper's probability-averaging. Future improvements are planned.

## Results

Best mean balanced accuracy across held-out generators (Midjourney + VQDM), by configuration:

| Configuration | Mean Balanced Acc | Midjourney Balanced Acc | VQDM Balanced Acc |
|---|---|---|---|
| Dual-stream CNN (baseline) | 0.642 | 0.649 | 0.634 |
| Frozen CLIP linear probe | 0.853 | 0.844 | 0.863 |
| CLIP + LoRA (r=8) | 0.843* | 0.750* | 0.936* |
| CLIP + LoRA + frequency branch | 0.835* | 0.732* | 0.937* |
| CLIP + LoRA-Null (r=8) | 0.857 | 0.751 | **0.964** |
| **CLIP + LoRA-Null + frequency branch** | **0.858** | **0.770** | 0.946 |

*Plain-LoRA and LoRA+frequency results vary noticeably across epochs/runs - see [Findings](#findings) below.

## Findings

- **CLIP-based approaches dramatically outperform training a detector from scratch.** Every single CLIP configuration easily beats the dual-stream CNN baseline by 20+ points of balanced accuracy, readily confirming the literature's claim for this specific generalization-focused evaluation.
- **Naive LoRA adaptation rapidly overfits to training-generator-specific artifacts.** Across every run, training loss collapses toward zero within 2–3 epochs while held-out Midjourney accuracy degrades or becomes noisy. These are classic signs of the model overfitting to narrow, generator-specific shortcuts rather than learning generalizable representations. Standard regularization (LoRA dropout, lower rank) did not fix this to any notable degree.
- **LoRA-Null (SVD-based null-space projection) measurably helps.** Implementing LoRA-Null achieved a clean and reproducible result (0.857) with more stable early-training generalization than the basic LoRA variant. The overfitting problem was not solved - overfitting returned after epoch 2-3 - likely because constraining ~32 of 1024 weight dimensions still leaves the rank-8 update plenty of room to memorize elsewhere.
- **The frequency branch has a negative or marginal effect.** Adding the frequency branch to the LoRA-Null variant improved Midjourney balanced accuracy by +2 points (0.751 -> 0.770) while hurting VQDM about the same (0.964 -> 0.946), resulting in essentially the same mean balanced accuracy (0.858 vs 0.857). Under plain LoRA (without null-space projection), the frequency branch effects were more negative than not. An interpretation of these effects is that the frequency features may only contribute meaningfully when the backbone's adaptation is constrained enough to not overfit before the randomly-initialized frequency CNN can learn useful filters.
- **Generalization difficulty is very much generator-dependent, not uniform.** Midjourney is consistently and easily the hardest of the held-out generators across every architecture and configuration tried (accuracy regularly stuck in the 0.3–0.55 range), while VQDM generalizes well and improves steadily under adaptation until it begins overfitting. This is consistent with literature suggesting closed-source commercial generators (Midjourney, DALL·E) leave different or weaker artifacts than open research diffusion models, making them harder universal detection targets.

## Limitations & Next Steps

- **Dataset scale.** Tiny-GenImage provides only ~1,750–2,000 fake images per generator. The full GenImage dataset (tens of thousands of images per generator) is the natural next step up - it's plausible the overfitting/generalization gap observed so far is partly due to data-scarcity rather than a purely architectural issue.
- **Frequency branch investigation.** Current results are a negative finding and far from settled. It is worth testing whether pretraining the frequency branch separately, or applying a distinct learning rate may change the outcome.
- **LoRA-Null tuning.** The top-k=32 protected-subspace size was a starting guess and not tuned. A wider protected subspace, or applying the constraint to additional layers, may extend the generalization benefit further into training.
- **Formal LoRA-Null ablation.** Comparing against a matched-capacity unconstrained LoRA configuration at the same effective rank would reinforce that the null-space constraint, not just training dynamics, led to the improvement.

## Repository Contents

```
tinygenimage_dataset.py   # Tiny-GenImage loading + cross-generator train/test split
model.py                  # CLIP-ViT detector: frozen probe / LoRA / LoRA-Null / +frequency branch
train.py                  # Training loop for CLIP-based models, with per-generator + balanced-acc evaluation
baseline_cnn.py           # Dual-stream CNN baseline (ResNet18 + DFT/DWT frequency branch)
train_baseline_cnn.py     # Training loop for the dual-stream CNN baseline
```

## References

- Ojha, U., Li, Y., & Lee, Y. J. (2023). *Towards Universal Fake Image Detectors that Generalize Across Generative Models.* [arXiv:2302.10174](https://arxiv.org/abs/2302.10174)
- Mahara, A., & Rishe, N. (2025). *Methods and Trends in Detecting AI-Generated Images: A Comprehensive Review.* [arXiv:2502.15176](https://arxiv.org/abs/2502.15176)
- Yousaf, B., Usama, M., Sultani, W., Mahmood, A., & Qadir, J. (2022). *Fake Visual Content Detection Using Two-Stream Convolutional Neural Networks.* [arXiv:2101.00676](https://arxiv.org/abs/2101.00676)
- Zhu, M. et al. (2024). *GenImage: A Million-Scale Benchmark for Detecting AI-Generated Images.* [arXiv:2306.08571](https://arxiv.org/abs/2306.08571)
- Hu, E. J. et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- Wang, G. et al. (2025). *LoRA-Null: Zero-Cost Adapting CLIP for Few-Shot Image Classification.* [arXiv:2503.02659](https://arxiv.org/abs/2503.02659)
- Ma, Z. et al. (2025). *AIGI Holmes: A Multi-Branch Pipeline for Reliable AI-Generated Image Detection.* [arXiv:2507.02664](https://arxiv.org/abs/2507.02664)
- Yang, C., Zhao, Y., Wang, S. (2019). *Deep Image Compression in the Wavelet Transform Domain Based on High Frequency Sub-Band Prediction. IEEE Access.* [10.1109/ACCESS.2019.2911403](https://doi.org/10.1109/ACCESS.2019.2911403)
