# AI-Generated Image Detection with CLIP-ViTs

Detecting AI-generated images by adapting a pretrained CLIP vision transformer, with a focus on *generalization to unseen generators*. 

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
| **CLIP + LoRA-Null** (`model.py`) | Full reproduction (Tang et al. 2025) of LoRA-Null initialization: for each adapted layer, pretrained activations are captured on the frozen backbone, SVD'd to find their null space, and the frozen weight `W0` is projected onto that null space; a second SVD of that projection jointly initializes `A` and`B`, with the frozen residual adjusted to `W0' = W0 - scaling*(B@A)` so the model output at initialization still matches the untouched pretrained backbone (verified directly - see [Findings](#findings)). |
| **+ grad-protect** (ablation on LoRA-Null) | Personal extension, not part of published methods: on top of LoRA-Null's init, `A`'s gradient is projected to zero out its component along the top-k most-used activation directions before every optimizer step. Where LoRA-Null only constrains *initialization* of `A` and `B`, this keeps `A` constrained to the null space *throughout* training. |
| **+ Frequency branch** (ablation on CLIP) | Fuses the same YCbCr DFT+DWT frequency branch from the CNN baseline into the CLIP model's classifier head, testing whether frequency information adds anything on top of CLIP's semantic features (drawing a parallel to the dual-stream CNN, CLIP image processing is fundamentally rooted in the spatial domain). |
 
A few implementation notes:
- LoRA is built on `transformers.CLIPVisionModelWithProjection`, not `open_clip` - `open_clip`'s ViT uses `torch.nn.MultiheadAttention`, which reads its `out_proj` weight directly for a fused functional call, rather than invoking it as a normal forward pass (silently and entirely bypassing LoRA's hook).
- The dual-stream CNN baseline is a *largely simplified* reproduction of Yousaf et al.: DFT and DWT features are processed by separate small CNNs rather than stacked into one 18-channel input to a single ResNet-50, and streams are fused via feature concatenation rather than the paper's probability-averaging. Future improvements are planned.

## Results

Best mean balanced accuracy across held-out generators (Midjourney + VQDM), by configuration:
 
| Configuration | Mean Balanced Acc | Midjourney Balanced Acc | VQDM Balanced Acc |
|---|---|---|---|
| Dual-stream CNN (baseline) | 0.642 | 0.649 | 0.634 |
| Frozen CLIP linear probe | 0.842 | 0.835 | 0.850 |
| CLIP + LoRA (r=8) | 0.879 | 0.796 | 0.963 |
| CLIP + LoRA + frequency branch | 0.835* | 0.732* | 0.937* |
| CLIP + LoRA-Null (r=8) | 0.878 | 0.794 | 0.961 |
| CLIP + LoRA-Null + frequency branch | - | - | - |
| **CLIP + LoRA-Null + grad-protect** | **0.913** | **0.848** | **0.977** |
| CLIP + LoRA-Null + grad-protect + frequency branch | - | - | - |
 
*LoRA+frequency has not yet been rerun under the same fixed-seed as the other rows and should be treated as a loose range- see [Findings](#findings) below.

The raw per-class recall underlying the balanced-accuracy numbers (i.e. real-image recall, and each held-out generator's raw fake-image recall before averaging with real recall) is reported below as well for reference.
 
| Configuration | Real Recall | Midjourney Recall (raw) | VQDM Recall (raw) |
|---|---|---|---|
| Dual-stream CNN (baseline) | 0.981 | 0.318 | 0.287 |
| Frozen CLIP linear probe | 0.963 | **0.707** | 0.737 |
| CLIP + LoRA (r=8) | 0.999 | 0.592 | 0.926 |
| CLIP + LoRA-Null (r=8) | 0.999 | 0.589 | 0.923 |
| CLIP + LoRA-Null + frequency branch | - | - | - |
| CLIP + LoRA-Null + grad-protect | 0.995 | 0.701 | **0.959** |
| CLIP + LoRA-Null + grad-protect + frequency branch | - | - | - |

Raw recall is not yet reported for the LoRA+frequency-branch configuration.

## Findings

- **CLIP-based approaches dramatically outperform training a detector from scratch.** Every single CLIP configuration easily beats the dual-stream CNN baseline by 20+ points of balanced accuracy, readily confirming the literature's claim for this specific generalization-focused evaluation.
- **Naive LoRA adaptation rapidly overfits to training-generator-specific artifacts.** Under a fixed seed and prioritizing the best balanced accuracy, plain LoRA peaks at epoch 2 (mean balanced acc 0.879) and then degrades every epoch after (0.824 -> 0.801 -> 0.829 by epoch 5) while training loss collapses toward zero (0.061 -> 0.0055 -> 0.0026 -> 0.0019 -> 0.0012). Midjourney's raw recall shows the same pattern most sharply: 0.469 -> 0.592 (peak) -> 0.439 -> 0.395 -> 0.430. These are classic signs of the model overfitting to narrow, generator-specific shortcuts rather than learning generalizable representations. Standard regularization (LoRA dropout, lower rank) did not fix this to any notable degree.
- **Matched-capacity comparison: the null-space initialization alone provides no measurable benefit over plain LoRA.** Interestingly, given the same seed and rank (=8) - plain LoRA (0.879) and LoRA-Null (0.878) land within noise of each other, most significantly Midjourney balanced accuracy (0.796 for plain LoRA vs. 0.794 for LoRA-Null). This is an unexpected and someone disappointing result: constraining *where the adapter starts* to the null space of pretrained activations has not, by itself, impacted peak generalization in this study. Only when a similar constraint is enforced *continuously* throughout training (grad-protect, 0.913) does a meaningful gain over LoRA appear - see the next point. 

- **The LoRA-Null init doesn't close the Midjourney gap to the frozen probe, and neither does plain LoRA.** Both LoRA variants beat the frozen probe (0.842 mean) on mean balanced accuracy, but almost entirely via the large VQDM gain (frozen probe raw recall 0.737 vs. ~0.92-0.93 for both LoRA variants); on Midjourney specifically - the harder held-out generator - the frozen probe clearly generalizes better (raw recall 0.707 vs. ~0.59 for both) and maintains a higher recall over epochs. Comparatively, Midjourney's raw recall under either LoRA variant is unstable (oscillating, no clear trend across epochs) rather than improving smoothly until overfitting like the frozen probe. LoRA tuning, null-space initialization or not, does not appear to be sufficient to match frozen-feature generalization on the hardest present case.
- **Continuously constraining training (grad-protect), not just initialization, significantly improves generalization.** Projecting `A`'s gradient to stay orthogonal to the top-k most-used activation directions at every step - not just at init - raised Midjourney's raw recall from 0.589 to 0.701 (within 1 point of the frozen probe), while also achieving the best VQDM raw recall (0.959) and best mean balanced accuracy (0.913) of any configuration tested *by far*. This is an extension beyond the published LoRA-Null method, and the result suggests the initialization-only method used is only a floor for this generalization-focused task. Continuing to protect the subspace during training resulting in significantly improved generalization; however, the rapid overfitting pattern is still present (Midjourney accuracy peaks at epoch 2-3, then degrades consistently). 
- **The frequency branch's effect varies between configurations.** Adding the frequency branch to plain LoRA (without null-space projection) as well as various test configurations resulted in only marginal improvements or degradations in performance. An interpretation of these effects is that the frequency features may only contribute meaningfully when the backbone's adaptation is constrained enough to not overfit before the randomly-initialized frequency CNN can learn useful filters.
- **Generalization difficulty is very much generator-dependent, not uniform.** Midjourney is consistently and easily the hardest of the held-out generators across every architecture and configuration tried, while VQDM generalizes well and improves steadily under adaptation until it begins overfitting. This is consistent with literature suggesting closed-source commercial generators (Midjourney, DALL·E) leave different or weaker artifacts than open research diffusion models, making them harder universal detection targets.

## Limitations & Next Steps

- **Dataset scale.** Tiny-GenImage provides only ~1,750–2,000 fake images per generator. The full GenImage dataset (tens of thousands of images per generator) is the natural next step up - it's plausible the overfitting/generalization gap observed so far is partly due to data-scarcity rather than a purely architectural issue.
- **Frequency branch ablations on LoRA-Null variants.** Further combining the ablations to verify the full potential of the frequency branch is needed. The possibility of independently pretraining the frequency branch or applying a distinct learning rate may also be investigated.
- **grad-protect's overfitting dynamic.** grad-protect raises peak Midjourney generalization but still degrades after epoch 2-3. Worth testing whether a wider protected subspace (`protect_k`), applying the projection to additional layers, or a lower learning rate late in training extends the improved-generalization window further.
- **Matched-capacity ablation - done for a single seed but worth repeating.** Plain LoRA and LoRA-Null's init-only variant were compared *once* at the same rank and seed, given best-epoch-selection; the twolanded within noise of each other. Re-running both across different seeds and hyperparameter configurations would reinforce the result and definitively confirm any true gap.

## Repository Contents

```
tinygenimage_dataset.py   # Tiny-GenImage loading + cross-generator train/test split
model.py                  # CLIP-ViT detector: frozen probe / LoRA / LoRA-Null / grad-protect / +frequency branch
train.py                  # Training loop for CLIP-based models, with per-generator + balanced-acc evaluation
baseline_cnn.py           # Dual-stream CNN baseline (ResNet18 + DFT/DWT frequency branch)
train_baseline_cnn.py     # Training loop for the dual-stream CNN baseline
```

## References

- Ojha, U. et al. (2023). *Towards Universal Fake Image Detectors that Generalize Across Generative Models.* [arXiv:2302.10174](https://arxiv.org/abs/2302.10174)
- Mahara, A. & Rishe, N. (2025). *Methods and Trends in Detecting AI-Generated Images: A Comprehensive Review.* [arXiv:2502.15176](https://arxiv.org/abs/2502.15176)
- Yousaf, B. et al. (2022). *Fake Visual Content Detection Using Two-Stream Convolutional Neural Networks.* [arXiv:2101.00676](https://arxiv.org/abs/2101.00676)
- Zhu, M. et al. (2024). *GenImage: A Million-Scale Benchmark for Detecting AI-Generated Images.* [arXiv:2306.08571](https://arxiv.org/abs/2306.08571)
- Hu, E. J. et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- Tang, P. et al. (2026). *Put the Space of LoRA Initialization to the Extreme to Preserve Pre-trained Knowledge.* AAAI 2026. [arXiv:2503.02659](https://arxiv.org/abs/2503.02659) | [HTML (v1)](https://arxiv.org/html/2503.02659v1)
- Zhang, Y. et al. (2025). *Null-LoRA: Low-Rank Adaptation on Null Space.* [arXiv:2512.15233](https://arxiv.org/abs/2512.15233)
- Ma, Z. et al. (2025). *AIGI Holmes: A Multi-Branch Pipeline for Reliable AI-Generated Image Detection.* [arXiv:2507.02664](https://arxiv.org/abs/2507.02664)
- Yang, C. et al. (2019). *Deep Image Compression in the Wavelet Transform Domain Based on High Frequency Sub-Band Prediction. IEEE Access.* [10.1109/ACCESS.2019.2911403](https://doi.org/10.1109/ACCESS.2019.2911403)