# Interview Notes — GLaS Gland Segmentation with U-Net

This document is a technical Q&A prepared to defend this project in an interview (e.g. for an AI/Computer Vision role in digital pathology). Each answer reflects what this specific codebase actually does — file references are given so you can pull up the exact code if asked to go deeper.

---

### 1. Why did you choose U-Net?

U-Net is the standard baseline for biomedical image segmentation, and for good structural reasons, not just convention: it's a fully-convolutional encoder-decoder with **skip connections** that directly address the two competing needs of segmentation — a large receptive field to recognize *what* a gland is (context), and precise spatial resolution to say *exactly where* its boundary is (localization). It also works well on small datasets like GlaS (85 training images) because it has no fully-connected layers and relies on convolutional weight sharing plus heavy spatial augmentation rather than needing millions of examples. It's implemented from scratch here (`src/models/unet.py`) rather than imported, specifically so I can explain every layer.

### 2. Why is segmentation different from classification?

Classification maps an entire image to a single label (or a small set of labels). Segmentation maps *every pixel* to a label — it's a dense prediction problem. That changes almost everything: the loss is computed per-pixel (and must handle spatial class imbalance), the architecture needs to preserve spatial resolution or explicitly recover it (hence encoder-decoder + skip connections, rather than a global-average-pool + FC head), data augmentation must be applied identically to the image and the label map, and evaluation metrics (Dice/IoU) measure region overlap rather than a single correct/incorrect label.

### 3. Why use Dice loss?

Dice loss directly optimizes a differentiable approximation of the Dice coefficient (`2*|A∩B| / (|A|+|B|)`), which is an overlap ratio normalized by the size of the prediction and the target. Unlike per-pixel BCE, which treats every pixel identically and can be dominated by whichever class has more pixels, Dice loss is inherently scaled by how much foreground actually exists in an image, which makes it much more robust to foreground/background imbalance. See `src/losses/dice_loss.py`.

### 4. Why combine BCE and Dice loss?

Each has a weakness the other covers. Dice loss's gradient is unstable early in training: when predictions are near-random, both the intersection and union in the Dice ratio are small, so the ratio (and its gradient) is noisy. BCE gives a smooth, well-conditioned per-pixel gradient from the very first batch, which stabilizes early optimization. Once the model has learned something, Dice's overlap-focused objective pushes harder on getting the *shape* of the segmentation right rather than just individual pixel probabilities. The combination (`BCEDiceLoss` in `src/losses/dice_loss.py`, weighted sum, default 0.5/0.5) is a standard, well-documented recipe for binary medical segmentation.

### 5. Why use IoU?

IoU (`TP / (TP+FP+FN)`) is the standard overlap metric in segmentation/detection because it's an intuitive, symmetric measure of how much predicted and ground-truth regions agree, and it's a stricter judge than Dice for the same prediction (`IoU <= Dice`, exactly `Dice = 2*IoU/(1+IoU)`). Reporting both gives a fuller picture: Dice is more forgiving of boundary noise, IoU penalizes it more, so a model that looks great on Dice but noticeably worse on IoU likely has systematic boundary imprecision rather than gross region errors.

### 6. Why must masks use nearest-neighbor interpolation?

A segmentation mask is a **categorical label map** — every pixel is exactly `0` or `1`, there is no continuum between them. Bilinear or bicubic interpolation is designed for continuous signals (like RGB intensity) and will blend neighboring pixel values, producing fractional outputs like `0.4` at every boundary. That value doesn't mean anything as a class label, and if you then binarize it (e.g. `> 0.5`), you've silently shifted every gland boundary by up to a pixel or more in an interpolation-artifact-dependent way — corrupting your ground truth before training even starts. Nearest-neighbor interpolation instead just picks the closest original label, preserving the {0,1} categorical property exactly. Images, by contrast, *are* continuous intensities, so bilinear interpolation for them is both safe and preferable (smoother resampling). See `src/data/transforms.py`.

### 7. Why should test data remain untouched during training?

The test set exists to estimate how the model performs on data it has never influenced in any way — not the training gradients, not the choice of hyperparameters, not early stopping, nothing. If test data leaks into training (directly, or indirectly through hyperparameter tuning against test performance), the reported metric stops measuring generalization and starts measuring memorization/overfitting to that specific set — you get an optimistic number that won't hold up on genuinely new data. In this project, the 80 official GlaS test images are discovered separately from the 85 training images (`src/data/split.py`), the train/val split and all tuning happens only within the 85, and `scripts/evaluate.py` touches the test set exactly once, at the end, to produce the final reported numbers.

### 8. What are skip connections doing in U-Net?

Each pooling step in the encoder increases the receptive field and channel depth but throws away precise spatial information (a max-pool literally discards 3 of every 4 activations). By the time you reach the bottleneck, you have excellent *semantic* information ("there's a gland around here") but very poor *spatial* precision (exactly where its edge is). Skip connections concatenate each encoder stage's pre-pooling feature map directly into the corresponding decoder stage, at the same spatial resolution. This gives the decoder direct access to fine spatial detail it would otherwise have no way to reconstruct, which is exactly why U-Net produces sharp boundaries instead of a blurry blob roughly where the object is. See the `Up.forward()` concatenation in `src/models/unet.py`.

### 9. What happens if glands occupy only a small percentage of the image?

This is the class imbalance problem. If foreground pixels are rare, a naive model can get very low BCE loss and even reasonably high pixel *accuracy* just by predicting "background" everywhere — accuracy is a bad metric here because it's dominated by the majority class. This is exactly why the loss uses Dice (normalized by region size, not total pixel count) and why the evaluation metrics are Dice/IoU/precision/recall computed from confusion-matrix counts rather than raw accuracy. I measured this directly on GlaS: gland pixels average ~50% of a tile in aggregate, but per-image proportion actually ranges from ~11% to ~89% across the 165 official images (`scripts/prepare_data.py` / `notebooks/exploration.ipynb`) — so while the dataset isn't globally imbalanced like, say, tumor-infiltrating lymphocyte detection, individual tiles can be, and the loss/metrics here don't assume balance either way.

### 10. How would you scale this approach to 50,000 x 50,000 pathology images?

This project trains and predicts on GlaS tiles at up to ~1500px per side — it does **not** process whole-slide images (WSIs), and the README says so explicitly. A real WSI pipeline built around the same core U-Net would look like:

1. **Whole-slide image** — load via a pyramidal format reader (e.g. OpenSlide) rather than a single in-memory array; a 50k x 50k image at full resolution is far too large to load or run through a CNN directly.
2. **Tiling** — divide the slide into fixed-size tiles (e.g. 512x512 or 1024x1024) at a chosen pyramid level/magnification, typically with **overlap** between adjacent tiles (see step 5).
3. **Tissue detection** — most of a slide is empty background/glass; run a cheap tissue-vs-background filter (Otsu thresholding on saturation, or a small classifier) first so the expensive segmentation model only runs on tiles that actually contain tissue.
4. **Patch inference** — run the trained U-Net on each tissue tile independently (this is exactly the model and preprocessing already built in this repo — the same `predict_mask` logic in `src/inference/predict.py` would apply per-tile).
5. **Overlapping tiles** — predict on overlapping tiles and blend the overlap region (e.g. average logits, or a weighted/Gaussian blend favoring tile centers) so that predictions don't have hard seams or edge artifacts at tile boundaries, where the model has the least context.
6. **Stitching** — reassemble the per-tile predictions into a single full-resolution mask aligned to the original slide coordinates.
7. **Post-processing** — clean up the stitched mask (remove small spurious regions, morphological smoothing, possibly connected-component analysis to enumerate individual glands for downstream quantitative analysis).

The model architecture and training code in this repo would not need to change for this — what changes is everything around inference: the data loading/tiling layer, a tissue detector, and a stitching/blending step.

### 11. What is patch-based inference?

Patch-based inference is running a model (trained on small, fixed-size crops) over a much larger image by sweeping a window across it — either with a stride equal to the patch size (non-overlapping tiles) or, more commonly for segmentation, a smaller stride so adjacent patches overlap. The per-patch outputs are then merged (see Q10, step 5). It's necessary whenever the full image is too large to fit in memory or exceeds the receptive field/resolution the model was trained at — which is always true for WSIs relative to a model trained on ~512px tiles.

### 12. How would you handle stain variation?

H&E staining intensity and color balance vary significantly across labs, scanners, reagent batches, and even within a single slide — this is a well-known source of domain shift in histopathology and is *not* addressed by this baseline (see README Limitations/Future Work). Standard approaches: (a) **stain normalization** at preprocessing time — e.g. Macenko or Vahadane color deconvolution/normalization to map every image toward a canonical stain appearance; (b) **stain augmentation** during training — randomly perturbing the H&E color channels (rather than just spatial transforms) so the model learns to be invariant to realistic staining variation instead of overfitting to the training set's particular color distribution; (c) training on multiple source institutions/scanners if available, so the model sees real stain variation rather than relying on synthetic augmentation alone.

### 13. How would you improve the model beyond vanilla U-Net?

In rough order of expected value for a project like this: (1) stain normalization/augmentation (Q12) — likely the single highest-leverage change given how small and single-source GlaS is; (2) a pretrained encoder (e.g. ImageNet or, better, a histopathology-pretrained self-supervised backbone) instead of training from scratch, especially valuable given only 68 training images; (3) architectural upgrades such as attention gates on the skip connections, deeper supervision, or a U-Net++/nnU-Net-style architecture; (4) k-fold cross-validation instead of a single split, to get a more reliable estimate of performance given the small dataset; (5) post-processing (morphological cleanup, connected-component filtering) and/or test-time augmentation; (6) an ensemble of models trained with different seeds/folds.

### 14. How would you perform instance segmentation instead of semantic segmentation?

This project deliberately collapses the GlaS dataset's instance-labeled masks (background=0, each gland a distinct id) down to a binary semantic mask (`src/data/dataset.py`), because distinguishing individual, possibly touching, glands is a different and harder problem. To do instance segmentation instead: (a) **post-processing approach** — keep a semantic (or better, boundary-aware) segmentation output and separate touching instances with a watershed transform, typically seeded from a predicted distance-to-boundary map or from local maxima of a predicted "gland center" heatmap; (b) **direct instance models** — use an architecture designed for instance segmentation, e.g. Mask R-CNN (detect-then-segment per instance) or a panoptic/embedding-based approach that predicts per-pixel embeddings and clusters them into instances. The original GlaS challenge was actually scored partly on object-level (instance) Dice/Hausdorff distance for exactly this reason — this project reports pixel-level semantic Dice/IoU instead, which is a narrower, simpler metric.

### 15. What are the limitations of this project?

See README §18 for the full list; summarized: (1) semantic, not instance, segmentation — touching glands aren't separated; (2) tile-level only, no WSI tiling/stitching pipeline implemented; (3) small dataset (85 official training images, 68 after the val split) with attendant variance; (4) no stain normalization/augmentation, so likely to be sensitive to staining differences on data outside GlaS; (5) single fixed train/val split rather than cross-validation; (6) not clinically validated in any way — this is a research/portfolio project, not a diagnostic tool.

### 16. What did your controlled experiments actually show — and were the results what you expected?

Honestly, no, and I think that's the more interesting answer to give. I ran three configs on the real GlaS data: (1) the baseline (augmentation + BCE+Dice loss), (2) the same model with augmentation disabled, (3) the same model with BCE-only loss. I expected augmentation and Dice loss to each help. Measured on the untouched 80-image test set, both ablations actually *outperformed* the baseline (baseline test Dice 0.7787; no-augmentation 0.8539; BCE-only 0.8050) — see README §12 for the full numbers, reported exactly as measured.

I don't read this as "augmentation and Dice loss are bad" — I read it as "one seeded run on a 68-image training set is a high-variance estimate, and I shouldn't over-interpret it." Plausible contributors: the augmentation policy (±30° rotation, 0.9-1.1x scale) may be too aggressive relative to how little data there is to learn true invariances from at this scale; early stopping means each run's *reported* checkpoint is whichever epoch happened to be best on a 17-image validation set, which is itself noisy; and with only 80 test images, a few hard or easy cases can swing the aggregate Dice by several points. The methodologically honest next step — which I didn't have compute/time budget to do here, and say so rather than paper over it — is k-fold cross-validation and/or multiple seeds per configuration to get a variance estimate on each of these deltas before concluding anything. If asked "so was Dice loss worth it," the accurate answer is: this single experiment doesn't tell me either way, and I'd want that further validation before trusting the comparison.
