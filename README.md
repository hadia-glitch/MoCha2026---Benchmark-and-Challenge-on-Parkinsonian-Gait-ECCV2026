# MoCha 2026 — Parkinsonian Gait Severity Classification

A classification head trained on top of a frozen motion-language encoder to predict
MDS-UPDRS gait severity (4 ordinal classes) from canonicalized SMPL motion sequences,
built for the **[MoCha 2026 @ ECCV](https://mocha.care-pd.ca/) — Benchmark and Challenge
on Parkinsonian Gait**.

**36th of 113 teams**

This repository documents the first submitted version of the model
(CodaBench leaderboard **macro F1 = 0.40**).

---

## 1. Task

Given a walking sequence encoded as SMPL pose parameters, predict the clinician-assigned
MDS-UPDRS gait severity class:

| Class | Meaning |
|---|---|
| 0 | Normal |
| 1 | Slight impairment |
| 2 | Mild impairment |
| 3 | Marked impairment |

Each input is a variable-length sequence of SMPL axis-angle poses `(T, 72)` and root
translations `(T, 3)`, sampled at a dataset-specific frame rate. Labels are ordinal
(0 < 1 < 2 < 3): a prediction of 0 for a true class 3 is a worse error than predicting 2,
which the evaluation reflects via **Quadratic Weighted Kappa (QWK)** alongside macro F1.

Training data comes from four labeled clinical sites in the
[CARE-PD](https://neurips2025.care-pd.ca/) dataset (3DGait, BMCLab, PD-GaM, T-SDU-PD).
The hidden evaluation set is drawn from **different, unseen clinical sites**, so the
real objective is not just fitting these four sites well — it's generalizing to gait
capture protocols the model has never seen.

## 2. Approach at a glance

```
raw SMPL sequence (pose, trans, fps)
        │
        ▼
MotionPreprocessor            (baseline-provided: canonicalization, resampling,
        │                      HumanML3D-style feature conversion)
        ▼
frozen motion-text encoder    (baseline-provided, finetuned checkpoint;
        │                      never updated during this project)
        ▼
512-dim motion embedding
        │
        ▼
ImprovedClassifier            (this project: 512→256→128→64→4 MLP + ordinal heads,
        │                      trained with class-imbalance-aware objectives)
        ▼
severity class ∈ {0,1,2,3}
```

The motion encoder (a HumanML3D-style text/motion retrieval model, provided and
finetuned by the challenge organizers) is used purely as a **frozen feature extractor**.
All the modeling effort here goes into the classification head trained on top of the
resulting 512-dim embeddings — which is also where the interesting problem lives, since
the label distribution is severely imbalanced and the model has to generalize across
sites it wasn't trained on.

## 3. The core problem: severe, ordinal class imbalance

Across the four labeled sites (2,953 walks total):

| Class | Count | % of data |
|---|---|---|
| 0 (Normal) | 1,244 | 42.1% |
| 1 (Slight) | 1,071 | 36.3% |
| 2 (Mild) | 593 | 20.1% |
| 3 (Marked) | 45 | **1.5%** |

Class 3 — the clinically most important class, since it flags the most severe
impairment — has 45 examples total. Naively optimizing accuracy or unweighted
cross-entropy on this distribution produces a classifier that essentially never
predicts class 3, since doing so is almost always the empirically "safe" choice. Since
the competition metric is **macro F1** (every class weighted equally regardless of
support) plus QWK, that failure mode is heavily penalized. Most of the design decisions
below exist specifically to counteract it.

### 3.1 Class-weighted loss

Both the primary cross-entropy term and the auxiliary ordinal losses (below) use
per-class weights computed from inverse class frequency:

```python
class_weights_np = np.array([
    (n_total / (4 * class_counts[c])) ** 1.2 for c in range(4)
])
```

This produces weights of `[0.53, 0.64, 1.30, 28.71]` — class 3 samples contribute
roughly 28× the gradient signal of class 0 samples per example, which is what lets a
class with 1.5% prevalence meaningfully influence training at all.

### 3.2 SMOTE oversampling in embedding space

Because the encoder is frozen, every walk maps to a fixed 512-dim embedding before the
classifier ever sees it. This makes the embeddings a natural space to apply **SMOTE**
(Synthetic Minority Over-sampling Technique): for each training fold, minority classes
are oversampled by interpolating between a sample and its nearest neighbors *of the
same class* in embedding space, with `k_neighbors` adapted per fold to the number of
available class-3 examples (as few as ~7–9 after a site is held out):

```python
k_nn = min(5, max(2, Counter(y_tr)[3] - 1))
sm = SMOTE(sampling_strategy=strat, k_neighbors=k_nn, random_state=42)
```

The oversampling target itself is capped relative to the majority class
(`n_majority // 5`) rather than fully balancing all classes, which avoids drowning the
training set in synthetic minority points.

### 3.3 Exploiting the ordinal structure directly

Standard multi-class classification treats classes 0–3 as unrelated categories, which
throws away the fact that they're ordered severity levels. On top of the main 4-way
softmax head, the classifier has three auxiliary binary heads trained to predict the
cumulative thresholds:

```python
self.head_b1 = nn.Linear(64, 1)  # P(y >= 1)
self.head_b2 = nn.Linear(64, 1)  # P(y >= 2)
self.head_b3 = nn.Linear(64, 1)  # P(y >= 3)
```

Each threshold task pools gradient signal across every class on either side of the cut
— e.g. `P(y >= 3)` is trained using *all* class-0/1/2 samples as negatives and all
class-3 samples as positives, which gives the rarest class far more effective training
signal than the 4-way softmax alone would. The three threshold losses are combined with
the main cross-entropy loss, still class-weighted:

```python
def ordinal_loss(logits, b1, b2, b3, targets, class_weights_tensor):
    ce = F.cross_entropy(logits, targets, weight=class_weights_tensor)
    t1, t2, t3 = (targets >= 1).float(), (targets >= 2).float(), (targets >= 3).float()
    sw = class_weights_tensor[targets]
    bce1 = F.binary_cross_entropy_with_logits(b1, t1, weight=sw)
    bce2 = F.binary_cross_entropy_with_logits(b2, t2, weight=sw)
    bce3 = F.binary_cross_entropy_with_logits(b3, t3, weight=sw)
    return ce + 0.3 * (bce1 + bce2 + bce3)
```

### 3.4 Embedding-space augmentation

During the first 70% of training, small Gaussian noise (`σ=0.05`) is added to each
embedding batch. Combined with SMOTE's synthetic points, this discourages the
classifier from memorizing exact minority-class embeddings and encourages it to learn a
slightly more robust decision region around them — relevant given how few genuine class
3 examples exist to define that region in the first place.

### 3.5 Regularization to prevent the head from just overfitting the minority points

Given how aggressively the loss is being pushed to attend to 45 examples, the
classifier head itself is deliberately regularized: `LayerNorm` after each hidden
layer, dropout (`0.4` then `0.3`), weight decay (`1e-3` via `AdamW`), gradient norm
clipping (`1.0`), and a cosine-annealed learning rate schedule.

## 4. Validation methodology: Leave-One-Site-Out (LOSO)

Because the hidden benchmark evaluates on clinical sites never seen during training,
standard random-split cross-validation would be a poor proxy for real performance — a
random split can leak site-specific gait-capture artifacts between train and validation
regardless of label imbalance. Instead, validation uses **Leave-One-Site-Out**
cross-validation: for each of the four labeled sites, a model is trained on the other
three and evaluated on the held-out one. This is the closest approximation available,
using only the labeled data, to "how does this generalize to a site it has never seen."

```python
for holdout_site in SITES:
    tr_mask = groups != holdout_site
    val_mask = groups == holdout_site
    ...
```

## 5. Results

**Per-site LOSO validation:**

| Held-out site | Macro F1 | QWK |
|---|---|---|
| 3DGait | 0.472 | 0.546 |
| BMCLab | 0.518 | 0.636 |
| PD-GaM | 0.520 | 0.577 |
| T-SDU-PD | 0.272 | 0.270 |
| **Mean** | **0.446** | **0.508** |

**Pooled out-of-fold:**

| Metric | Value |
|---|---|
| Macro F1 | 0.504 |
| QWK | 0.540 |

```
              precision    recall  f1-score   support
           0       0.70      0.82      0.76      1244
           1       0.62      0.34      0.44      1071
           2       0.33      0.45      0.38       593
           3       0.30      0.78      0.44        45
    accuracy                           0.57      2953
   macro avg       0.49      0.60      0.50      2953
weighted avg       0.59      0.57      0.56      2953
```

**Hidden benchmark (CodaBench leaderboard): macro F1 = 0.40.**

Class 3 recall (0.78) confirms the imbalance-handling above is doing real work — the
model finds most severe cases despite having seen only ~34 of them per training fold
before oversampling. Class 3 precision (0.30) is the visible cost: aggressive
class-3 upweighting trades some false positives for that recall, which is the honest
tradeoff of this particular weighting scheme rather than a hidden flaw. T-SDU-PD is
consistently the hardest held-out site, which the LOSO protocol surfaces directly
rather than hiding behind an aggregate score.

## 6. Repository structure

```
├── README.md
├── notebook/
│   └── MoCha_gait_severity_v1_training.ipynb   # full training pipeline (this doc)
├── submission_code/
│   └── baseline_model.py                        # the exact inference code submitted
├── weights_classifier/
│   └── classifier.pth                           # this project's trained head only
└── requirements.txt
```

**What is *not* included, and why:** the frozen motion encoder checkpoint, the base
text-motion-matching checkpoint, and the SMPL body model file are all provided by the
challenge organizers / the SMPL project, distributed via Git LFS from the official
[MoCha_baseline_bundle](https://github.com/TaatiTeam/MoCha_baseline_bundle). One of
them exceeds GitHub's 100 MB file limit, and the SMPL model specifically is distributed
under its own registration-gated license that does not permit redistribution. Only
`classifier.pth` — the ~700 KB head trained in this project — is this project's own
artifact and is included directly.

## 7. Reproducing this

1. Clone the official baseline bundle and pull its LFS assets:
   ```bash
   git clone https://github.com/TaatiTeam/MoCha_baseline_bundle
   cd MoCha_baseline_bundle && git lfs pull
   ```
2. Obtain the CARE-PD canonicalized SMPL pickles per the
   [dataset instructions](https://huggingface.co/datasets/vida-adl/CARE-PD).
3. Run `notebook/MoCha_gait_severity_v1_training.ipynb` (designed for Google Colab;
   adjust `DATA_DIR` / Drive paths for a local environment) end-to-end. It will:
   - extract 512-dim embeddings for every labeled walk via the frozen encoder,
   - run the 4-fold LOSO evaluation described above,
   - train the final head on all labeled data,
   - package a CodaBench-ready submission zip using `submission_code/baseline_model.py`
     as the updated inference entry point.
4. To reuse the already-trained head from this project directly: drop
   `weights_classifier/classifier.pth` into `<baseline_bundle>/weights/classifier/` and
   replace `<baseline_bundle>/submission/baseline_model.py` with
   `submission_code/baseline_model.py`.

## 8. Environment

See `requirements.txt`. The notebook was run on Google Colab (GPU runtime) against the
package versions available there at the time; the frozen-encoder and SMPL-related
dependencies (`torch`, `chumpy`, `human_body_prior`) are pinned by the baseline bundle
itself rather than this repository.

## 9. Known limitations

Documented honestly, since they're the natural next steps rather than hidden flaws:

- The `**1.2` class-weight exponent is a heuristic, not tuned against a
  precision/recall tradeoff objective — it was chosen to guarantee class 3 gets
  meaningful gradient signal, at the cost of the precision drop visible above.
- LOSO variance across sites is substantial, which the single pooled leaderboard score doesn't surface.
- The 512-dim representation comes from an encoder trained for general motion-text
  retrieval, not for clinical severity specifically — it is used as-is, frozen.

## 10. Acknowledgments

- Challenge, dataset, and baseline bundle: MoCha 2026 organizing committee
  (Adeli, Shaban, Taati, et al.), [mocha.care-pd.ca](https://mocha.care-pd.ca/).
- Training data: [CARE-PD](https://neurips2025.care-pd.ca/).
- Frozen motion encoder architecture: HumanML3D-style text-motion retrieval model,
  provided and finetuned by the organizers as part of the baseline bundle.
- SMPL body model: © Max Planck Institute for Intelligent Systems; used under its own
  license, not redistributed here.

## License

The code original to this repository (the notebook and `submission_code/baseline_model.py`
modifications) may be used under the license of your choice — add one if you intend to
publish this repository (e.g. MIT). The baseline bundle components referenced above
remain under the terms of the official MoCha_baseline_bundle repository, and the SMPL
model remains under its own license.
