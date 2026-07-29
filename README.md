# Skincare recommendation revision

This directory is the reproducible revision workspace for manuscript
`Access-2026-23560`. The original notebooks and Overleaf export are preserved
outside this directory and under `manuscript/original/`.

## Data

The raw data is the **Sephora Products and Skincare Reviews** dataset by Nady
Inky, distributed under CC BY 4.0:

<https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews>

Raw CSV files are intentionally excluded from Git. Place the five
`reviews_*.csv` files and `product_info.csv` in `data/raw/`. The repository
does not redistribute the dataset.

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
pytest -q

PYTHONPATH=src python -m skincare_rec.cli all \
  --config configs/revision.yaml \
  --neural lightfm ncf lightgcn
```

The `all` command runs the audit, chronological warm-start split, validation
ablation, all warm-start baselines, duplicate sensitivity analysis, five
temporal cutoffs, five item-held-out cold-start folds, leakage-controlled
reverse recommendation, clustering stability checks, and figure generation.
The completed run can take several hours depending on hardware.

All manuscript numbers come from the machine-readable files in `results/`.
The legacy notebooks are preserved as historical material but are not evidence
for revised claims. Random seeds and all evaluation settings are fixed in
`configs/revision.yaml`.

## Protocol safeguards

- Positive preference is `rating >= 4`; negative is `rating <= 2`; neutral is
  `rating == 3`.
- Repeated positive reviews for one user–product pair are collapsed before the
  split.
- TF-IDF vocabulary and IDF are fitted only on training products, with one
  product per document.
- Test targets and previously seen items are excluded from recommendation
  candidates.
- Item-held-out folds keep exact and near-duplicate formulations together.
- Reverse recommendation removes the target product from each candidate user
  profile.

## Submission artifacts

The clean and highlighted manuscripts, response letter, resubmission
checklist, compliance matrix, and controlled change log are generated under
`output/` and `reports/`. The public revision repository is:

<https://github.com/Oykubickici/SkincareRecommendation>
