# scGPT Perturbation Adapter (Frozen Backbone)

This project is an experimental extension of [scGPT](https://github.com/bowang-lab/scGPT)

By freezing a pretrained **scGPT** backbone and train **only new modules** to predict
**post-perturbation gene expression** from a control cell + a perturbation condition.

- Backbone: scGPT (Hugging Face weights) — stays **frozen**.
- Trainable: **GraphAdapter**, **PerturbConditioner**, **GeneRegressorHead**.
- Demo data: **PBMC3k** with synthetic perturbations so you can run end-to-end.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python train.py
python eval_infer.py --ckpt runs/best.pt
```

## Swap in real data

Your `AnnData` should include:
- `adata.layers['control']`  — baseline (log1p) expression
- `adata.layers['target']`   — post-perturbation (log1p) expression
- `adata.obs['perturb']`     — string perturbation labels (e.g., 'ctrl', 'IFNG', 'KO_STAT1', ...)

Update `train.py` to import your loader instead of `data.pbmc3k_synth.load_pbmc3k_synth`.
