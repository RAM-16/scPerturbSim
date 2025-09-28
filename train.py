import os, yaml, numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from data.pbmc3k_synth import load_pbmc3k_synth
from scgpt_perturb_adapter.utils import set_seed, to_device
from scgpt_perturb_adapter.backbone import FrozenBackbone
from scgpt_perturb_adapter.tokenizer import GeneTokenizer
from scgpt_perturb_adapter.graph_adapter import GraphAdapter
from scgpt_perturb_adapter.heads import PerturbConditioner, GeneRegressorHead

class PerturbDataset(Dataset):
    def __init__(self, X_ctrl, X_tgt, pert_ids, gene_ids, max_tokens=1000, mask_ratio=0.15):
        self.Xc = X_ctrl.tocsr()
        self.Xt = X_tgt.tocsr()
        self.pert_ids = pert_ids.astype(np.int64)
        self.gene_ids = np.asarray(gene_ids)
        self.max_tokens = max_tokens
        self.mask_ratio = mask_ratio

    def __len__(self): return self.Xc.shape[0]

    def __getitem__(self, i):
        row = self.Xc.getrow(i)
        idx = row.indices
        if idx.size > self.max_tokens - 1:
            idx = np.sort(np.random.choice(idx, size=self.max_tokens - 1, replace=False))
        token_ids = np.concatenate([[0], self.gene_ids[idx] + 1]).astype(np.int64)
        attn_mask = np.ones_like(token_ids, dtype=np.int64)
        y_full = self.Xt.getrow(i).toarray().ravel()
        y = y_full[idx]
        T = token_ids.shape[0]
        gene_pos = np.arange(1, T)
        m = max(1, int(self.mask_ratio * gene_pos.size))
        masked = np.zeros(T, dtype=np.int64)
        chosen = np.random.choice(gene_pos, size=m, replace=False)
        masked[chosen] = 1
        return {
            "input_ids": torch.from_numpy(token_ids),
            "attention_mask": torch.from_numpy(attn_mask),
            "pert_id": torch.tensor(self.pert_ids[i], dtype=torch.long),
            "target": torch.from_numpy(np.concatenate([[0.0], y]).astype(np.float32)),
            "mask": torch.from_numpy(masked),
        }

def collate(batch):
    maxlen = max(item["input_ids"].shape[0] for item in batch)
    out = {}
    def pad1(key, pad_val=0):
        arr = []
        for b in batch:
            x = b[key]
            pad = maxlen - x.shape[0]
            if pad > 0:
                x = torch.cat([x, torch.full((pad,), pad_val, dtype=x.dtype)], dim=0)
            arr.append(x.unsqueeze(0))
        return torch.cat(arr, dim=0)
    out["input_ids"] = pad1("input_ids", 0)
    out["attention_mask"] = pad1("attention_mask", 0)
    out["mask"] = pad1("mask", 0)
    out["target"] = pad1("target", 0)
    out["pert_id"] = torch.stack([b["pert_id"] for b in batch], dim=0)
    return out

def split_indices(n, val_size=0.15, test_size=0.15, seed=42):
    from sklearn.model_selection import train_test_split
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    idx_train, idx_tmp = train_test_split(idx, test_size=val_size+test_size, random_state=rng, shuffle=True)
    rel = (test_size) / (val_size + test_size)
    idx_val, idx_test = train_test_split(idx_tmp, test_size=rel, random_state=rng, shuffle=True)
    return idx_train, idx_val, idx_test

def build_gene_graph(adata, corr_threshold=0.25, max_neighbors=16):
    X = adata.layers["control"].toarray()
    C = np.corrcoef(X.T)
    G = C.shape[0]
    neighbors = []
    for g in range(G):
        r = C[g]
        idx = np.where(np.abs(r) >= corr_threshold)[0]
        idx = idx[idx != g]
        idx = idx[np.argsort(-np.abs(r[idx]))][:max_neighbors]
        neighbors.append(idx.tolist())
    return neighbors

def mse_masked(pred, target, mask):
    diff = (pred - target) ** 2
    diff = diff * mask
    denom = mask.sum().clamp(min=1)
    return diff.sum() / denom

def main():
    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    adata = load_pbmc3k_synth(hvg=cfg["data"]["hvg"],
                              min_cells=cfg["data"]["min_cells"],
                              n_perturb=cfg["data"]["n_perturb"],
                              ctrl_label=cfg["data"]["ctrl_label"],
                              seed=cfg["seed"])

    ctrl = adata.layers["control"]
    tgt  = adata.layers["target"]
    perts = list(adata.uns["perturbations"])
    pert2id = {p:i for i,p in enumerate(perts)}
    pert_ids = adata.obs["perturb"].map(pert2id).to_numpy().astype(np.int64)

    tokenizer = GeneTokenizer(cfg["model"]["hf_id"], dataset_genes=list(adata.var_names))
    gene_ids = np.array(tokenizer.encode_genes(list(adata.var_names)))
    keep = gene_ids >= 0
    adata = adata[:, keep].copy()
    ctrl = ctrl[:, keep]
    tgt  = tgt[:, keep]
    gene_ids = gene_ids[keep]

    idx_train, idx_val, idx_test = split_indices(adata.n_obs,
                                                 val_size=cfg["data"]["val_size"],
                                                 test_size=cfg["data"]["test_size"],
                                                 seed=cfg["seed"])

    ds_train = PerturbDataset(ctrl[idx_train], tgt[idx_train], pert_ids[idx_train], gene_ids,
                              max_tokens=cfg["data"]["max_tokens_per_cell"],
                              mask_ratio=cfg["train"]["mask_ratio"])
    ds_val   = PerturbDataset(ctrl[idx_val],   tgt[idx_val],   pert_ids[idx_val],   gene_ids,
                              max_tokens=cfg["data"]["max_tokens_per_cell"],
                              mask_ratio=cfg["train"]["mask_ratio"])
    ds_test  = PerturbDataset(ctrl[idx_test],  tgt[idx_test],  pert_ids[idx_test],  gene_ids,
                              max_tokens=cfg["data"]["max_tokens_per_cell"],
                              mask_ratio=cfg["train"]["mask_ratio"])

    dl_train = DataLoader(ds_train, batch_size=cfg["train"]["batch_size"], shuffle=True,  collate_fn=collate)
    dl_val   = DataLoader(ds_val,   batch_size=cfg["train"]["batch_size"], shuffle=False, collate_fn=collate)
    dl_test  = DataLoader(ds_test,  batch_size=cfg["train"]["batch_size"], shuffle=False, collate_fn=collate)

    backbone = FrozenBackbone(cfg["model"]["hf_id"], freeze=cfg["model"]["freeze_backbone"]).to(device)
    adapter  = GraphAdapter(hidden_size=backbone.hidden_size, bottleneck=cfg["adapter"]["bottleneck"]).to(device)
    cond     = PerturbConditioner(num_perts=len(perts), hidden_size=backbone.hidden_size).to(device)
    head     = GeneRegressorHead(hidden_size=backbone.hidden_size).to(device)

    for p in backbone.parameters(): p.requires_grad = False

    opt = torch.optim.AdamW(
        [{"params": adapter.parameters(), "lr": cfg["train"]["lr_adapter"]},
         {"params": cond.parameters(),   "lr": cfg["train"]["lr_cond"]},
         {"params": head.parameters(),   "lr": cfg["train"]["lr_head"]}],
        weight_decay=cfg["train"]["weight_decay"]
    )

    nbrs = build_gene_graph(adata, corr_threshold=cfg["adapter"]["corr_threshold"],
                            max_neighbors=cfg["adapter"]["max_neighbors"])

    def batch_neighbors(input_ids):
        B, T = input_ids.shape
        out = []
        for t in range(T):
            if t == 0: out.append([]); continue
            ids = input_ids[:, t]
            if int(ids.max()) == 0: out.append([]); continue
            gid = int(ids[0].item()) - 1
            out.append(nbrs[gid] if 0 <= gid < len(nbrs) else [])
        return out

    def run_epoch(loader, train=True):
        if train: adapter.train(); cond.train(); head.train()
        else:     adapter.eval();  cond.eval();  head.eval()
        loss_sum, count = 0.0, 0
        for batch in tqdm(loader, desc="train" if train else "eval"):
            batch = to_device(batch, device)
            with torch.set_grad_enabled(train):
                tok = batch["input_ids"].long()
                attn = batch["attention_mask"].long()
                mask = batch["mask"].float()
                target = batch["target"].float()
                pert = batch["pert_id"].long()

                token_emb, cls = backbone.embed_tokens(tok, attention_mask=attn)
                token_emb = adapter(token_emb, batch_neighbors(tok))
                token_emb = cond(token_emb, pert)
                yhat = head(token_emb)

                loss = mse_masked(yhat, target, mask)

                if train:
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()

            bs = tok.size(0)
            loss_sum += loss.item() * bs
            count += bs
        return loss_sum / count

    best = float("inf")
    os.makedirs(cfg["train"]["out_dir"], exist_ok=True)
    ckpt_path = os.path.join(cfg["train"]["out_dir"], "best.pt")

    for epoch in range(1, cfg["train"]["max_epochs"] + 1):
        tr = run_epoch(dl_train, train=True)
        va = run_epoch(dl_val, train=False)
        print(f"[Epoch {epoch}] train {tr:.4f} | val {va:.4f}")
        if va < best:
            best = va
            torch.save({"adapter": adapter.state_dict(),
                        "cond": cond.state_dict(),
                        "head": head.state_dict(),
                        "cfg": cfg}, ckpt_path)
            print(f"  ↳ saved {ckpt_path}")

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        adapter.load_state_dict(ckpt["adapter"]); cond.load_state_dict(ckpt["cond"]); head.load_state_dict(ckpt["head"])
    te = run_epoch(dl_test, train=False)
    print(f"[Test] MSE (masked): {te:.4f}")

if __name__ == "__main__":
    main()
