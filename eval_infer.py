import argparse, yaml, torch, numpy as np
from torch.utils.data import DataLoader
from data.pbmc3k_synth import load_pbmc3k_synth
from scgpt_perturb_adapter.backbone import FrozenBackbone
from scgpt_perturb_adapter.graph_adapter import GraphAdapter
from scgpt_perturb_adapter.heads import PerturbConditioner, GeneRegressorHead
from scgpt_perturb_adapter.tokenizer import GeneTokenizer
from train import PerturbDataset, collate, build_gene_graph, mse_masked

def main(args):
    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    adata = load_pbmc3k_synth(hvg=cfg["data"]["hvg"],
                              min_cells=cfg["data"]["min_cells"],
                              n_perturb=cfg["data"]["n_perturb"],
                              ctrl_label=cfg["data"]["ctrl_label"],
                              seed=cfg["seed"])

    ctrl = adata.layers["control"]; tgt = adata.layers["target"]
    perts = list(adata.uns["perturbations"])
    pert2id = {p:i for i,p in enumerate(perts)}
    pert_ids = adata.obs["perturb"].map(pert2id).to_numpy().astype(np.int64)

    tokenizer = GeneTokenizer(cfg["model"]["hf_id"], dataset_genes=list(adata.var_names))
    gene_ids = np.array(tokenizer.encode_genes(list(adata.var_names)))
    keep = gene_ids >= 0
    adata = adata[:, keep].copy()
    ctrl = ctrl[:, keep]; tgt = tgt[:, keep]; gene_ids = gene_ids[keep]

    ds = PerturbDataset(ctrl, tgt, pert_ids, gene_ids,
                        max_tokens=cfg["data"]["max_tokens_per_cell"],
                        mask_ratio=cfg["train"]["mask_ratio"])
    dl = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=False, collate_fn=collate)

    backbone = FrozenBackbone(cfg["model"]["hf_id"], freeze=True).to(device)
    adapter  = GraphAdapter(hidden_size=backbone.hidden_size, bottleneck=cfg["adapter"]["bottleneck"]).to(device)
    cond     = PerturbConditioner(num_perts=len(perts), hidden_size=backbone.hidden_size).to(device)
    head     = GeneRegressorHead(hidden_size=backbone.hidden_size).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    adapter.load_state_dict(ckpt["adapter"]); cond.load_state_dict(ckpt["cond"]); head.load_state_dict(ckpt["head"])
    adapter.eval(); cond.eval(); head.eval()

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

    total_loss, total_n = 0.0, 0
    with torch.no_grad():
        for batch in dl:
            tok = batch["input_ids"].to(device).long()
            attn = batch["attention_mask"].to(device).long()
            mask = batch["mask"].to(device).float()
            target = batch["target"].to(device).float()
            pert = batch["pert_id"].to(device).long()
            token_emb, cls = backbone.embed_tokens(tok, attention_mask=attn)
            token_emb = adapter(token_emb, batch_neighbors(tok))
            token_emb = cond(token_emb, pert)
            yhat = head(token_emb)
            loss = mse_masked(yhat, target, mask)
            bs = tok.size(0)
            total_loss += loss.item() * bs
            total_n += bs
    print(f"Masked MSE: {total_loss/total_n:.4f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    main(ap.parse_args())
