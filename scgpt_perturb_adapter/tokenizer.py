from transformers import AutoConfig
import json
from pathlib import Path

class GeneTokenizer:
    def __init__(self, hf_id: str, dataset_genes):
        self.gene2id = {}
        self.unknown = -1
        try:
            cfg = AutoConfig.from_pretrained(hf_id)
            vocab_path = getattr(cfg, "gene_vocab_path", None)
            if vocab_path and Path(vocab_path).exists():
                self.gene2id = json.loads(Path(vocab_path).read_text())
            elif hasattr(cfg, "gene2id"):
                self.gene2id = cfg.gene2id
        except Exception:
            pass
        if not self.gene2id:
            self.gene2id = {g: i for i, g in enumerate(dataset_genes)}
            print("[Tokenizer] Using dataset-local vocab (pretrained coverage may be limited).")

    def encode_genes(self, genes):
        return [self.gene2id.get(g, self.unknown) for g in genes]
