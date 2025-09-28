import torch
from torch import nn
from transformers import AutoModel, AutoConfig

class FrozenBackbone(nn.Module):
    def __init__(self, hf_id: str, freeze: bool = True):
        super().__init__()
        self.config = AutoConfig.from_pretrained(hf_id)
        self.encoder = AutoModel.from_pretrained(hf_id)
        self.hidden_size = getattr(self.config, "hidden_size", 512)
        if freeze:
            for p in self.encoder.parameters(): p.requires_grad = False
            self.encoder.eval()

    @torch.no_grad()
    def embed_tokens(self, input_ids, attention_mask=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state
        cls = last_hidden[:, 0]
        return last_hidden, cls
