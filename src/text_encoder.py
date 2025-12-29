from transformers import AutoModel
import torch
import torch.nn as nn

class TextEncoder(nn.Module):

    
    def __init__(self, model_name="emilyalsentzer/Bio_ClinicalBERT", embed_dim=768):
        super(TextEncoder, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.projector = nn.Linear(self.bert.config.hidden_size, embed_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state  
        projected_tokens = self.projector(token_embeddings)
        return projected_tokens
