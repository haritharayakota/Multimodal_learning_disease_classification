import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import gensim
from collections import Counter

class UltraEarlyFusion(nn.Module):
    def __init__(self, word2vec_path, num_classes=14,
                 embed_dim=300, patch_size=16, max_text_len=128, freeze_w2v=False):
        super(UltraEarlyFusion, self).__init__()
        
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.max_text_len = max_text_len
        self.word2vec_path = word2vec_path
        self.freeze_w2v = freeze_w2v
        self.embedding_layer = None
        self.token_to_idx = None
        self.text_pos_embeddings = nn.Embedding(max_text_len, embed_dim)
        self.patch_proj = nn.Linear(3 * patch_size * patch_size, embed_dim)
        self.img_pos_embed = nn.Parameter(torch.zeros(1, 1000, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=10, batch_first=True)
        self.shared_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def build_vocab_from_batch(self, texts, min_freq=1):
        """Builds vocab dynamically on the first forward pass."""
        counter = Counter()
        for text in texts:
            if isinstance(text, str):
                counter.update(text.lower().split())

        vocab = [w for w, f in counter.items() if f >= min_freq]
        print(f"[Vocab built with {len(vocab)} tokens]")
        w2v_model = gensim.models.KeyedVectors.load_word2vec_format(self.word2vec_path, binary=True)       
        token_to_idx = {"<PAD>": 0}
        embedding_matrix = [torch.zeros(self.embed_dim)]
        
        for token in vocab:
            token_to_idx[token] = len(token_to_idx)
            if token in w2v_model.key_to_index:
                embedding_matrix.append(torch.tensor(w2v_model[token]))
            else:
                embedding_matrix.append(torch.randn(self.embed_dim))
        
        embedding_matrix = torch.stack(embedding_matrix)
        embedding_layer = nn.Embedding.from_pretrained(embedding_matrix, freeze=self.freeze_w2v)
        
        self.embedding_layer = embedding_layer
        self.token_to_idx = token_to_idx

    def encode_texts(self, texts):
        if self.token_to_idx is None:
            self.build_vocab_from_batch(texts)

        tokenized = []
        for text in texts:
            if not isinstance(text, str):
                text = ""
            token_ids = [self.token_to_idx.get(token.lower(), 0) for token in text.split()]
            token_ids = token_ids[:self.max_text_len]
            tokenized.append(torch.tensor(token_ids))
        
        padded = pad_sequence(tokenized, batch_first=True, padding_value=0)
        return padded
    def forward(self, images, raw_texts):
        device = images.device
        B, C, H, W = images.shape
        patches = images.unfold(2, self.patch_size, self.patch_size) \
        .unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(B, C, -1, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 1, 3, 4)    
        patches = patches.flatten(2)
        img_tokens = self.patch_proj(patches)
        img_tokens = img_tokens + self.img_pos_embed[:, :img_tokens.size(1), :].to(device)
        token_ids = self.encode_texts(raw_texts).to(device) 
        pos_ids = torch.arange(token_ids.size(1), device=device) 
        pos_embed = self.text_pos_embeddings(pos_ids)
        self.embedding_layer = self.embedding_layer.to(device)
        txt_tokens = self.embedding_layer(token_ids) + pos_embed.unsqueeze(0)
        combined_tokens = torch.cat((img_tokens, txt_tokens), dim=1)
        fused_tokens = self.shared_encoder(combined_tokens)
        pooled_rep = fused_tokens.mean(dim=1)
        return self.classifier(pooled_rep)
