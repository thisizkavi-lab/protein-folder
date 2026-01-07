import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        # x shape: [Batch, Seq_Len, Dim]
        return x + self.pe[:, :x.size(1)]

class ProteinFolder(nn.Module):
    def __init__(self, vocab_size=20, d_model=64, nhead=4, num_layers=4, dim_feedforward=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Deeper Transformer for Real Data
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, batch_first=True, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output projection
        self.contact_map_head = nn.Linear(d_model, 1) 

    def forward(self, src):
        src = self.embedding(src) * math.sqrt(self.embedding.embedding_dim) # Scaling factor
        src = self.pos_encoder(src)
        
        encoded = self.transformer_encoder(src)
        
        # Residue interaction via broadcasting
        batch, seq_len, d_model = encoded.shape
        i = encoded.unsqueeze(2) 
        j = encoded.unsqueeze(1) 
        
        # Interaction features
        interaction = torch.relu(i * j) 
        
        # Predict contact probability
        contact_map = self.contact_map_head(interaction).squeeze(-1)
        
        return torch.sigmoid(contact_map)
