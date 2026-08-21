import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import copy
from dgllife.model.model_zoo.gcn_predictor import GCNPredictor
import os, random, numpy as np, torch

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
try:
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
except Exception:
    pass

def get_position_encoding(seq_len, embed):
    pe = np.array([[pos / (10000.0 ** (i // 2 * 2.0 / embed)) for i in range(embed)] for pos in range(seq_len)])
    pe[:, 0::2] = np.sin(pe[:, 0::2])
    pe[:, 1::2] = np.cos(pe[:, 1::2])
    return pe

class Positional_Encoding(nn.Module):
    def __init__(self, embed, pad_size, dropout):
        super(Positional_Encoding, self).__init__()
        pe = np.array([[pos / (10000.0 ** (i // 2 * 2.0 / embed)) for i in range(embed)]
                       for pos in range(pad_size)], dtype=np.float32)
        pe[:, 0::2] = np.sin(pe[:, 0::2])
        pe[:, 1::2] = np.cos(pe[:, 1::2])
        self.register_buffer('pe', torch.tensor(pe))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        squeeze_back = False
        if x.dim() == 2:
            # treat as [B, 1, embed]
            x = x.unsqueeze(1)
            squeeze_back = True

        L = x.size(1)
        pe = self.pe[:L].to(x.device)
        out = x + pe  # broadcast add
        out = self.dropout(out)
        if squeeze_back:
            out = out.squeeze(1)
        return out

class Scaled_Dot_Product_Attention(nn.Module):
    '''Scaled Dot-Product'''
    def __init__(self):
        super(Scaled_Dot_Product_Attention, self).__init__()

    def forward(self, Q, K, V, scale=None):
        attention = torch.matmul(Q, K.permute(0, 2, 1))  # Q*K^T
        if scale:
            attention = attention * scale
        attention = F.softmax(attention, dim=-1)
        context = torch.matmul(attention, V)
        return context

class Multi_Head_Attention(nn.Module):
    def __init__(self, dim_model, num_head, dropout=0.0):
        super(Multi_Head_Attention, self).__init__()
        self.num_head = num_head
        assert dim_model % num_head == 0
        self.dim_head = dim_model // self.num_head
        self.fc_Q = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_K = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_V = nn.Linear(dim_model, num_head * self.dim_head)
        self.attention = Scaled_Dot_Product_Attention()
        self.fc = nn.Linear(num_head * self.dim_head, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        batch_size = x.size(0)
        Q = self.fc_Q(x)
        K = self.fc_K(x)
        V = self.fc_V(x)
        Q = Q.view(batch_size * self.num_head, -1, self.dim_head)
        K = K.view(batch_size * self.num_head, -1, self.dim_head)
        V = V.view(batch_size * self.num_head, -1, self.dim_head)
        scale = K.size(-1) ** -0.5
        context = self.attention(Q, K, V, scale) # Scaled_Dot_Product_Attention
        context = context.view(batch_size, -1, self.dim_head * self.num_head)
        out = self.fc(context)
        out = self.dropout(out)
        out = out + x
        out = self.layer_norm(out)
        return out

class Position_wise_Feed_Forward(nn.Module):
    def __init__(self, dim_model, hidden, dropout=0.0):
        super(Position_wise_Feed_Forward, self).__init__()
        self.fc1 = nn.Linear(dim_model, hidden)
        self.fc2 = nn.Linear(hidden, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = out + x
        out = self.layer_norm(out)
        return out

class Encoder(nn.Module):
    def __init__(self, dim_model, num_head, hidden, dropout):
        super(Encoder, self).__init__()
        self.attention = Multi_Head_Attention(dim_model, num_head, dropout)
        self.feed_forward = Position_wise_Feed_Forward(dim_model, hidden, dropout)

    def forward(self, x):
        out = self.attention(x)
        out = self.feed_forward(out)
        return out

class ConfigTrans(object):
    def __init__(self):
        self.model_name = 'Transformer'
        self.dropout = 0.5
        self.num_classes = 1
        self.num_epochs = 100
        # self.batch_size = 12
        self.pad_size = 1
        self.learning_rate = 0.001
        self.embed = 128
        self.dim_model = 128
        self.hidden = 1024
        self.last_hidden = 512
        self.num_head = 8
        self.num_encoder = 2

config = ConfigTrans()

class Transformer_test(nn.Module):
    def __init__(self):
        super(Transformer_test, self).__init__()
        self.postion_embedding = Positional_Encoding(config.embed, config.pad_size, config.dropout)
        self.encoder = Encoder(config.dim_model, config.num_head, config.hidden, config.dropout)
        self.encoders = nn.ModuleList([copy.deepcopy(self.encoder) for _ in range(config.num_encoder)])

    def forward(self, x):

        if x.dim() == 2:
            x = x.float()
        out = self.postion_embedding(x)
        if out.dim() == 2:
            out = out.unsqueeze(1)
            need_squeeze = True
        else:
            need_squeeze = False

        for encoder in self.encoders:
            out = encoder(out)
        out = out.view(out.size(0), -1)  # [B, embed]
        if need_squeeze:
            out = out  # already [B,embed]
        return out

class Model_TGCN(nn.Module):
    def __init__(self, transformer_dim, gcn_out_dim, num_fc_out, num_classes=1):
        super(Model_TGCN, self).__init__()
        self.num_fc = nn.Linear(103, num_fc_out)    # 103 → num_fc_out
        self.norm = nn.BatchNorm1d(num_fc_out)

        total_dim = transformer_dim + gcn_out_dim + num_fc_out
        self.fc1 = nn.Linear(total_dim, num_classes)
        self.sig = nn.Sigmoid()

        self.tsne_list = []
        self._tsne_record_limit = 200

    def forward(self, x_t, x_g, x_l):

        num_out = self.num_fc(x_l)       # [B, num_fc_out]
        num_out = self.norm(num_out)
        feat = torch.cat([x_t, x_g, num_out], dim=-1)  # [B, total_dim]

        if not self.training and len(self.tsne_list) < self._tsne_record_limit:
            try:
                self.tsne_list.append(feat.detach().cpu().numpy())
            except Exception:
                pass

        out = self.fc1(feat)
        return self.sig(out).view(-1)
