import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
import time
import numpy as np
from Sage import SAGE


class MaskedTemporalSelfAttention(nn.Module):
    def __init__(
            self, input_dim, hidden_dim, out_dim,
            attn_wnd, time_steps, aggr, dropout_p,
            nhead, num_layers, device
    ):
        super(MaskedTemporalSelfAttention, self).__init__()
        self.time_steps = time_steps
        self.attn_wnd = attn_wnd if attn_wnd!=-1 else time_steps
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim * 2, nhead=nhead, dropout=dropout_p)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.masks = torch.triu(torch.full((self.time_steps, self.time_steps), float('-inf')), diagonal=1).to(device)
        re_masks = torch.triu(torch.full((self.time_steps, self.time_steps), float('-inf')), diagonal=self.attn_wnd)
        self.masks += re_masks.transpose(0, 1).to(device)
        self.out = nn.Linear(hidden_dim * 2, out_dim, bias=False)
        self.Sage = SAGE(input_dim, hidden_dim, aggr, dropout_p)
        self.act = nn.ReLU()

    def forward(self, data_list_i, data_list_j):
        # print(np.random.randint(5, 10))
        # time.sleep(1)
        temporal_pairs_embeddings = []
        for t in range(self.time_steps):
            x_i = data_list_i[t][2][0].srcdata['feats']
            x_j = data_list_j[t][2][0].srcdata['feats']
            sage_o_i = self.Sage(data_list_i[t][2], x_i)
            sage_o_j = self.Sage(data_list_j[t][2], x_j)
            sage_o_i = F.normalize(sage_o_i, p=2, dim=1)
            sage_o_j = F.normalize(sage_o_j, p=2, dim=1)
            _, idx_i = torch.unique(data_list_i[t][1], return_inverse=True)
            _, idx_j = torch.unique(data_list_j[t][1], return_inverse=True)
            temporal_pairs_embeddings.append(torch.cat((sage_o_i[idx_i], sage_o_j[idx_j]), -1))
        #
        temporal_pairs_embeddings = torch.stack(temporal_pairs_embeddings)
        att_in = temporal_pairs_embeddings
        #
        outs = self.transformer_encoder(temporal_pairs_embeddings, mask=self.masks)# 公式10改成 H波=H_FFN

        return att_in,self.out(outs).squeeze(-1)
