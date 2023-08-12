import torch
import torch.nn as nn
import torch.nn.functional as F
from Sage import SAGE


class TDa(nn.Module):
    def __init__(
            self, input_dim, hidden_dim, out_dim,
            attn_wnd, time_steps, aggr, dropout_p,
            device
    ):
        super(TDa, self).__init__()
        self.time_steps = time_steps
        self.attn_wnd = attn_wnd if attn_wnd!=-1 else time_steps
        self.device = device
        self.out = nn.Linear(hidden_dim * 2, out_dim, bias=False)
        self.Sage = SAGE(input_dim, hidden_dim, aggr, dropout_p)

        self.W_half = nn.Linear(hidden_dim * 4, hidden_dim * 2)
        self.W_ff2 = nn.Linear(hidden_dim * 2, 1)
        self.W_ff1 = nn.Linear(self.time_steps, 1)

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
        temporal_pairs_embeddings = torch.stack(temporal_pairs_embeddings)

        temporal_pairs_embeddings = temporal_pairs_embeddings.transpose(0, 1)

        difference_embeddings = []
        pre = temporal_pairs_embeddings

        for i in range(self.attn_wnd - 1):
            pre = torch.cat(
                (torch.zeros([pre.shape[0], i + 1, pre.shape[-1]]).to(self.device), pre[:, i:-1, :]),
                1
            )

            difference_embeddings.append(temporal_pairs_embeddings - pre)

        difference_embeddings = torch.stack(difference_embeddings)
        difference_weights = self.W_ff2(difference_embeddings).squeeze(-1)
        difference_weights = torch.relu(difference_weights)

        diff_weights = self.W_ff1(difference_weights).squeeze(-1)
        diff_weights = diff_weights.unsqueeze(-1).unsqueeze(-1)
        diff_weights = F.softmax(diff_weights, dim=0)

        diff_att = torch.mul(diff_weights, torch.sigmoid(difference_embeddings))
        diff_att = torch.sum(diff_att, dim=0)

        outs = torch.mul(temporal_pairs_embeddings, diff_att)
        outs = outs.transpose(0, 1)

        return self.out(outs).squeeze(-1)

