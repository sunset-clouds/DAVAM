import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from torch.nn import functional as F

class VectorQuantize(nn.Module):
    def __init__(self,dim, n_embed, decay=0.99, eps=1e-5):
        super(VectorQuantize,self).__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        self.register_buffer('embed', embed)
        self.register_buffer('cluster_size', torch.zeros(n_embed))
        self.register_buffer('embed_avg', embed.clone())

    def forward(self, input):
        flatten = input.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        ### if pytorch support one_hot function, please use the F.one_hot
        # embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        ### if pytorch don't support one hot, please use the scatter_
        embed_ind_expand = embed_ind.unsqueeze(1)
        embed_onehot = torch.cuda.FloatTensor(\
            embed_ind_expand.size(0),self.n_embed).fill_(0).scatter_(1,embed_ind_expand,1).type(flatten.dtype)

        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            self.cluster_size.data.mul_(self.decay).add_(
                1 - self.decay, embed_onehot.sum(0)
            )
            embed_sum = flatten.transpose(0, 1) @ embed_onehot
            self.embed_avg.data.mul_(self.decay).add_(1 - self.decay, embed_sum)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).sum(-1).sum(-1)
        quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))
