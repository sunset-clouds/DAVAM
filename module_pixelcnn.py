import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class GatedActivation(nn.Module):
    def __init__(self):
        super(GatedActivation,self).__init__()

    def forward(self, x):
        #x [batch_size,2*out_chs,src_lens]
        x, y = x.chunk(2, dim=1)
        #x,y [batch_size,out_chs,src_lens]

        #return [batch_size,out_chs,src_lens]
        return F.tanh(x) * F.sigmoid(y)
        #return x * F.sigmoid(y)

class CausalConv1d(nn.Module):
    def __init__(self,in_chs,out_chs,kernel_size,mask_type):
        super(CausalConv1d,self).__init__()

        self.pad = torch.nn.ConstantPad1d((kernel_size-1,0),0)
        self.conv = nn.Conv1d(in_chs,out_chs,kernel_size)

        self.mask_type = mask_type

    def forward(self,x):
        #x (batch_size,in_chs,src_lens)
        x = self.pad(x)
        #x (batch_size,in_chs,src_lens+kernel_size-1)

        if self.mask_type == "causal":
            self.conv.weight.data[:, :,-1].zero_()

        out = self.conv(x)
        #return  out (batch_size,out_chs,src_lens)
        return out

class GatedCausalConv1d(nn.Module):
    def __init__(self,in_chs,out_chs,kernel_size,mask_type,residual=True):
        super(GatedCausalConv1d,self).__init__()
        
        self.residual = residual
        self.causalConv = CausalConv1d(in_chs,2*out_chs,kernel_size,mask_type)

        self.residualConv = nn.Conv1d(out_chs,out_chs, 1)

        self.gate = GatedActivation()

        if self.residual:
            assert in_chs == out_chs

    def forward(self,x):
        out = self.causalConv(x)
        out = self.gate(out)

        if self.residual:
            out = self.residualConv(out) + x
        else:
            out = self.residualConv(out)

        return out

class GatedPixelCNN(nn.Module):
    def __init__(self,args):
        super(GatedPixelCNN,self).__init__()
        self.embedding = nn.Embedding(args.embed_number, args.pixel_embed_dim)
        self.layers = nn.ModuleList()
        for i in range(args.n_layers):
            mask_type = 'causal' if i == 0 else 'not causal'
            kernel_size = 5
            residual = False if i == 0 else True
            self.layers.append(
                GatedCausalConv1d(args.pixel_embed_dim,args.pixel_embed_dim,kernel_size,mask_type,residual)
            )

        # Add the output layer
        self.output_conv = nn.Sequential(
            nn.Conv1d(args.pixel_embed_dim,256, 1),
            nn.ReLU(True),
            nn.Conv1d(256,args.embed_number,1)
        )

        self.device = args.device

    def forward(self,x):
        # x (batch_size,src_lens)
        #print("x size:",x.size())
        x = self.embedding(x)

        #x (batch_size,src_lens,pixel_embed_dim)
        x = x.permute(0,2,1)
        
        #x (batch_size,pixel_embed_dim,src_lens)
        
        for i, layer in enumerate(self.layers):
            x = layer(x)
            
        # return (batch_size,args.embed_number,src_lens)
        return self.output_conv(x)

    def generate(self,batch_size,src_lens):
        latent = torch.zeros((batch_size,src_lens),dtype=torch.int64, device=self.device)

        for i in range(src_lens):
            logits = self.forward(latent)
            probs = F.softmax(logits[:, :, i], -1)

            latent.data[:,i].copy_(probs.multinomial(1).squeeze().data)

        return latent
        

        

        
        
        
        
