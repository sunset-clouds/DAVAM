import math
import torch
import torch.nn as nn
import random
from VectorQuantize import VectorQuantize
from torch.nn import functional as F

class uniform_initializer(object):
    def __init__(self, stdv):
        self.stdv = stdv
    def __call__(self, tensor):
        nn.init.uniform_(tensor, -self.stdv, self.stdv)

class xavier_normal_initializer(object):
    def __call__(self, tensor):
        nn.init.xavier_normal_(tensor)


class LSTMEncoder(nn.Module):
    def __init__(self,args,vocab_size):
        super(LSTMEncoder,self).__init__()
        self.embed = nn.Embedding(vocab_size,args.embed_dim)
        self.lstm = nn.LSTM(input_size = args.embed_dim,
                            hidden_size = args.encoder_hidden_dim,
                            num_layers = 1,
                            batch_first = True)
        
        self.VQembeded = VectorQuantize(dim=args.vq_dim,n_embed=args.embed_number,decay=0.99,eps=1e-5)

        self.linear = nn.Linear(args.encoder_hidden_dim,args.vq_dim,bias=False)

        self.reset_parameters()
        
    def reset_parameters(self):
        model_init = uniform_initializer(0.01)
        emb_init = uniform_initializer(0.1)

        for param in self.parameters():
            model_init(param)
        emb_init(self.embed.weight)

    def forward(self,inputs):
        #input:[batch_size,src_len]
        #word_embed:[batch size,src_len,emb dim]
        word_embed = self.embed(inputs)
        
        #hidden_states:[batch size,src_len,encoder_hidden_dim]
        hidden_states,_= self.lstm(word_embed)

        #vq state:[batch size,src_len,vq_dim]
        vq_states = self.linear(hidden_states)
        
        #VQembed [batch size,src_len,vq_dim] loss_VQ [batch_size]
        VQembed,loss_VQ,_ = self.VQembeded(vq_states)

        ###remove 
        #last_VQembed [batch size,vq_dim]
        #last_VQembed = hidden_VQembed[:,-1,:]

        ###Verify the size the vq_states,VQembed,last_VQembed####
        #print("vq_states size:",vq_states.size())
        #print("VQembed size:",VQembed.size())

        return VQembed,loss_VQ

    def obtain_embed_ind(self,inputs):
        word_embed = self.embed(inputs)
        
        #hidden_states:[batch size,src_len,encoder_hidden_dim]
        hidden_states,_= self.lstm(word_embed)

        #vq state:[batch size,src_len,vq_dim]
        vq_states = self.linear(hidden_states)
        
        _,_,embed_ind = self.VQembeded(vq_states)
        
        ##return (batch_size,src_lens)
        return embed_ind

    def obtain_embedding_from_ind(self,embed_ind):
        embeddings = self.VQembeded.embed_code(embed_ind)

        ##return (batch_size,src_lens,vq_dim)
        return embeddings

class Attention(nn.Module):
    def __init__(self, args):
        super(Attention,self).__init__()
        self.attn = nn.Linear(2*args.decoder_hidden_dim,args.decoder_hidden_dim)
        self.v = nn.Parameter(torch.rand(args.decoder_hidden_dim))
        
    def forward(self, hidden, hidden_states):
        
        #hidden = [batch size,decoder_hidden_dim]
        #hidden_states = [batch size,src_len,decoder_hidden_dim]
        batch_size = hidden_states.size(0)
        src_len = hidden_states.size(1)
        #repeat encoder hidden state src_len times
        #hidden = [batch size, src len, decoder_hidden_dim]
        #hidden_states = [batch size,src_len,decoder_hidden_dim]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        #energy = [batch size, src len,decoder_hidden_dim]
        energy = torch.tanh(self.attn(torch.cat((hidden, hidden_states), dim = 2))) 
        
        #energy = [batch size,decoder_hidden_dim,src len]
        energy = energy.permute(0, 2, 1)

        #v = [dec hid dim]--->[batch size, 1, decoder_hidden_dim]
        v = self.v.repeat(batch_size, 1).unsqueeze(1)

        #attention= [batch size, src len]
        attention = torch.bmm(v, energy).squeeze(1)
        
        return F.softmax(attention, dim=1)

class LSTMDecoder(nn.Module):
    def __init__(self,args,vocab):
        super(LSTMDecoder,self).__init__()
        self.vocab_size = len(vocab) # no padding when setting padding_idx to -1
        self.vocab = vocab
        self.embed = nn.Embedding(self.vocab_size,args.embed_dim,padding_idx=-1)
        self.dropout_in = nn.Dropout(args.dropout_rate)
        self.dropout_out = nn.Dropout(args.dropout_rate)

        self.attention = Attention(args)
        
        self.device = args.device

        self.trans_linear = nn.Linear(args.vq_dim,args.decoder_hidden_dim,bias=False)
        #concatenate z with input
        self.lstm =nn.LSTMCell(input_size=args.embed_dim + 2*args.decoder_hidden_dim,
                            hidden_size=args.decoder_hidden_dim)

        #prediction layer
        self.pred_linear = nn.Linear(2*args.decoder_hidden_dim,self.vocab_size,bias=False)

        # vocab_mask[vocab['<pad>']] = 0
        vocab_mask = torch.ones(self.vocab_size,device=self.device)
        
        self.criterion = nn.CrossEntropyLoss(weight=vocab_mask, reduce=False)

        self.reset_parameters()

    def reset_parameters(self):
        model_init = uniform_initializer(0.01)
        emb_init = uniform_initializer(0.1)

        for param in self.parameters():
            model_init(param)
        emb_init(self.embed.weight)

    def forward(self,inputs,VQembed,teacher_forcing_ratio = 0.5):
        batch_size,seq_lens = inputs.size()

        outputs = torch.zeros(batch_size,seq_lens,self.vocab_size,device=self.device)

        #VQembed: [batch_size,seq_lens,vq_dim]
        #hidden_VQembed:[batch_size,seq_lens,decoder_hidden_dim]
        #last_hidden：[batch_size,decoder_hidden_dim]
        hidden_VQembed = self.trans_linear(VQembed)
        last_hidden = hidden_VQembed[:,-1,:]

        #hidden_VQembed =  torch.tanh(hidden_VQembed)

        ###Verify the size of hidden_VQembed and last_hidden
        #print("hidden_VQembed size:",hidden_VQembed.size())
        #print("last_hidden size:",last_hidden.size())
        
        #c_init,h_init [batch_size,hidden_dim]
        c_init = last_hidden
        h_init = torch.tanh(c_init)

        decoder_hidden = (h_init, c_init)

        hidden_attention = h_init
        
        #first input to the decoder is the <sos> tokens
        input_split = inputs[:,0]
        
        for time_step in range(1,seq_lens):
            #input_split might come from the groundtruth or last prediction
            #according to teacher_forcing_ratio
            #input_split [batch_size]
            
            #word_embed [batch_size,embed_dim]
            word_embed = self.dropout_in(self.embed(input_split))

            #Attention
            # hidden_attention after tanh
            #hidden_VQembed = [batch size,src_len,decoder_hidden_dim]
            #attn [batch size, src len]
            attn = self.attention(hidden_attention,hidden_VQembed)

            #[batch size, src len] ---> [batch size, 1, src len]
            attn = attn.unsqueeze(1)

            #weighted = [batch size, 1, encoder_hidden_dim]--->[batch size,decoder_hidden_dim]
            weighted = torch.bmm(attn,hidden_VQembed).squeeze(1)
            
            #word_concat [batch_size,embed_dim+encoder_hidden_dim]
            word_concat = torch.cat((word_embed,weighted,h_init),-1)

            #h_state,c_state [batch_size,decoder_hidden_dim]
            #decoder_hidden = (h_state,c_state)
            h_state,c_state = self.lstm(word_concat,decoder_hidden)
            decoder_hidden = (h_state,c_state)

            hidden_attention = h_state

            h_state = self.dropout_out(h_state)
            
            #output_logits [batch_size,hidden_dim]
            output_logits = self.pred_linear(torch.cat((h_state,weighted),dim=-1))

            #place predictions in a tensor holding predictions for each token
            outputs[:,time_step,:] = output_logits
            #get the highest predicted token from our predictions
            toptoken = output_logits.argmax(dim=1)

            #teacher_force decide whether to use ground truth information or not
            #if teacher_force = 1, use the ground truth information,
            #if teacher_force = 0, don't use the ground truth information and use
            #the last time step prediction
            teacher_force = random.random() < teacher_forcing_ratio
            input_split = inputs[:,time_step] if teacher_force else toptoken
            
        return outputs

    def reconstruction_loss(self,inputs,predicted):
        #remove start symbol<eos> [batch_size,seqlens-1]
        tgt = inputs[:,1:]
        batch_size = inputs.size(0)

        #print("tgt.size:",tgt.size())
        #remove start logits(all is zero)[batch,seqlens-1,vocab_size]
        predicted = predicted[:,1:,:].contiguous()
        #print("predicted.size:",predicted.size())
        
        #[batch*(seqlens-1),vocab_size]
        predicted = predicted.view(-1,self.vocab_size)
        
        #[batch*(seqlens-1)]
        tgt = tgt.contiguous().view(-1)
        loss = self.criterion(predicted,tgt)

        loss = loss.view(batch_size,-1).sum(-1)
        
        ##return loss size[batch_size]
        return loss

    def GenerateSamples(self,latent_variables):
        batch_size = latent_variables.size(0)
        seq_lens = latent_variables.size(1)

        decoded_batch = [[] for _ in range(batch_size)]

        hidden_variables = self.trans_linear(latent_variables)
        last_hidden = hidden_variables[:,-1,:]

        #hidden_variables =  torch.tanh(hidden_variables)
        
        #c_init,h_init [batch_size,hidden_dim]
        c_init = last_hidden
        h_init = torch.tanh(c_init)

        decoder_hidden = (h_init, c_init)

        hidden_attention = h_init
        
        #first input to the decoder is the <sos> tokens
        input_split = torch.tensor([self.vocab["<s>"]] * batch_size, dtype=torch.long,device=self.device)
        
        for time_step in range(1,seq_lens):
            #input_split might come from last prediction
            #input_split [batch_size]
            
            #word_embed [batch_size,embed_dim]
            word_embed = self.dropout_in(self.embed(input_split))

            #Attention
            # hidden_attention after tanh
            #hidden_VQembed = [batch size,src_len,decoder_hidden_dim]
            #attn [batch size, src len]
            attn = self.attention(hidden_attention,hidden_variables)

            #[batch size, src len] ---> [batch size, 1, src len]
            attn = attn.unsqueeze(1)

            #weighted = [batch size, 1, encoder_hidden_dim]--->[batch size,decoder_hidden_dim]
            weighted = torch.bmm(attn,hidden_variables).squeeze(1)
            
            #word_concat [batch_size,embed_dim+encoder_hidden_dim]
            word_concat = torch.cat((word_embed,weighted,h_init),-1)

            #h_state,c_state [batch_size,decoder_hidden_dim]
            #decoder_hidden = (h_state,c_state)
            h_state,c_state = self.lstm(word_concat,decoder_hidden)
            decoder_hidden = (h_state,c_state)

            hidden_attention = h_state

            h_state = self.dropout_out(h_state)
            
            #output_logits [batch_size,hidden_dim]
            output_logits = self.pred_linear(torch.cat((h_state,weighted),dim=-1))
                                       
            #get the highest predicted token from our predictions
            toptoken = output_logits.argmax(dim=1)
            for i in range(batch_size):
                decoded_batch[i].append(self.vocab.id2word(toptoken[i].item()))
                                       
            input_split = toptoken
            
        return decoded_batch
        

        
                                

