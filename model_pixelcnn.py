import math
import torch
import torch.nn as nn
import sys
from dataloader import MonoTextData
from module import LSTMDecoder,LSTMEncoder
from module_pixelcnn import GatedPixelCNN
from model import DisCreteVAE
from torch import nn, optim
import time
import numpy as np
import random
import os

class GatedPixelCNNPrior(nn.Module):
    def __init__(self,args,model,prior):
        super(GatedPixelCNNPrior,self).__init__()
        self.train_data = MonoTextData(args.train_data, label=args.label)
        self.vocab = self.train_data.vocab
        self.vocab_size = len(self.vocab)
        self.val_data = MonoTextData(args.val_data, label=args.label, vocab=self.vocab)
        self.test_data = MonoTextData(args.test_data, label=args.label, vocab=self.vocab)

        self.train_data_batch = self.train_data.create_data_batch(batch_size=args.batch_size,
                                                        device=args.device,
                                                        batch_first=True)

        self.val_data_batch = self.val_data.create_data_batch(batch_size=args.batch_size,
                                                    device=args.device,
                                                    batch_first=True)

        self.test_data_batch = self.test_data.create_data_batch(batch_size=args.batch_size,
                                                      device=args.device,
                                                      batch_first=True)
        self.model = model
        self.prior = prior
        with torch.no_grad():
            saver_name = os.path.join(args.saver_dir, "DisCreteVAE_Best.pt")
            self.model.load_state_dict(torch.load(saver_name))
            self.model.eval()
        self.criterion = nn.CrossEntropyLoss(reduce=False)
        self.args = args

    ##PixelCNN model NLL loss represent the KL Term
    def build_model(self,inputs):
        #logits(batch_size,args.embed_number,src_lens)
        #inputs (batch_size,src_lens)
        batch_size = inputs.size(0)
        logits = self.prior(inputs)

        #logits(batch_size,src_lens,args.embed_number)
        logits = logits.permute(0,2,1).contiguous()

        #target (batch_size*src_lens)
        target = inputs.contiguous().view(-1)

        logits = logits.view(-1,self.args.embed_number)
        
        kl = self.criterion(logits,target)
        kl = kl.view(batch_size,-1).mean(-1)
        
        return kl

    #Calculate the NLL loss and PPL and KL loss
    def eval_model(self,args,teacher_forcing_ratio=0,verbose=True):
        report_rec_loss = report_kl_loss = 0
        report_num_words = report_num_sents = 0
        self.prior.eval()
        self.model.eval()
        
        for i in np.random.permutation(len(self.val_data_batch)):
            batch_data = self.val_data_batch[i]
            batch_size, sent_len = batch_data.size()
            
            # not predict start symbol
            report_num_words += (sent_len - 1) * batch_size
            report_num_sents += batch_size

            with torch.no_grad():
                inds = self.model.return_embed_ind(batch_data)
                loss,loss_rc,loss_vq = self.model.build_model(batch_data,vq_weight=1.0,teacher_forcing_ratio=teacher_forcing_ratio)
                
            kl = self.build_model(inds)
            
            report_kl_loss += kl.sum().item()
            report_rec_loss += loss_rc.sum().item()
            
        nll = report_rec_loss / report_num_sents
        ppl = np.exp((report_kl_loss+report_rec_loss)/ report_num_words)
        kl = report_kl_loss/report_num_sents

        if verbose:
            print('Valid--- Rec: %.4f KL: %.4f PPL:%.4f' % (nll,kl,ppl))
            sys.stdout.flush()

            #print("sentence:",self.generateSentence())
        return kl

    def test_model(self,args,teacher_forcing_ratio=0,verbose=True):
        report_rec_loss = report_kl_loss = 0
        report_num_words = report_num_sents = 0
        self.prior.eval()
        self.model.eval()
        for i in np.random.permutation(len(self.test_data_batch)):
            batch_data = self.test_data_batch[i]
            batch_size, sent_len = batch_data.size()

            # not predict start symbol
            report_num_words += (sent_len - 1) * batch_size
            report_num_sents += batch_size

            with torch.no_grad():
                inds = self.model.return_embed_ind(batch_data)
                loss,loss_rc,loss_vq = self.model.build_model(batch_data,vq_weight=1.0,teacher_forcing_ratio=teacher_forcing_ratio)

            kl = self.build_model(inds)
            
        report_kl_loss += kl.sum().item()
        report_rec_loss += loss_rc.sum().item()
            
        nll = report_rec_loss / report_num_sents
        ppl = np.exp((report_kl_loss+report_rec_loss)/ report_num_words)
        kl = report_kl_loss/report_num_sents

        if verbose:
            print('Test--- Rec: %.4f KL: %.4f PPL:%.4f' % (nll,kl,ppl))
            sys.stdout.flush()

            #print("sentence:",self.generateSentence())
        return kl
             
    def train_model(self,args):
        self.prior.train()
        opt_dict = {"not_improved": 0, "lr":args.pixel_lr_start, "best_loss": 1e4}
        log_niter = 100

        optimizer = optim.SGD(self.prior.parameters(), lr=args.pixel_lr_start, momentum=args.momentum)
        iter_ = decay_cnt = 0
        best_loss = 1e4

        start = time.time()

        for epoch in range(args.pixel_epochs):
            report_kl_loss = 0
            report_num_sents = 0
            self.prior.train()
            for i in np.random.permutation(len(self.train_data_batch)):
                batch_data = self.train_data_batch[i]
                batch_size, sent_len = batch_data.size()

                report_num_sents += batch_size
                optimizer.zero_grad()

                with torch.no_grad():
                    self.model.eval()
                    inds = self.model.return_embed_ind(batch_data)

                kl = self.build_model(inds)

                loss = kl.mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.prior.parameters(), args.clip_grad)
                optimizer.step()
                
                report_kl_loss += kl.sum().item()

                if iter_ % log_niter == 0:
                    train_kl = report_kl_loss /report_num_sents
                    print('Job suffix: %s, epoch: %d, iter: %d, kl_loss:%.4f time elapsed %.2fs' %
                           (args.job_suffix, epoch, iter_,train_kl,time.time() - start))

                    sys.stdout.flush()
                    report_kl_loss = 0
                    report_num_sents = 0

                iter_+=1
                
            if epoch % args.pixel_save_every:
                saver_name = os.path.join(args.saver_dir, "PixelCNNPrior_"+str(epoch)+".pt")
                torch.save(self.prior.state_dict(),saver_name)

            self.prior.eval()
            with torch.no_grad():
                loss = self.eval_model(args,args.eval_tf_ratio)

            if loss < best_loss:
                print('update best loss')
                best_loss = loss

                ###save the best model
                saver_name = os.path.join(args.saver_dir, "PixelCNNPrior_Best.pt")
                torch.save(self.prior.state_dict(),saver_name)

            if loss > opt_dict["best_loss"]:
                opt_dict["not_improved"] += 1
                if opt_dict["not_improved"] >= args.decay_epoch and epoch >=15:
                    opt_dict["best_loss"] = loss
                    opt_dict["not_improved"] = 0
                    opt_dict["lr"] = opt_dict["lr"] * args.lr_decay

                    saver_name = os.path.join(args.saver_dir, "PixelCNNPrior_Best.pt")
                    self.prior.load_state_dict(torch.load(saver_name))

                    print('new lr: %f' % opt_dict["lr"])
                    decay_cnt += 1
                    optimizer = optim.SGD(self.prior.parameters(), lr=opt_dict["lr"], momentum=args.momentum)
            else:
                opt_dict["not_improved"] = 0
                opt_dict["best_loss"] = loss

            if decay_cnt == args.max_decay:
                break
            self.prior.train()

        self.prior.eval()
        saver_name = os.path.join(args.saver_dir, "PixelCNNPrior_Best.pt")
        self.prior.load_state_dict(torch.load(saver_name))
        with torch.no_grad():
            self.test_model(args,args.eval_tf_ratio)

    def generateSentence(self):
        #(batch_size,src_lens)
        latent_ind = self.prior.generate(self.args.sentence_batch_size,self.args.sentence_src_lens)
        sentences = self.model.return_sentence_from_ind(latent_ind)

        batch_size =  len(sentences)
        src_lens = len(sentences[0])
        sents =[]
        sent=""
        for i in range(batch_size):
            for j in range(src_lens):
                byte = sentences[i][j]

                if type(byte) ==bytes:
                    char = byte.decode(encoding='utf-8')
                else:
                    char = byte
                sent += char+" "
            
            sents.append(sent)
            sent =""
        return sents

    def FinalTestKL_Loss(self,args):
        with torch.no_grad():
            saver_name = os.path.join(args.saver_dir, "DisCreteVAE_Best.pt")
            self.model.load_state_dict(torch.load(saver_name))
            saver_name = os.path.join(args.saver_dir, "PixelCNNPrior_Best.pt")
            self.prior.load_state_dict(torch.load(saver_name))
            self.model.eval()
            self.prior.eval()
            loss = self.test_model(args,self.model)

    def FinalSentenceGenerating(self,args):
        with torch.no_grad():
            saver_name = os.path.join(args.saver_dir, "DisCreteVAE_Best.pt")
            self.model.load_state_dict(torch.load(saver_name))
            saver_name = os.path.join(args.saver_dir, "PixelCNNPrior_Best.pt")
            self.prior.load_state_dict(torch.load(saver_name))
            self.model.eval()
            self.prior.eval()

            print("FinalSentenceGenerating")

            for index in range(args.sentence_number):
                sentences = self.generateSentence()
                print("sentecne:",sentences)

                

                    

            
        

        


        
        
    






