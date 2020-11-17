import math
import torch
import torch.nn as nn
import sys
from dataloader import MonoTextData
from module import LSTMDecoder,LSTMEncoder
from torch import nn, optim
import time
import numpy as np
import random
import os

class DisCreteVAE(nn.Module):
    def __init__(self,args):
        super(DisCreteVAE,self).__init__()
        self.train_data = MonoTextData(args.train_data, label=args.label)
        self.vocab = self.train_data.vocab
        self.vocab_size = len(self.vocab)
        self.val_data = MonoTextData(args.val_data, label=args.label, vocab=self.vocab)
        self.test_data = MonoTextData(args.test_data, label=args.label, vocab=self.vocab)

        print('Train data: %d samples' % len(self.train_data))
        print('Val data: %d samples' % len(self.val_data))
        print('Test data: %d samples' % len(self.test_data))

        print('finish reading datasets, vocab size is %d' % len(self.vocab))
        print('dropped sentences: %d' % self.train_data.dropped)
        sys.stdout.flush()

        self.train_data_batch = self.train_data.create_data_batch(batch_size=args.batch_size,
                                                        device=args.device,
                                                        batch_first=True)

        self.val_data_batch = self.val_data.create_data_batch(batch_size=args.batch_size,
                                                    device=args.device,
                                                    batch_first=True)

        self.test_data_batch = self.test_data.create_data_batch(batch_size=args.batch_size,
                                                      device=args.device,
                                                      batch_first=True)

        self.encoder = LSTMEncoder(args,self.vocab_size)
        self.decoder = LSTMDecoder(args,self.vocab)

    def build_model(self,inputs,vq_weight=1.0,teacher_forcing_ratio=0.5):
        #input: [batch_size,src_len]
        #hidden_VQembed: [batch size,src_len,encoder_hidden_dim]
        #last_hidden: [batch size,decoder_hidden_dim]
        #loss_vq [batch_size]

        hidden_VQembed,loss_vq = self.encoder(inputs)

        predicted = self.decoder(inputs,hidden_VQembed,teacher_forcing_ratio)

        loss_rc = self.decoder.reconstruction_loss(inputs,predicted)

        loss = loss_rc + vq_weight*loss_vq
        return loss,loss_rc,loss_vq

    def return_embed_ind(self,inputs):
        embed_ind = self.encoder.obtain_embed_ind(inputs)

        return embed_ind

    def return_sentence_from_ind(self,embed_ind):
        latent_embed = self.encoder.obtain_embedding_from_ind(embed_ind)

        sentences = self.decoder.GenerateSamples(latent_embed)
        return sentences

    def eval_model(self,args,teacher_forcing_ratio = 0,verbose=True):
        report_vq_loss = report_rec_loss = 0
        report_num_words = report_num_sents = 0
        self.eval()

        for i in np.random.permutation(len(self.val_data_batch)):
            batch_data = self.val_data_batch[i]
            batch_size, sent_len = batch_data.size()

            # not predict start symbol
            report_num_words += (sent_len - 1) * batch_size
            report_num_sents += batch_size

            loss,loss_rc,loss_vq = self.build_model(batch_data,vq_weight=1.0,teacher_forcing_ratio=teacher_forcing_ratio)

            assert(not loss_rc.requires_grad)

            loss_rc = loss_rc.sum()
            loss_vq = loss_vq.sum()
            
            report_rec_loss += loss_rc.item()
            report_vq_loss += loss_vq.item()

        val_loss = (report_rec_loss  + report_vq_loss) / report_num_sents

        nll = report_rec_loss / report_num_sents
        vq = report_vq_loss / report_num_sents
        if verbose:
            print('Validation --- avg_loss: %.4f, vq: %.4f, recon: %.4f, nll: %.4f' % \
                   (val_loss, report_vq_loss / report_num_sents,
                    report_rec_loss / report_num_sents, nll))
            sys.stdout.flush()

        return val_loss, nll, vq

    def test_model(self,args,teacher_forcing_ratio = 0,verbose=True):
        report_vq_loss = report_rec_loss = 0
        report_num_words = report_num_sents = 0
        self.eval()
        for i in np.random.permutation(len(self.test_data_batch)):
            batch_data = self.test_data_batch[i]
            batch_size, sent_len = batch_data.size()

            # not predict start symbol
            report_num_words += (sent_len - 1) * batch_size
            report_num_sents += batch_size

            loss,loss_rc,loss_vq = self.build_model(batch_data,vq_weight=1.0,teacher_forcing_ratio=teacher_forcing_ratio)

            assert(not loss_rc.requires_grad)

            loss_rc = loss_rc.sum()
            loss_vq = loss_vq.sum()

            report_rec_loss += loss_rc.item()
            report_vq_loss += loss_vq.item()

        test_loss = (report_rec_loss  + report_vq_loss) / report_num_sents

        nll =  report_rec_loss / report_num_sents
        vq = report_vq_loss / report_num_sents
        if verbose:
            print('Test --- avg_loss: %.4f, vq: %.4f, recon: %.4f, nll: %.4f' % \
                   (test_loss, report_vq_loss / report_num_sents,
                    report_rec_loss / report_num_sents, nll))
            sys.stdout.flush()

        return test_loss, nll, vq

    def train_model(self,args,train_teacher_forcing_ratio = 0.5,eval_teacher_forcing_ratio = 0):
        self.train()
        opt_dict = {"not_improved": 0, "lr": args.lr_start, "best_loss": 1e4}
        log_niter = 100
        enc_optimizer = optim.SGD(self.encoder.parameters(), lr= args.lr_start, momentum=args.momentum)
        dec_optimizer = optim.SGD(self.decoder.parameters(), lr= args.lr_start, momentum=args.momentum)

        iter_ = decay_cnt = 0
        best_loss = 1e4
        best_vq = best_nll = 0

        start = time.time()
        vq_weight = args.vq_start
        anneal_rate = (args.vq_weight_max - args.vq_start) / (args.warm_up * (len(self.train_data) / args.batch_size))
        for epoch in range(args.epochs):
            report_vq_loss = report_rec_loss = 0
            report_num_words = report_num_sents = 0

            for i in np.random.permutation(len(self.train_data_batch)):
                batch_data = self.train_data_batch[i]
                batch_size, sent_len = batch_data.size()

                # not predict start symbol
                report_num_words += (sent_len - 1) * batch_size
                report_num_sents += batch_size

                # vq_weight = 1.0 and update epoch by epoch
                vq_weight = min(args.vq_weight_max, vq_weight + anneal_rate)

                enc_optimizer.zero_grad()
                dec_optimizer.zero_grad()

                loss,loss_rc,loss_vq = self.build_model(batch_data,vq_weight,train_teacher_forcing_ratio)

                loss = loss.mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), args.clip_grad)

                loss_rc = loss_rc.sum()
                loss_vq = loss_vq.sum()

                enc_optimizer.step()
                dec_optimizer.step()

                report_rec_loss += loss_rc.item()
                report_vq_loss += loss_vq.item()
                if iter_ % log_niter == 0:
                    train_loss = (report_rec_loss  + report_vq_loss) / report_num_sents

                    print('Job suffix: %s, epoch: %d, iter: %d, avg_loss: %.4f, vq: %.4f, recon: %.4f,' \
                           'time elapsed %.2fs' %
                           (args.job_suffix, epoch, iter_, train_loss, report_vq_loss / report_num_sents,
                           report_rec_loss / report_num_sents, time.time() - start))

                    sys.stdout.flush()

                    report_rec_loss = report_vq_loss = 0
                    report_num_words = report_num_sents = 0

                iter_ += 1

            ## save every x epoch
            #if epoch % args.save_every:
            #    saver_name = os.path.join(args.saver_dir, "DisCreteVAE_"+str(epoch)+".pt")
            #    torch.save(self.state_dict(),saver_name)

            print('vq weight %.4f' % vq_weight)

            self.eval()
            with torch.no_grad():
                loss, nll, vq = self.eval_model(args,eval_teacher_forcing_ratio)

            if loss < best_loss:
                print('update best loss')
                best_loss = loss
                best_nll = nll
                best_vq = vq

                ###save the best model
                saver_name = os.path.join(args.saver_dir, "DisCreteVAE_Best.pt")
                torch.save(self.state_dict(),saver_name)

            if loss > opt_dict["best_loss"]:
                opt_dict["not_improved"] += 1
                if opt_dict["not_improved"] >= args.decay_epoch and epoch >=15:
                    opt_dict["best_loss"] = loss
                    opt_dict["not_improved"] = 0
                    opt_dict["lr"] = opt_dict["lr"] * args.lr_decay

                    saver_name = os.path.join(args.saver_dir, "DisCreteVAE_Best.pt")
                    self.load_state_dict(torch.load(saver_name))

                    print('new lr: %f' % opt_dict["lr"])
                    decay_cnt += 1
                    enc_optimizer = optim.SGD(self.encoder.parameters(), lr=opt_dict["lr"], momentum=args.momentum)
                    dec_optimizer = optim.SGD(self.decoder.parameters(), lr=opt_dict["lr"], momentum=args.momentum)

            else:
                opt_dict["not_improved"] = 0
                opt_dict["best_loss"] = loss

            if decay_cnt == args.max_decay:
                break
            self.train()

        ####Test process
        self.eval()
        saver_name = os.path.join(args.saver_dir, "DisCreteVAE_Best.pt")
        self.load_state_dict(torch.load(saver_name))
        with torch.no_grad():
            loss, nll, vq = self.test_model(args,eval_teacher_forcing_ratio)






