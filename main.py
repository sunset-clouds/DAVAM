import sys
import os
import time
import numpy as np

import torch
from torch import nn, optim


from model import DisCreteVAE
from  model_pixelcnn import GatedPixelCNNPrior
from module_pixelcnn import GatedPixelCNN
from util import Logger
import argparser

def main(args):
    model = DisCreteVAE(args).to(args.device)
    prior = GatedPixelCNN(args).to(args.device)
    if args.stage==1:
        ###training Discrete VAE
        model.train_model(args,train_teacher_forcing_ratio=args.train_tf_ratio, eval_teacher_forcing_ratio=args.eval_tf_ratio)
    elif args.stage==2:
        ####training the pixelcnn prior
        priormodel = GatedPixelCNNPrior(args,model,prior).to(args.device)
        priormodel.train_model(args)
    elif args.stage==3:
        #### generating sentence 
        priormodel = GatedPixelCNNPrior(args,model,prior).to(args.device)

        priormodel.FinalTestKL_Loss(args)
        #priormodel.FinalSentenceGenerating(args)
        


if __name__ == '__main__':
    args = argparser.parse_arg()
    sys.stdout = Logger(args)
    # print args
    dict_args = vars(args)
    for k, v in zip(dict_args.keys(), dict_args.values()):
        print("{0}: {1}".format(k, v))
    
    main(args)


