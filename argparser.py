import argparse
import os
import torch
import numpy as np
import random
import string
import datetime

def parse_arg():
    parser = argparse.ArgumentParser(description='PyTorch Lagging VAE LSTM text generative model.')

    # job params
    parser.add_argument('--use_random', action='store_true', help='wheter to randomly generate seed')
    parser.add_argument('--seed', type=int, default=783435, metavar='S', help='random seed (also job id)')
    parser.add_argument('--gpu_id', default='0', help='config which gpu to use')
    parser.add_argument('--job_suffix', default='', help='randomly generated job id')

    # optimization parameters
    parser.add_argument('--clip_grad', type=float, default=5.0, help='clip_grad')
    parser.add_argument('--decay_epoch', type=int, default=2, help='decay_epoch')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='lr_decay')
    parser.add_argument('--max_decay', type=float, default=5.0, help='max_decay')
    parser.add_argument('--momentum', type=float, default=0, help='sgd momentum')
    parser.add_argument('--nsamples', type=int, default=1, help='number of samples for training')

    # annealing paramters
    parser.add_argument('--warm_up', type=int, default=10, help="number of annealing epochs")
    parser.add_argument('--vq_start', type=float, default=0.15, help="starting vq weight")
    parser.add_argument('--vq_weight_max', type=float, default=5.0, help="maxs vq weight")
    parser.add_argument('--lr_start', type=float, default=1.0, help="starting lr")

    # others
    parser.add_argument('--label', action='store_true', default=False)
    parser.add_argument('--UseGroundTruthInTest', action='store_false', default=True)
    parser.add_argument('--train_tf_ratio', type=float, default=1.0, help='teaching force ratio for training')
    parser.add_argument('--eval_tf_ratio', type=float, default=1.0, help='teaching force ratio for eval')
    parser.add_argument('--stage', type=int, default=1, help='1 represents train discreteVAE; 2 represents pixelCNN; 3 represents generate text',choices=[1, 2, 3])

    #model parameters
    parser.add_argument('--encoderType', type=str, default="lstm", help="type of rnn-based encoder")
    parser.add_argument('--decoderType', type=str, default="lstm", help="type of rnn-based decoder")
    parser.add_argument('--embed_number', type=int, default=512, help="the dimension of latent vector")
    parser.add_argument('--embed_dim', type=int, default=256, help="the dimension of embedding vector")
    parser.add_argument('--vq_dim', type=int, default=32, help="the dimension of encoder hidden state")
    parser.add_argument('--encoder_hidden_dim', type=int, default=256, help="the dimension of encoder hidden state")
    parser.add_argument('--decoder_hidden_dim', type=int, default=256, help="the dimension of decoder hidden state")
    parser.add_argument('--dropout_rate', type=float, default=0.5, help="the rate of decoder dropout rate")
    parser.add_argument('--batch_size', type=int, default=32, help="training batch size")
    parser.add_argument('--epochs', type=int, default=100, help="the maximum epoch of training")
    parser.add_argument('--test_nepoch', type=int, default=5, help="how many epoch to test once")

    #path parameters
    parser.add_argument('--dataset', default='ptb', help='config which dataset', choices=['yahoo', 'snli', 'ptb'])
    parser.add_argument('--train_data', type=str, default="datasets/ptb/train.txt", help="the path of training dataset")
    parser.add_argument('--val_data', type=str, default="datasets/ptb/valid.txt", help="the path of validation dataset")
    parser.add_argument('--test_data', type=str, default="datasets/ptb/test.txt", help="the path of testing dataset")
    parser.add_argument('--saver_dir', type=str, default='./saver', help="the path of saveing traing model")
    parser.add_argument('--log_dir', type=str, default='./saver', help="the path of saveing traing model")
    parser.add_argument('--save_every', type=int, default=20, help='save model every x epoch')

    #pixel cnn prior parameters
    parser.add_argument('--pixel_embed_dim', type=int, default=512, help="the embed_dim of latent variables")
    parser.add_argument('--n_layers', type=int, default=15, help="the number layers of gated causal convolution")
    parser.add_argument('--pixel_lr_start', type=float, default=0.5, help="the start learning to train pixelcnn prior")
    parser.add_argument('--pixel_epochs', type=int, default=50, help="the epochs to train pixelcnn network")
    parser.add_argument('--pixel_save_every', type=int, default=10, help="the epochs to train pixelcnn network")

    #generate sentence parameters
    parser.add_argument('--sentence_batch_size', type=int, default=32, help="the batch size of generateing sentence in stage 3")
    parser.add_argument('--sentence_src_lens', type=int, default=30, help="the lens of generating sentence in stage 3")
    parser.add_argument('--sentence_number', type=int, default=10, help="the lens of generating sentence in stage 3")
    
    args = parser.parse_args()

    # process args
    if args.use_random:
        args.seed = random.randint(0, 1e8)

    # config GPU
    print("Using GPU Id: %d" % int(args.gpu_id))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.device = torch.device("cuda")

    # makeup saver_dir and log_dir
    args.job_suffix = ''.join(random.sample(string.ascii_letters+string.digits, 5))
    if args.saver_dir == "./saver":
        args.saver_dir = os.path.join(args.saver_dir, args.dataset, \
                     '-'.join([str(args.seed), args.job_suffix]))
        args.log_dir = args.saver_dir

    if not os.path.exists(args.saver_dir):
        os.makedirs(args.saver_dir)

    ####Fixed the seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True

    return args

