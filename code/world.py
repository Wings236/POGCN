import os
import torch
from parse import parse_args
import multiprocessing

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
args = parse_args()

config = {}
config['bpr_batch_size'] = args.bpr_batch
config['latent_dim_rec'] = args.recdim
config['lightGCN_n_layers']= args.layer
config['test_u_batch_size'] = args.testbatch
config['multicore'] = args.multicore
config['lr'] = args.lr
config['decay'] = args.decay
config["dropout"] = args.dropout
config['keep_prob'] = args.keep_prob
config['level'] = args.level
config['sample_level'] = args.sample_level
config['out_behavior'] = args.out_behavior


GPU = torch.cuda.is_available()
device = torch.device('cuda:'+str(args.gpu_id) if GPU else "cpu")
CORES = multiprocessing.cpu_count() // 2
seed = args.seed

dataset = args.dataset
behaviors = eval(args.behaviors)
model_name = args.model

# train part
TRAIN_epochs = args.epochs
topks = eval(args.topks)


from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)


# Special color display
def cprint(words : str):
    print(f"\033[0;30;43m{words}\033[0m")
