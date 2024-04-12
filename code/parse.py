import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    # datasst
    parser.add_argument('--dataset', type=str, default='taobao',
                        help="dataset select")
    parser.add_argument('--behaviors', type=str, default="['click', ['fav', 'cart'], 'buy']", help="behaviors order")
    parser.add_argument('--out_behavior', type=int, default=-1, help="select a result of behavior to output")
    parser.add_argument('--seed', type=int, default=8888, help='random seed')
    parser.add_argument('--gpu_id', type=int, default="0", help="gpu id select")
    
    # mdoel param
    parser.add_argument('--recdim', type=int,default=64,help="the embedding size of POGCN")
    parser.add_argument('--layer', type=int,default=3,help="the layer num of POG")
    parser.add_argument('--model', type=str, default='POGCN', help='model select')
    parser.add_argument('--dropout', type=int,default=0, help="using the dropout or not")
    parser.add_argument('--keep_prob', type=float,default=0.6, help="the batch size for bpr loss training procedure")
    parser.add_argument('--level', type=float, default=1.0, help="the distance of different important level")
    parser.add_argument('--sample_level', type=float, default=1.0, help="the sample rate of different important level")
    # train and test
    parser.add_argument('--epochs', type=int,default=200)
    parser.add_argument('--bpr_batch', type=int,default=2048,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--lr', type=float,default=0.001,
                        help="the learning rate")
    parser.add_argument('--decay', type=float,default=1e-4,
                        help="the weight decay for l2 normalizaton")
    parser.add_argument('--testbatch', type=int,default=100,
                        help="the batch size of users for testing")
    
    parser.add_argument('--topks', nargs='?',default="[20]",
                        help="@k test list")
    
    parser.add_argument('--multicore', type=int, default=0, help='whether we use multiprocessing or not in test')
    
    return parser.parse_args()
