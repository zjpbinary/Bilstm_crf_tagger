import argparse
parser = argparse.ArgumentParser(description='网络参数配置')
parser.add_argument("--gpu_id", default='cpu')
parser.add_argument("--embed_dim", default=300, type=int)
parser.add_argument('--lstm_dim', default=150, type=int)
parser.add_argument('--hid_dim', default=150, type=int)
parser.add_argument('--dropout', default=0.5, type=float)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--epoch', default=10, type=int)
parser.add_argument('--batch_size', default=32, type=int)
args = parser.parse_args()

