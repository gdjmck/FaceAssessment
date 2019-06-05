import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Face resolution regression')
    parser.add_argument('--dataDir', default='high-res', help='data folder, must contain folder with name blur*')
    parser.add_argument('--saveDir', required=True, help='path to save model checkpoint and tensorboard file')
    parser.add_argument('--resume', action='store_true', default=False, help='resume previous training')
    parser.add_argument('--gpu', action='store_true', default=False, help='use gpu')

    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--epochs', default=100, help='training epochs')
    parser.add_argument('--test', action='store_true', default=False, help='run a test epoch')
    parser.add_argument('--test_split', type=float, default=0.2, help='ratio of testset to split train and test set')

    return parser.parse_args()
