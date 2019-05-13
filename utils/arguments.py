import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Face resolution regression')
    parser.add_argument('--dataDir', default='high-res', help='data folder, must contain folder with name blur*')
    parser.add_argument('--saveDir', required=True, help='path to save model checkpoint and tensorboard file')
    parser.add_argument('--resume', action='store_true', default=False, help='resume previous training')
    parser.add_argument('--gpu', action='store_ture', default=False, help='use gpu')

    parser.add_argument('--lr', default=1e-4, help='learning rate')
    parser.add_argument('--epochs', default=100, help='training epochs')

    return parser.parse_args()