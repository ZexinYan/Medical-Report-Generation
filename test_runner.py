from tester import Tester
import argparse
import torch


if __name__ == '__main__':
    # Testing settings
    parser = argparse.ArgumentParser(description='PyTorch To test Chest-Xray by using densenet')
    parser.add_argument('--model', type=str, default='DenseNet',
                        help='The model name [DenseNet121, DenseNet161, DenseNet169, '
                             'DenseNet201, CheXNet, ResNet18, ResNet34, ResNet50,'
                             ' ResNet101, ResNet152, VGG191]')
    parser.add_argument('--images-dir', type=str, default='../images',
                        help='the path of the images directory')
    parser.add_argument('--test-csv', type=str, default='',
                        help='the path of the test csv')
    parser.add_argument('--mode', type=str, default='test',
                        help='the mode of testing')
    parser.add_argument('--weight-dir', type=str, default='',
                        help='the path of trained model')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='the batch size when testing')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--reshape-size', type=int, default=224,
                        help='the size of the input image')
    parser.add_argument('--classes', type=int, default=156,
                        help='the #classes of target')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    tester = Tester(args=args)
    tester.test()
