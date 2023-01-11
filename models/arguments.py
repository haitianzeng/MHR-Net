import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Training script')

    # General arguments
    parser.add_argument('-d', '--dataset', default='h36m', type=str, metavar='NAME', help='target dataset')  # h36m or humaneva
    parser.add_argument('-k', '--keypoints', default='cpn_ft_h36m_dbb', type=str, metavar='NAME', help='2D detections to use')
    parser.add_argument('-str', '--subjects-train', default='S1,S5,S6,S7,S8', type=str, metavar='LIST',
                        help='training subjects separated by comma')
    parser.add_argument('-ste', '--subjects-test', default='S9,S11', type=str, metavar='LIST', help='test subjects separated by comma')
    parser.add_argument('-a', '--actions', default='*', type=str, metavar='LIST',
                        help='actions to train/test on, separated by comma, or * for all')
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='saved checkpoint directory')
    parser.add_argument('--teacher_checkpoint', default='checkpoint/teacher', type=str, metavar='PATH',
                    help='teacher checkpoint load path')
    parser.add_argument('-r', '--resume', default='', type=str, metavar='FILENAME',
                        help='checkpoint to resume (file name)')
    parser.add_argument('-decay', '--epoch-lr-decay', default=15, type=int, metavar='N',
                        help='epoch number to decay lr')
    parser.add_argument('-nw', '--num_workers', default=2, type=int, metavar='N')
    parser.add_argument('--reploss-weight', default=5.0, type=float, metavar='M', help='Reprojection loss weight')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='optimizer momentum')
    parser.add_argument('--weight-decay', default=0.0005, type=float, metavar='D', help='optimizer weight decay')
    parser.add_argument('--subset', default=1, type=float, metavar='FRACTION', help='reduce dataset size by fraction')
    parser.add_argument('--downsample', default=1, type=int, metavar='FACTOR', help='downsample frame rate by factor')
    parser.add_argument('--no-eval', action='store_true', help='disable epoch evaluation while training')
    parser.add_argument('--evaluate', action='store_true', help='enable evaluation while testing')
    parser.add_argument('--vis_results', action='store_true', help='enable visualization while testing')
    parser.add_argument('-lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('-lrd', '--lr-decay', default=0.5, type=float, metavar='LR', help='learning rate decay per epoch')

    # Model arguments
    parser.add_argument('-hp', '--hparams', default='n_keypoints=17,weight_init_std=0.01,n_fully_connected=1024,n_layers=6,dict_basis_size=10',
                        type=str, metavar='N', help='model hyper parameters')
    parser.add_argument('-e', '--epochs', default=50, type=int, metavar='N', help='number of training epochs')
    parser.add_argument('-b', '--batch_size', default=128, type=int, metavar='N', help='batch size in terms of predicted frames')
    parser.add_argument('--seed', default=666, type=int, metavar='N', help='seed for training')

    args = parser.parse_args()
    # Check invalid configuration
    if args.resume and args.evaluate:
        print('Invalid flags: --resume and --evaluate cannot be set at the same time')
        exit()

    return args
