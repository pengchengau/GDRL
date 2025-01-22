import argparse


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
    parser.add_argument('--gamma', type=float, default=0.995, metavar='G',
                        help='discount factor (default: 0.995)')
    parser.add_argument('--tau', type=float, default=0.97, metavar='G',
                        help='gae (default: 0.97)')
    parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                        help='l2 regularization regression (default: 1e-3)')
    parser.add_argument('--max-kl', type=float, default=1e-2, metavar='G',
                        help='max kl value (default: 1e-2)')
    parser.add_argument('--damping', type=float, default=1e-1, metavar='G',
                        help='damping (default: 1e-1)')
    parser.add_argument('--batch-size', type=int, default=2, metavar='N',
                        help='the batch size (default: 32)')
    parser.add_argument('--render', action='store_true',
                        help='render the environment')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='interval between training status logs (default: 10)')
    parser.add_argument('--U', type=int, default=3, metavar='N',
                        help='The total number of users (default: 3)')
    parser.add_argument('--N', type=int, default=16, metavar='N',
                        help='The total number of HAP (default: 16)')
    parser.add_argument('--L', type=int, default=16, metavar='N',
                        help='The total number of LEO (default: 16)')
    parser.add_argument('--total_step', type=int, default=100, metavar='N',
                        help='The number of mini-slot for each epoch (default: 20)')
    parser.add_argument('--T', type=int, default=100, metavar='N',
                        help='The number of total mini-slots (default: 100)')
    parser.add_argument('--num_dis_action', type=int, default=16, metavar='N',
                        help='The number of continuous representation of the discrete action(default: 6)')
    parser.add_argument('--num_con_action', type=int, default=16, metavar='N',
                        help='The number of continues action(default: 6)')
    parser.add_argument('--epsilon', type=int, default=0, metavar='N',
                        help='The exploration rate of the algorithm(default: 0.8)')
    args = parser.parse_args()
    return args
