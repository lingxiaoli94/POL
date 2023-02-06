import argparse

def parse_config_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--val', action='store_true', default=False)
    parser.add_argument('--restart', action='store_true', default=False)
    parser.add_argument('--problem_list', nargs='*')
    parser.add_argument('--method_list', nargs='*')
    parser.add_argument('--list_problems', action='store_true', default=False)
    parser.add_argument('--list_methods', action='store_true', default=False)
    parser.add_argument('--train_step', type=int, default=100000)
    parser.add_argument('--val_freq', type=int, default=-1)
    parser.add_argument('--instance_batch_size', type=int, default=32)
    parser.add_argument('--prior_batch_size', type=int, default=256)
    parser.add_argument('--num_particle', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    return args
