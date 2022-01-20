import argparse
from algorithm import ProbsAlgo


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--i', type=str, default='./data/test.csv',
                        help='path to csv file with true labels')
    parser.add_argument('--n', type=int, default=100,
                        help='number of Monte Carlo method')
    parser.add_argument('--o', type=str, default='./data/test.png',
                        help='path to save the plots')
    parser.add_argument('--p', nargs=3, type=float, default=[0.34, 0.33, 0.33],
                        help='probability density for 3 classes')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    probs_algo = ProbsAlgo(path_to_data=args.i, probs=args.p, n_iter=args.n)
    probs_algo.plot_and_save_result(args.o)


if __name__ == "__main__":
    main()
