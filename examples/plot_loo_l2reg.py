import os
import pickle
import matplotlib.pyplot as plt
from lib.utils import moving_average
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Leave-one-out estimation with sensitivities from MPE at convergence. '
                                                 'Tuning of the L2-regularization parameter.')
    parser.add_argument('--name_exp', default='mnist_mlp', type=str, help='name of experiment')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    # Smoothing window for moving average
    window = 4

    # Load saved results
    dir = 'pickles/'
    file = open(dir + args.name_exp + '_cv.pkl', 'rb')
    results_dict = pickle.load(file)
    file.close()

    deltas = results_dict['deltas']
    test_nlls = results_dict['test_nlls']
    estimated_nll = results_dict['estimated_nll']
    args_experiment = results_dict['args']
    print(args_experiment)
    # test_accs = results_dict['test_accs']
    # test_errors = (1 - test_accs) * 100

    # Get moving average
    test_nlls = moving_average(test_nlls, window)
    estimated_nll = moving_average(estimated_nll, window)
    # test_accs = moving_average(test_accs, window)
    deltas = deltas[window-1:]

    # Figure
    fontsize = 41
    linewidth = 9.5
    figsize = (11, 6)

    fig, ax = plt.subplots(figsize=figsize)

    plt.xlabel('$\delta$', fontsize=fontsize, labelpad=15)
    ax.tick_params(axis='both', which='major', labelsize=11, labelcolor='gray')
    plt.locator_params(axis='y', nbins=2)
    plt.yticks(fontsize=fontsize-2)
    plt.xticks(fontsize=fontsize-2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xscale('log')

    # Plot results
    ln1 = ax.plot(deltas, test_nlls, c='0.7', label='Test NLL', linewidth=linewidth+8)
    ln2 = ax.plot(deltas, estimated_nll, 'k-', label='LOO-CV', linewidth=linewidth-2)

    # Legend
    lns = ln1+ln2
    labels = [l.get_label() for l in lns]
    plt.legend(lns, labels, loc='upper center', fontsize=fontsize)

    # Save figure
    dir = 'plots/'
    os.makedirs(os.path.dirname(dir), exist_ok=True)
    plt.tight_layout()
    plt.savefig(dir + args.name_exp +'_cv.pdf', format="pdf")
    plt.show()


