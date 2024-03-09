import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Leave-one-out estimation with sensitivities from MPE. '
                                                 'Predicting generalization during training.')
    parser.add_argument('--name_exp', default='fmnist_lenet_sgd_kfac', type=str, help='name of experiment')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    # Load saved results
    dir = 'pickles/'
    file = open(dir + args.name_exp + '_iter.pkl', 'rb')
    results_dict = pickle.load(file)
    file.close()

    epochs = results_dict['epochs']
    estimated_nll = results_dict['estimated_nll']
    test_nlls = results_dict['test_nlls']
    args_experiment = results_dict['args']
    print(args_experiment)
    epochs_xaxis = np.arange(1, epochs[-1] + 1, 1)

    # Figure
    fontsize = 41
    linewidth = 13
    figsize = (11, 6.5)

    fig, ax = plt.subplots(figsize=figsize)

    plt.xlabel('Epochs', fontsize=fontsize, labelpad=15)
    ax.tick_params(axis='both', which='major', labelsize=11, labelcolor='gray')
    plt.yticks(fontsize=fontsize-2)
    plt.xticks(fontsize=fontsize-2)
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Plot results
    ln1 = ax.plot(epochs_xaxis, test_nlls, c='0.7', label='Test NLL', linewidth=linewidth-3, zorder=1)
    marker_style = dict(markerfacecolor='w', marker='s', fillstyle='full', markersize=23, markeredgewidth=5)
    ln2 = ax.plot(epochs, estimated_nll, 'k-',label='LOO-CV',linewidth=linewidth-2, zorder=10,**marker_style)
    
    # Legend
    lns = ln1+ln2
    labels = [l.get_label() for l in lns]
    plt.legend(lns, labels, loc='upper right', fontsize=fontsize)
    
    # Save figure
    dir = 'plots/'
    os.makedirs(os.path.dirname(dir), exist_ok=True)
    plt.tight_layout()
    plt.savefig(dir + args.name_exp + '_iter.pdf', format="pdf", bbox_inches='tight')
    plt.show()
    plt.close()



