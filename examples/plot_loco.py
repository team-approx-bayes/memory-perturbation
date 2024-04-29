import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle
import os

def get_args():
    parser = argparse.ArgumentParser(description='Plot the results of predicting generalization for leave-one-class-out with MPE.')
    parser.add_argument('--names_exp', default=['fmnist_mlp', 'fmnist_lenet'], type=str, help='names of experiments')
    parser.add_argument('--plot_name', default ='fmnist', type= str, help='name for saving figure')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    markers = ['o', 's']
    face = ['tab:red', 'None']
    colors = ['tab:red', 'tab:blue']
    nc = 10
    las = []

    fig = plt.figure(figsize=(6, 6))
    ax = plt.gca()
    ax.spines['bottom'].set_color('gray')
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_color('gray')
    ax.spines['left'].set_linewidth(1.5)

    # plot results
    for ix, name_exp in enumerate(args.names_exp):
        results_file = open('pickles/' + name_exp + '_leaveclassout.pkl', 'rb')
        results = pickle.load(results_file)
        results_file.close()

        org_acc = results['org_acc']
        perturb_acc = results['perturb_acc']
        est_nll = results['est_nll']
        true_nll = results['perturb_nll']

        acc_diff = org_acc - perturb_acc
        for c in np.arange(nc):
            ss = plt.scatter(true_nll[c], est_nll[c], s=10 + 1800 * acc_diff[c], marker=markers[ix],
                             edgecolors=colors[ix], facecolor=face[ix], alpha=0.8, linewidth=2)
            plt.annotate(str(c), (true_nll[c]+0.004, est_nll[c]+0.004), c=colors[ix], fontsize=20)
            if c == 0:
                las.append(ss)

    plt.legend(las, ('MLP', 'LeNet'), scatterpoints=1, loc='upper left', ncol=1, fontsize=15)
    plt.xlabel('Test Neg. Log-Likelihood (NLL)', c='k', fontsize=15)
    plt.ylabel('Estimated NLL with MPE',c='k', fontsize=15)
    plt.tick_params(axis='both', which='major', labelcolor='gray', width=10)
    plt.tight_layout()

    dir = 'plots/'
    os.makedirs(os.path.dirname(dir), exist_ok=True)
    plt.savefig(dir + args.plot_name +'_leaveclassout.pdf', format="pdf", bbox_inches='tight')
    plt.show()
    plt.close()
