import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Evolution of sensitivities during training.')
    parser.add_argument('--name_exp', default='fmnist_lenet', type=str, help='name of experiment')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    # Load saved results
    dir = 'pickles/'
    with open(dir + args.name_exp + '_evolving.pkl', 'rb') as file:
        results_dict = pickle.load(file)
    file.close()

    sensitivity = results_dict['sensitivity'][-1]
    args_experiment = results_dict['args']
    print(args_experiment)

    ranking = np.argsort(sensitivity)[::-1]
    numbins = 100

    # Figure
    fontsize = 16
    ax, fig = plt.subplots(figsize=(6, 6))
    ax = plt.gca()
    plt.xlabel('Example', fontsize=fontsize)
    plt.ylabel('Sensitivity', fontsize=fontsize)
    plt.box(False)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tick_params(axis='y', which='both', colors='gray')
    plt.tick_params(axis='x', which='major', colors='gray', width=2, length=14)
    plt.tick_params(axis='x', which='minor', colors='gray', width=1, length=8)
    ax.set_xticks(np.arange(numbins), minor=True)
    ax.set_xticklabels([])
    ax.invert_xaxis()

    # Plot results
    lws = [7, 6, 5, 4]
    colors = [0.6, 0.45, 0.25, 0.1]
    line_caption = ['First', 'Second', 'Third', 'Forth']

    c = 0
    for iteration in range(len(results_dict['sensitivity'])):
        sensitivity = results_dict['sensitivity'][iteration]
        meansensi = np.mean(sensitivity[ranking].reshape(numbins, -1), axis=1)
        plt.plot(np.arange(numbins), meansensi, '-', linewidth=lws[c], color = (colors[c], colors[c], colors[c]), label=line_caption[c])
        c += 1

    # Legend
    plt.legend(title='Training checkpoints', loc='upper left', fontsize=fontsize, title_fontsize=fontsize)

    # Save figure
    dir = 'plots/'
    os.makedirs(os.path.dirname(dir), exist_ok=True)
    plt.savefig(dir + args.name_exp +'_evolving.pdf', format="pdf", bbox_inches='tight')
    plt.show()
    plt.close()
