import os
import sys
import argparse
import pickle
import tqdm
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils import parameters_to_vector
from torch.optim import AdamW, SGD, Adam

sys.path.append("..")
from lib.models import get_model
from lib.datasets import get_dataset
from lib.utils import get_quick_loader, predict_train, predict_test, get_estimated_nll
from lib.variances import get_pred_vars_laplace

def get_args():
    parser = argparse.ArgumentParser(description='Leave-one-out estimation with sensitivities from MPE at convergence. '
                                                 'Tuning of the L2-regularization parameter.')

    # Experiment
    parser.add_argument('--name_exp', default='mnist_mlp', type=str, help='name of experiment')
    parser.add_argument('--n_exp', default=100, type=float, help='number of deltas evaluated')
    parser.add_argument('--deltas_start', default=0, type=int, help='delta start-value in log-space')
    parser.add_argument('--deltas_stop', default=3, type=int, help='delta end-value in log-space')

    # Data, Model
    parser.add_argument('--dataset', default='MNIST', choices=['MNIST', 'FMNIST', 'CIFAR10'])
    parser.add_argument('--model', default='large_mlp', choices=['large_mlp', 'lenet', 'cnn_deepobs'])

    # Optimization
    parser.add_argument('--optimizer', default='sgd', choices=['sgd', 'adam', 'adamw'])
    parser.add_argument('--lr', default=1e-2, type=float, help='learning rate')
    parser.add_argument('--lrmin', default=1e-3, type=float, help='min learning rate of scheduler')
    parser.add_argument('--bs', default=256, type=int, help='batch size')
    parser.add_argument('--epochs', default=500, type=int, help='number of epochs')

    parser.add_argument('--bs_jacs', default=50, type=int, help='Jacobian batch size for Laplace variance computation')

    parser.set_defaults()
    return parser.parse_args()

def train_network(net, delta):
    net.train()

    if args.optimizer == 'adamw':
        optim = AdamW(net.parameters(), lr=args.lr, weight_decay=delta / n_train)
    elif args.optimizer == 'adam':
        optim = Adam(net.parameters(), lr=args.lr)
    elif args.optimizer == 'sgd':
        optim = SGD(net.parameters(), lr=args.lr, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs, eta_min=args.lrmin)

    losses = []
    for _ in tqdm.tqdm(list(range(args.epochs))):
        running_loss = 0
        for X, y in trainloader:
            optim.zero_grad()
            X, y = X.float(), y
            fs = net(X)
            loss_ = criterion(fs, y)
            if args.optimizer == 'adamw':
                reg_ = 0
            else:
                p_ = parameters_to_vector(net.parameters())
                reg_ = 1 / 2 * delta * p_.square().sum()
            loss = loss_ + (1 / n_train) * reg_
            loss.backward()
            optim.step()

            running_loss += loss.item()
        losses.append(running_loss)
        scheduler.step()

    return net, losses

if __name__ == "__main__":
    args = get_args()
    print(args)
    
    deltas = np.logspace(args.deltas_start, args.deltas_stop, num=int(args.n_exp))

    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device', device)

    # Loss
    criterion = nn.CrossEntropyLoss(reduction='mean')

    # Data
    ds_train, ds_test = get_dataset(args.dataset)
    input_size = len(ds_train.data[0, :])**2
    nc = max(ds_train.targets) + 1
    n_train = len(ds_train)
    tr_targets, te_targets = torch.asarray(ds_train.targets), torch.asarray(ds_test.targets)

    # Dataloaders
    trainloader = get_quick_loader(DataLoader(ds_train, batch_size=args.bs)) # training
    trainloader_eval = DataLoader(ds_train, batch_size=args.bs, shuffle=False) # train evaluation
    testloader_eval = DataLoader(ds_test, batch_size=args.bs, shuffle=False) # test evaluation
    trainloader_vars = DataLoader(ds_train, batch_size=args.bs_jacs, shuffle=False) # variance computation

    # Train models with different L2-regularization parameters
    residuals_list, vars_list, logits_list = [], [], []
    test_accs, test_nlls = [], []
    for delta in deltas:
        print(f'Delta {delta}')
        print('--------------------------------')

        # Initialize network
        net = get_model(args.model, nc, input_size, device, seed)

        # Training
        net, _ = train_network(net, delta)

        # Evaluate on test data
        test_acc, test_nll = predict_test(net, testloader_eval, nc, te_targets, device)
        test_accs.append(test_acc), test_nlls.append(test_nll)
        print(f"Test Acc: {(100 * test_acc):>0.2f}%, Test NLL: {test_nll:>6f}")

        # Evaluate on training data
        residuals, logits, train_acc, train_nll = predict_train(net, trainloader_eval, nc, tr_targets, device, return_logits=True)
        residuals_list.append(residuals), logits_list.append(logits)
        print(f"Train Acc: {(100 * train_acc):>0.2f}%, Train NLL: {train_nll:>6f}")

        # Compute predictive Laplace variances
        vars_c = get_pred_vars_laplace(net, trainloader_vars, delta, nc, 'kfac', device=device)
        vars_list.append(vars_c)

    residuals_list, vars_list, logits_list = np.asarray(residuals_list), np.asarray(vars_list), np.asarray(logits_list)

    # Estimate test-NLL with approximate leave-one-out cross-validation
    estimated_nll = get_estimated_nll(nc, residuals_list, vars_list, logits_list, tr_targets)

    # Save results
    results_dict = {'estimated_nll': estimated_nll,
                    'deltas': deltas,
                    'test_accs': test_accs,
                    'test_nlls': test_nlls,
                    'args': args
                     }
    dir = 'pickles/'
    os.makedirs(os.path.dirname(dir), exist_ok=True)
    with open(dir + args.name_exp + '_cv.pkl', 'wb') as f:
        pickle.dump(results_dict, f, pickle.HIGHEST_PROTOCOL)


