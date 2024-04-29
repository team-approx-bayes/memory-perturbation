import os
import sys
import pickle
import argparse
import numpy as np

import torch
from torch import nn
from torch.nn.utils import parameters_to_vector
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam, AdamW

from ivon import IVON as IBLR

sys.path.append("..")
from lib.models import get_model
from lib.datasets import get_dataset
from lib.utils import get_quick_loader, predict_train, predict_test, flatten, get_estimated_nll
from lib.variances import get_covariance_from_iblr, get_covariance_from_adam, get_pred_vars_laplace, get_pred_vars_optim

def get_args():
    parser = argparse.ArgumentParser(description='Leave-one-out estimation with sensitivities from MPE. '
                                                 'Predicting generalization during training.')

    # Experiment
    parser.add_argument('--name_exp', default='fmnist_lenet_sgd_kfac', type=str, help='name of experiment')

    # Data, Model
    parser.add_argument('--dataset', default='FMNIST', choices=['MNIST', 'FMNIST', 'CIFAR10'])
    parser.add_argument('--model', default='lenet', choices=['large_mlp', 'small_mlp', 'lenet', 'cnn_deepobs','resnet20_frn'])

    # Optimization
    parser.add_argument('--optimizer', default='sgd', choices=['sgd', 'iblr','adam', 'adamw'])
    parser.add_argument('--lr', default=1e-2, type=float, help='learning rate')
    parser.add_argument('--lrmin', default=1e-4, type=float, help='min learning rate of scheduler')
    parser.add_argument('--bs', default=256, type=int, help='batch size')
    parser.add_argument('--epochs', default=100, type=int, help='number of epochs')
    parser.add_argument('--delta', default=60, type=float, help='L2-regularization parameter')

    # IBLR
    parser.add_argument('--hess_init', default=0.1, type=float, help='Hessian initialization')

    # Variance computation
    parser.add_argument('--optim_var', dest='optim_var', action='store_true', help='variance from optimizer (IBLR, Adam)')
    parser.add_argument('--var_version', default='kfac', choices=['kfac', 'diag'], help='Laplace-GGN matrix structure', )
    parser.add_argument('--bs_jacs', default=500, type=int, help='Jacobian batch size for variance computation')

    # Defaults
    parser.set_defaults(optim_var=False)
    return parser.parse_args()

def train_one_epoch_iblr(net, optim):
    net.train()
    running_loss = 0
    for X, y in trainloader:
        with optim.sampled_params(train=True):
            optim.zero_grad()
            fs = net(X)
            loss = criterion(fs, y)
            loss.backward()
        optim.step()
        running_loss += loss.item()
    scheduler.step()
    return net, optim

def train_one_epoch_sgd_adam(net, optim):
    net.train()
    running_loss = 0
    for X, y in trainloader:
        def closure():
            optim.zero_grad()
            fs = net(X)
            loss_ = criterion(fs, y)
            if args.optimizer == 'adamw':
                reg_ = 0
            else:
                p_ = parameters_to_vector(net.parameters())
                reg_ = 1/2 * args.delta * p_.square().sum()
            loss = loss_ + (1/n_train)*reg_
            loss.backward()
            return loss, fs
        loss, fs = optim.step(closure)
        running_loss += loss.item()
    scheduler.step()
    return net, optim

def get_optimizer():
    if args.optimizer == 'adam':
        optim = Adam(net.parameters(), lr=args.lr, weight_decay=0)
    elif args.optimizer == 'adamw':
        optim = AdamW(net.parameters(), lr=args.lr, weight_decay=args.delta / n_train)
    elif args.optimizer == 'iblr':
        optim = IBLR(net.parameters(), lr=args.lr, mc_samples=1, ess=n_train, weight_decay=args.delta/n_train,
                      beta1=0.9, beta2=0.99999, hess_init=args.hess_init)
    elif args.optimizer == 'sgd':
        optim = SGD(net.parameters(), lr=args.lr, momentum=0.9)
    else:
        raise NotImplementedError
    return optim

def get_prediction_vars(optim):
    if args.optim_var:  # Variances from optimizer state
        if args.optimizer == 'adam':
            sigma_sqrs = get_covariance_from_adam(optim, args.delta, n_train)
        elif args.optimizer == 'iblr':
            sigma_sqrs = get_covariance_from_iblr(optim)
        else:
            raise NotImplementedError
        sigma_sqrs = torch.asarray(flatten(sigma_sqrs)).to(device)
        vars = get_pred_vars_optim(net, trainloader_vars, sigma_sqrs, device)

    else:  # Laplace variance approximation
        if args.var_version == 'kfac':
            vars = get_pred_vars_laplace(net, trainloader_vars, args.delta, nc, version='kfac')
        elif args.var_version == 'diag':
            vars = get_pred_vars_laplace(net, trainloader_vars, args.delta, nc, version='diag')
        else:
            raise NotImplementedError

    return vars, optim

if __name__ == "__main__":
    args = get_args()
    print(args)

    evaluate_epochs = [1,2,5,10,20,30,40,50,75,100,150,200,250,300]

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

    # Model
    net = get_model(args.model, nc, input_size, device, seed)

    # Optimizer
    optim = get_optimizer()

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs, eta_min=args.lrmin)

    # Training and computing sensitivities in intervals
    residuals_list, vars_list, logits_list = [], [], []
    test_accs, test_nlls, epochs_list = [], [], []
    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}\n-------------------------------")

        # Train for one epoch
        if args.optimizer == 'iblr':
            net, optim = train_one_epoch_iblr(net, optim)
        else:
            net, optim = train_one_epoch_sgd_adam(net, optim)

        # Evaluate on test data
        test_acc, test_nll = predict_test(net, testloader_eval, nc, te_targets, device)
        test_accs.append(test_acc), test_nlls.append(test_nll)
        print(f"Test Acc: {(100 * test_acc):>0.2f}%, Test NLL: {test_nll:>6f}")

        if (epoch+1) in evaluate_epochs:
            epochs_list.append(epoch + 1)

            # Evaluate on training data; residuals and logits
            residuals, logits, train_acc, train_nll = predict_train(net, trainloader_eval, nc, tr_targets, device, return_logits=True)
            residuals_list.append(residuals), logits_list.append(logits)
            print(f"Train Acc: {(100 * train_acc):>0.2f}%, Train NLL: {train_nll:>6f}")
            print("Computing sensitivities...")

            # Prediction variances
            vars, optim = get_prediction_vars(optim)
            vars_list.append(vars)

    residuals_list, vars_list, logits_list = np.asarray(residuals_list), np.asarray(vars_list), np.asarray(logits_list)

    # Estimate test-NLL with approximate leave-one-out cross-validation
    estimated_nll = get_estimated_nll(nc, residuals_list, vars_list, logits_list, tr_targets)

    results_dict = {
                    'estimated_nll': estimated_nll,
                    'test_accs': test_accs,
                    'test_nlls': test_nlls,
                    'epochs': np.asarray(epochs_list),
                    'args': args,
                     }
    dir = 'pickles/'
    os.makedirs(os.path.dirname(dir), exist_ok=True)
    with open(dir + args.name_exp + '_iter.pkl', 'wb') as f:
        pickle.dump(results_dict, f, pickle.HIGHEST_PROTOCOL)