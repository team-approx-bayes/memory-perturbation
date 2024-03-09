import os
import sys
import pickle
import argparse
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.functional import F

sys.path.append("..")
from ivon import IVON as IBLR
from lib.models import get_model
from lib.datasets import get_dataset
from lib.utils import get_quick_loader, predict_train2, predict_test, flatten, clamp
from lib.variances import get_pred_vars_optim, get_covariance_from_iblr

def get_args():
    parser = argparse.ArgumentParser(description='Evolution of sensitivities during training.')

    # Experiment
    parser.add_argument('--name_exp', default='fmnist_lenet', type=str, help='name of experiment')

    # Data, Model
    parser.add_argument('--dataset', default='FMNIST', choices=['MNIST', 'FMNIST', 'CIFAR10'])
    parser.add_argument('--model', default='lenet', choices=['large_mlp', 'lenet', 'resnet20_frn'])

    # Optimization with IBLR
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--lrmin', default=0, type=float, help='min learning rate of scheduler')
    parser.add_argument('--bs', default=256, type=int, help='batch size')
    parser.add_argument('--epochs', default=100, type=int, help='number of epochs')
    parser.add_argument('--delta', default=60, type=float, help='L2-regularization parameter')

    parser.add_argument('--hess_init', default=0.1, type=float, help='Hessian initialization')

    # Variance computation
    parser.add_argument('--bs_jacs', default=500, type=int, help='Jacobian batch size for variance computation')

    # Defaults
    parser.set_defaults()
    return parser.parse_args()

def get_prediction_vars(optim):
    sigma_sqrs = get_covariance_from_iblr(optim)
    sigma_sqrs = torch.asarray(flatten(sigma_sqrs)).to(device)
    vars = get_pred_vars_optim(net, trainloader_vars, sigma_sqrs, device)
    return vars, optim

def get_sampled_residuals(vars, logits, n_samples, all_targets, eps=1e-10):
    with torch.no_grad():
        randn = torch.randn((n_train, n_samples, nc))
        logits_noise = randn * torch.sqrt(torch.asarray(vars)).reshape(n_train, 1, nc)
        logits_samples = logits_noise + torch.asarray(logits).reshape(n_train, 1, nc)
        probs_mean = torch.mean(F.softmax(logits_samples, dim=-1), dim=1)
        residuals = (probs_mean - F.one_hot(all_targets, nc))
        residuals = clamp(eps, residuals)
    return residuals

if __name__ == "__main__":
    args = get_args()
    print(args)

    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)

    n_samples = 150

    if args.dataset == 'MNIST':
        evaluate_iterations = [10, 1000, 5000, 12250]
    elif args.dataset == 'FMNIST':
        evaluate_iterations = [10, 1000, 5000, 12250]
    elif args.dataset == 'CIFAR10':
        evaluate_iterations = [10, 2000, 10000, 24500]
    else:
        raise NotImplementedError
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

    # Initialize network
    net = get_model(args.model, nc, input_size, device, seed)

    # Optimizer
    optim = IBLR(net.parameters(), lr=args.lr, mc_samples=1, ess=n_train, weight_decay=args.delta / n_train,
                 beta1=0.9, beta2=0.99999, hess_init=args.hess_init)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs, eta_min=args.lrmin)

    # Training and computing sensitivities in intervals
    residuals_list, vars_list, lambdas_list = [], [], []
    iterations_list, losses = [], []
    iteration = 0
    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}\n-------------------------------")

        net.train()
        running_loss = 0
        for X, y in trainloader:
            iteration += 1

            with optim.sampled_params(train=True):
                optim.zero_grad()
                fs = net(X)
                loss = criterion(fs, y)
                loss.backward()
            optim.step()
            running_loss += loss.item()

            if iteration in evaluate_iterations:
                iterations_list.append(iteration + 1)
                print("Computing sensitivities. Iteration: ", iteration)

                # Evaluate on training data
                _, _, lambdas, logits, train_acc, train_nll = predict_train2(net, trainloader_eval, nc, tr_targets, device, True)
                lambdas_list.append(lambdas)
                print(f"Train Acc: {(100 * train_acc):>0.2f}%, Train NLL: {train_nll:>6f}")

                # Prediction variances
                vars, optim = get_prediction_vars(optim)
                vars_list.append(vars)

                residuals = get_sampled_residuals(vars, logits, n_samples, tr_targets)
                residuals_list.append(residuals)

        losses.append(running_loss)
        scheduler.step()

    residuals_list, vars_list, lambdas_list = np.asarray(residuals_list), np.asarray(vars_list), np.asarray(lambdas_list)

    # Evaluate on test data
    test_acc, test_nll = predict_test(net, testloader_eval, nc, te_targets, device)
    print(f"Test Acc: {(100 * test_acc):>0.2f}%, Test NLL: {test_nll:>6f}")

    # Compute sensitivities
    sensitivity = residuals_list * lambdas_list * vars_list
    sensitivity = np.sum(np.abs(sensitivity), axis=-1)

    results_dict = {
                    'sensitivity': sensitivity,
                    'iterations_list': np.asarray(iterations_list),
                    'test_acc': test_acc,
                    'test_nll': test_nll,
                    'args': args,
                     }
    dir = 'pickles/'
    os.makedirs(os.path.dirname(dir), exist_ok=True)
    with open(dir + args.name_exp + '_evolving.pkl', 'wb') as f:
        pickle.dump(results_dict, f, pickle.HIGHEST_PROTOCOL)