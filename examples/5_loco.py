import os
import sys
import pickle
import argparse
import numpy as np

import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import DataLoader, Subset

sys.path.append("..")
from lib.models import get_model
from lib.datasets import get_dataset
from lib.utils import get_quick_loader, predict_train, predict_test, get_estimated_nll, train_network
from lib.variances import get_pred_vars_laplace

def get_args():
    parser = argparse.ArgumentParser(description='Predicting generalization for leave-one-class-out with MPE.')

    # Experiment
    parser.add_argument('--name_exp', default='fmnist_lenet', type=str, help='name of experiment')

    # Data, Model
    parser.add_argument('--dataset', default='FMNIST', choices=['MNIST', 'FMNIST'])
    parser.add_argument('--model', default='lenet', choices=['small_mlp', 'large_mlp', 'lenet'])

    # Optimization with SGD
    parser.add_argument('--lr', default=1e-1, type=float, help='learning rate')
    parser.add_argument('--lrmin', default=1e-3, type=float, help='min learning rate of scheduler')
    parser.add_argument('--bs', default=256, type=int, help='batch size')
    parser.add_argument('--epochs', default=300, type=int, help='number of epochs')
    parser.add_argument('--delta', default=100, type=float, help='L2-regularization parameter')

    # Retraining
    parser.add_argument('--lr_retrain', default=1e-4, type=float, help='retraining: learning rate')
    parser.add_argument('--lrmin_retrain', default=1e-5, type=float, help='retraining: min learning rate scheduler')
    parser.add_argument('--epochs_retrain', default=1000, type=int, help='retraining: number of epochs')

    # Variance computation
    parser.add_argument('--bs_jacs', default=500, type=int, help='Jacobian batch size for variance computation')
    parser.set_defaults()
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    print(args)

    # Set seed
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device', device)

    # Data
    ds_train, ds_test = get_dataset(args.dataset)
    input_size = len(ds_train.data[0, :]) ** 2
    nc = max(ds_train.targets) + 1
    n_train = len(ds_train)
    tr_targets, te_targets = torch.asarray(ds_train.targets), torch.asarray(ds_test.targets)

    # Model
    net = get_model(args.model, nc, input_size, device, seed)

    # Dataloaders
    trainloader = get_quick_loader(DataLoader(ds_train, batch_size=args.bs))  # training
    trainloader_eval = DataLoader(ds_train, batch_size=args.bs, shuffle=False)  # train evaluation
    testloader_eval = DataLoader(ds_test, batch_size=args.bs, shuffle=False)  # test evaluation
    trainloader_vars = DataLoader(ds_train, batch_size=args.bs_jacs, shuffle=False)  # variance computation

    # Train base network and store parameters
    net, losses = train_network(net, trainloader, args.lr, args.lrmin, args.epochs, n_train, args.delta)
    w_star = parameters_to_vector(net.parameters()).detach().cpu().clone()

    # Evaluate on training data; residuals and logits
    residuals, tr_logits, train_acc, train_nll = predict_train(net, trainloader_eval, nc, tr_targets, device, return_logits=True)
    print(f"Train Acc: {(100 * train_acc):>0.2f}%, Train NLL: {train_nll:>6f}")

    # Evaluate on test data
    test_acc, test_nll = predict_test(net, testloader_eval, nc, te_targets, device)
    print(f"Test Acc: {(100 * test_acc):>0.2f}%, Test NLL: {test_nll:>6f}")

    # Compute prediction variances
    vars = get_pred_vars_laplace(net, trainloader_vars, args.delta, nc, version='kfac', device=device)

    residuals, vars, tr_logits = np.asarray(residuals), np.asarray(vars), np.asarray(tr_logits)

    # Retrain with one class removed
    est_perturb_NLL, perturb_ACC, perturb_NLL = np.zeros(nc), np.zeros(nc), np.zeros(nc)
    for remove_class in range(nc):
        removed_ix = np.array(np.where(np.equal(tr_targets.numpy(), remove_class))).squeeze()
        print('remove class {}, {} examples'.format(remove_class, len(removed_ix)))
        remain_index = np.setdiff1d(np.arange(0, n_train, 1), removed_ix)
        ds_train_perturbed = Subset(ds_train, remain_index)

        trainloader_retrain = get_quick_loader(DataLoader(ds_train_perturbed, batch_size=args.bs, shuffle=True))
        n_retrain = len(remain_index)

        # Estimate NLL
        est_nll = get_estimated_nll(nc, residuals[removed_ix], vars[removed_ix], tr_logits[removed_ix], tr_targets[removed_ix], loco=True)

        # Warmstarting
        vector_to_parameters(w_star, net.parameters())
        net = net.to(device)

        # Retraining
        net, _ = train_network(net, trainloader_retrain, args.lr_retrain, args.lrmin_retrain, args.epochs_retrain, n_retrain, args.delta)

        # Evaluate: get true NLL and ACC
        test_acc_wc, test_nll_wc = predict_test(net, testloader_eval, nc, te_targets, device)

        est_perturb_NLL[remove_class] = est_nll
        perturb_ACC[remove_class] = test_acc_wc
        perturb_NLL[remove_class] = test_nll_wc

    results_dict = {'org_acc': test_acc,
                    'est_nll': est_perturb_NLL,
                    'perturb_nll': perturb_NLL,
                    'perturb_acc': perturb_ACC}
    dir = 'pickles/'
    os.makedirs(os.path.dirname(dir), exist_ok=True)
    with open(dir + args.name_exp + '_leaveclassout.pkl', 'wb') as f:
        pickle.dump(results_dict, f, pickle.HIGHEST_PROTOCOL)















