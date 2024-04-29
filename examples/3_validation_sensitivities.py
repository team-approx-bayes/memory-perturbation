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
from lib.utils import get_quick_loader, predict_train2, predict_test, train_network
from lib.variances import get_pred_vars_laplace

def get_args():
    parser = argparse.ArgumentParser(description='Compare sensitivities to true deviations in softmax outputs from retraining.')

    # Experiment
    parser.add_argument('--name_exp', default='mnist_mlp', type=str, help='name of experiment')

    # Data, Model
    parser.add_argument('--dataset', default='MNIST', choices=['MNIST', 'FMNIST', 'CIFAR10'])
    parser.add_argument('--model', default='large_mlp',choices=['large_mlp', 'lenet', 'cnn_deepobs'])

    # Optimization with SGD
    parser.add_argument('--lr', default=1e-2, type=float, help='learning rate')
    parser.add_argument('--lrmin', default=1e-3, type=float, help='min learning rate of scheduler')
    parser.add_argument('--bs', default=256, type=int, help='batch size')
    parser.add_argument('--epochs', default=500, type=int, help='number of epochs')
    parser.add_argument('--delta', default=100, type=float, help='L2-regularization parameter')

    # Retraining
    parser.add_argument('--lr_retrain', default=1e-3, type=float, help='retraining: learning rate')
    parser.add_argument('--lrmin_retrain', default=1e-4, type=float, help='retraining: min learning rate scheduler')
    parser.add_argument('--epochs_retrain', default=300, type=int, help='retraining: number of epochs')
    parser.add_argument('--n_retrain', default=1000, type=int, help='number of retrained examples')

    # Variance computation
    parser.add_argument('--bs_jacs', default=50, type=int, help='Jacobian batch size for variance computation')
    parser.set_defaults()
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    print(args)

    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device', device)

    # Data
    ds_train, ds_test, transform_train = get_dataset(args.dataset, return_transform=True)
    input_size = len(ds_train.data[0, :])**2
    nc = max(ds_train.targets) + 1
    n_train = len(ds_train)
    tr_targets, te_targets = torch.asarray(ds_train.targets), torch.asarray(ds_test.targets)

    # Model
    net = get_model(args.model, nc, input_size, device, seed)

    # Dataloaders
    trainloader = get_quick_loader(DataLoader(ds_train, batch_size=args.bs)) # training
    trainloader_eval = DataLoader(ds_train, batch_size=args.bs, shuffle=False) # train evaluation
    testloader_eval = DataLoader(ds_test, batch_size=args.bs, shuffle=False) # test evaluation
    trainloader_vars = DataLoader(ds_train, batch_size=args.bs_jacs, shuffle=False) # variance computation

    # Train base network and store parameters
    net, losses = train_network(net, trainloader, args.lr, args.lrmin, args.epochs, n_train, args.delta)
    w_star = parameters_to_vector(net.parameters()).detach().cpu().clone()

    # Evaluate on training data; residuals and lambdas
    residuals, probs, lambdas, train_acc, train_nll = predict_train2(net, trainloader_eval, nc, tr_targets, device)
    print(f"Train Acc: {(100 * train_acc):>0.2f}%, Train NLL: {train_nll:>6f}")

    # Evaluate on test data
    test_acc, test_nll = predict_test(net, testloader_eval, nc, te_targets, device)
    print(f"Test Acc: {(100 * test_acc):>0.2f}%, Test NLL: {test_nll:>6f}")

    # Compute prediction variances
    vars = get_pred_vars_laplace(net, trainloader_vars, args.delta, nc, version='kfac', device=device)

    # Compute and store sensitivities
    sensitivities = np.asarray(residuals) * np.asarray(lambdas) * np.asarray(vars)
    sensitivities = np.sum(np.abs(sensitivities), axis=-1)

    scores_dict = {'sensitivities': sensitivities}
    dir = 'pickles/'
    os.makedirs(os.path.dirname(dir), exist_ok=True)
    with open('pickles/' + args.name_exp + '_scores.pkl', 'wb') as f:
        pickle.dump(scores_dict, f, pickle.HIGHEST_PROTOCOL)

    # Random subsampling of examples for retraining
    indices = np.arange(0, n_train, 1)
    np.random.shuffle(indices)
    indices_retrain = indices[0:args.n_retrain]

    # Retrain with one example removed
    softmax_deviations = np.zeros((args.n_retrain, nc))
    for i in range(args.n_retrain):
        print('\nRemoved example ', i)

        # Warmstarting
        vector_to_parameters(w_star, net.parameters())
        net = net.to(device)

        # Remove one example from training set
        idx_removed = indices_retrain[i]
        idx_remain = np.setdiff1d(np.arange(0, n_train, 1), idx_removed)
        ds_train_perturbed = Subset(ds_train, idx_remain)
        trainloader_retrain = get_quick_loader(DataLoader(ds_train_perturbed, batch_size=args.bs, shuffle=True))

        # Retraining
        net, losses = train_network(net, trainloader_retrain, args.lr_retrain, args.lrmin_retrain, args.epochs_retrain, n_train-1, args.delta)

        # Evaluate softmax deviations
        net.eval()
        with torch.no_grad():
            X_removed = transform_train(torch.asarray(ds_train.data[idx_removed]).numpy()).cuda()
            logits_wminus = net(X_removed.expand((1, -1, -1, -1)).to(device))
            probs_wminus = torch.softmax(logits_wminus, dim=-1).cpu().numpy()
            softmax_deviations[i] = probs_wminus - probs[idx_removed]

    # L1-norm
    softmax_deviations = np.sum(np.abs(softmax_deviations), axis=-1)

    # Save softmax deviations by removing an example and retraining (baseline)
    retrain_dict = {'indices_retrain': indices_retrain,
                    'softmax_deviations': softmax_deviations,
                     }
    with open('pickles/' + args.name_exp + '_retrain.pkl', 'wb') as f:
        pickle.dump(retrain_dict, f, pickle.HIGHEST_PROTOCOL)

