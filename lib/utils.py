import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import accuracy_score

def get_estimated_nll(nc, residuals, vars, logits, all_targets, eps=1e-10):
    sensitivities = residuals * vars
    logits_perturbed = sensitivities + logits
    probs_perturbed = torch.softmax(torch.from_numpy(logits_perturbed), dim=-1)

    estimated_nll = []
    for i in range(len(probs_perturbed)):
        nll = - torch.sum((torch.log(probs_perturbed[i].clamp(min=eps)) * F.one_hot(all_targets, nc)),dim=1).mean().numpy()
        estimated_nll.append(nll)
    return estimated_nll

def predict_train(net, loader, nc, all_targets, device='cuda', return_logits=False, eps=1e-10):
    with torch.no_grad():
        net.eval()

        preds_list = []
        res_list, logits_list, probs_list = np.empty(shape=(0, nc)), np.empty(shape=(0, nc)), torch.empty((0, nc))
        for X, y in loader:
            X, y = X.to(device), y.to(device)

            logits = net.forward(X)
            probs = F.softmax(logits, dim=-1)
            _, preds = probs.max(1)

            residuals = (probs - F.one_hot(y, nc))
            residuals = clamp(eps, residuals)

            preds_list.append(preds.cpu().numpy())
            probs_list = torch.vstack((probs_list, probs.cpu()))
            res_list = np.vstack((res_list, residuals))
            if return_logits:
                logits_list = np.vstack((logits_list, logits.cpu().numpy()))

        train_acc = accuracy_score(all_targets, np.concatenate(preds_list))
        train_nll = - torch.sum((torch.log(probs_list.clamp(min=eps)) * F.one_hot(all_targets, nc)),dim=1).mean().numpy()

        if return_logits:
            return res_list.tolist(), logits_list.tolist(), train_acc, train_nll
        else:
            return res_list.tolist(), train_acc, train_nll

def predict_train2(net, loader, nc, all_targets, device='cuda', return_logits=False, eps=1e-10):
    with torch.no_grad():
        net.eval()

        preds_list = []
        res_list, lams_list, logits_list, probs_list = np.empty(shape=(0, nc)), np.empty(shape=(0, nc)), \
            np.empty(shape=(0, nc)), torch.empty((0, nc))
        for X, y in loader:
            X, y = X.to(device), y.to(device)

            logits = net.forward(X)
            probs = F.softmax(logits, dim=-1)
            _, preds = probs.max(1)

            lams = softmax_hessian(probs, eps).cpu().numpy()
            residuals = (probs - F.one_hot(y, nc))
            residuals = clamp(eps, residuals)

            preds_list.append(preds.cpu().numpy())
            probs_list = torch.vstack((probs_list, probs.cpu()))
            lams_list = np.vstack((lams_list, lams))
            res_list = np.vstack((res_list, residuals))
            if return_logits:
                logits_list = np.vstack((logits_list, logits.cpu().numpy()))

        train_acc = accuracy_score(all_targets, np.concatenate(preds_list))
        train_nll = - torch.sum((torch.log(probs_list.clamp(min=eps)) * F.one_hot(all_targets, nc)),dim=1).mean().numpy()

        if return_logits:
            return res_list.tolist(), np.asarray(probs_list.tolist()), lams_list.tolist(), logits_list.tolist(), train_acc, train_nll
        else:
            return res_list.tolist(), np.asarray(probs_list.tolist()), lams_list.tolist(), train_acc, train_nll

def predict_test(net, loader, nc, all_targets, device='cuda', eps=1e-10):
    with torch.no_grad():
        net.eval()

        preds_list, probs_list = [], torch.empty((0, nc))
        for X, y in loader:
            X, y = X.to(device), y.to(device)

            logits = net.forward(X)
            probs = F.softmax(logits, dim=-1)
            _, preds = probs.max(1)

            preds_list.append(preds.cpu().numpy())
            probs_list = torch.vstack((probs_list, probs.cpu()))

        test_acc = accuracy_score(all_targets, np.concatenate(preds_list))
        test_nll = - torch.sum((torch.log(probs_list.clamp(min=eps)) * F.one_hot(all_targets, nc)),dim=1).mean().numpy()
        return test_acc, test_nll
    
def moving_average(y, window):
    average_y = []
    for ind in range(len(y) - window + 1):
        average_y.append(np.mean(y[ind:ind+window]))
    return np.asarray(average_y)

def flatten(matrix):
    flat_list = []
    for row in matrix:
        flat_list.extend(row)
    return flat_list

def softmax_hessian(probs, eps):
    return torch.clamp(probs - probs * probs, min=eps, max=1 - eps)

def clamp(eps, input):
    input[input >= 0] = torch.clamp(input[input >= 0], eps, (1 - eps))
    input[input < 0] = torch.clamp(input[input < 0], -(1 - eps), -eps)
    return input.cpu().numpy()

def get_quick_loader(loader, device='cuda'):
    if device=='cuda':
        return [(X.to(device), y.to(device)) for X, y in loader]
    else:
        return loader

