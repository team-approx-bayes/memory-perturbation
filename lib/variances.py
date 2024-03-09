import copy
import numpy as np
from laplace import Laplace
from laplace.curvature import AsdlGGN
import torch
from torch.utils.data import DataLoader
from torch.func import jacrev, vmap

def get_pred_vars_laplace(net, trainloader, delta, nc, version='kfac', device='cuda'):
    if version == 'kfac':
        hessian_structure = 'kron'
    elif version == 'diag':
        hessian_structure = 'diag'
    laplace_object = Laplace(
        net, 'classification',
        subset_of_weights='all',
        hessian_structure=hessian_structure,
        prior_precision=delta,
        backend=AsdlGGN)

    torch.cuda.empty_cache()
    laplace_object.fit(trainloader)

    fvars = np.empty(shape=(0, nc))
    for inputs, _ in trainloader:
        inputs = inputs.to(device)
        _, fvar = laplace_object._glm_predictive_distribution(inputs)
        fvars = np.vstack((fvars, np.diagonal(fvar.cpu().numpy(), axis1=1, axis2=2)))

    del laplace_object
    torch.cuda.empty_cache()
    return fvars.tolist()

def make_functional(mod, disable_autograd_tracking=False):
    params_dict = dict(mod.named_parameters())
    params_names = params_dict.keys()
    params_values = tuple(params_dict.values())

    stateless_mod = copy.deepcopy(mod)
    stateless_mod.to('meta')

    def fmodel(new_params_values, *args, **kwargs):
        new_params_dict = {name: value for name, value in zip(params_names, new_params_values)}
        return torch.func.functional_call(stateless_mod, new_params_dict, args, kwargs)

    if disable_autograd_tracking:
        params_values = torch.utils._pytree.tree_map(torch.Tensor.detach, params_values)
    return fmodel, params_values

def get_pred_vars_optim(model, loader, sigma_diag, device='cuda'):
    fvars = []
    fnet, params = make_functional(model, disable_autograd_tracking=True)

    def fnet_single(params, x):
        f = fnet(params, x.unsqueeze(0)).squeeze(0)
        return (f, f)

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        Js, f = vmap(jacrev(fnet_single, has_aux=True), (None, 0))(params, X)
        Js = torch.cat([j.flatten(2) for j in Js], dim=2)
        fvar = torch.einsum('nkp,p,ncp->nkc', Js, sigma_diag, Js)
        fvar = fvar.diagonal(dim1=1, dim2=2)
        fvars.append(fvar)

    fvars = torch.squeeze(torch.cat(fvars)).cpu()
    return fvars.tolist()

def get_covariance_from_iblr(optim):
    sigma_sqrs = []
    offset = 0
    for group in optim.param_groups:
        std = 1 / (group["ess"] * (group["hess"] + group["weight_decay"])).sqrt()
        goffset = 0
        for p in group['params']:
            if p.requires_grad:
                if p is None:
                    continue
                numel = p.numel()
                std_ = std[offset: offset + numel]
                sigma_sqrs.append((std_**2).tolist())
                goffset += numel
                offset += numel
        assert goffset == group["numel"]
    return sigma_sqrs

def get_covariance_from_adam(optim, delta, n_train):
    with torch.no_grad():
        sigma_sqrs = []
        for group in optim.param_groups:
            for p in group['params']:
                state = optim.state[p]
                sigma_sqr = 1 / (torch.flatten(torch.sqrt(state['exp_avg_sq'])) * n_train + delta)
                sigma_sqrs.append(sigma_sqr.tolist())
    return sigma_sqrs

