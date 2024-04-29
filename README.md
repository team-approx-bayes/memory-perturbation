# The Memory-Perturbation Equation 

This repository contains code to reproduce experiments of the NeurIPS 2023 paper:

**The Memory Perturbation Equation: Understanding Model's Sensitivity to Data**  
*P. Nickl, L. Xu\*, D. Tailor\*, T. Möllenhoff, M.E. Khan*\
Paper: https://arxiv.org/abs/2310.19273

![mpe fig1](https://github.com/pnickl/mpe-code-release/blob/main/fig1.png)

## Installation

- To create a conda environment `mpe` with all necessary dependencies run: `conda env create --file environment.yml`
- We use torch 2.2.1 with cuda 12.1.1

**Additional information on dependencies**

- Visit [this GitHub repository](https://github.com/team-approx-bayes/ivon) for a usage guide of the iBLR optimizer.
- For Laplace variance approximations we use the [laplace-torch](https://aleximmer.github.io/Laplace/) package. 

## General information on the experiments

- Results of example scripts are saved to `examples/pickles/`
- The figures produced by plotting scripts are saved to `examples/plots/`
- Argument `--name_exp`: This argument defines the name of the pickle file for storing the results of the example scripts. The same argument should be also used for the respective plotting script.
- Argument `--var_version`: For experiments that use SGD or Adam and sensitivity computation with Laplace approximations, this arguments specifies a diagonal GGN (`--var_version diag`) or KFAC (`--var_version kfac`) approximation.
- Argument `--bs_jacs`: The experiments have been tested on a Tesla V100 GPU with 16GB memory. To adjust for varying GPU memory size, the batch size for Jacobian computations can be modified with `--bs_jacs`. 

## Predicting generalization during training 

- `1_loo_iterations.py`: This script reproduces Figs. 1(b), 5, 8, 9 and 10.
- Flag `--optim_var`: This flag needs to be passed for experiments that compute sensitivities on-the-fly from the optimizer state of iBLR or Adam. For SGD or Adam with sensitivies from an additional Laplace variance computation the flag should not be set.
- Plot with `python plot_loo_iterations.py --name_exp XXX`

In the following we provide exemplary commands for reproducing experiments. 

**Fig. 1(b):** CIFAR10, ResNet20 with sensitivities from iBLR
```
python 1_loo_iterations.py --name_exp cifar10_resnet20_iblr --dataset CIFAR10 --model resnet20_frn --optimizer iblr --hess_init 0.01 --epochs 200 --lr 0.2 --lrmin 0 --bs 50 --delta 10 --bs_jacs 300 --optim_var
```

**Fig. 5:** FMNIST, LeNet5 with sensitivities from iBLR and from SGD with diagonal GGN / KFAC
```
python 1_loo_iterations.py --name_exp fmnist_lenet_iblr --dataset FMNIST --model lenet --optimizer iblr --hess_init 0.1 --epochs 100 --lr 0.1 --lrmin 0 --bs 256 --delta 60 --bs_jacs 500 --optim_var
```
```
python 1_loo_iterations.py --name_exp fmnist_lenet_sgd_diag --dataset FMNIST --model lenet --optimizer sgd --epochs 100 --lr 1e-2 --lrmin 1e-4 --bs 256 --delta 60 --bs_jacs 500 --var_version diag
```
```
python 1_loo_iterations.py --name_exp fmnist_lenet_sgd_kfac --dataset FMNIST --model lenet --optimizer sgd --epochs 100 --lr 1e-2 --lrmin 1e-4 --bs 256 --delta 60 --bs_jacs 500 --var_version kfac
```

## Predicting generalization for hyperparameter tuning ##

- `2_loo_l2reg.py`: This script reproduces Fig. 4.
- Plot with `python plot_loo_l2reg.py --name_exp XXX`

**Fig. 4 (a):** MNIST, MLP (500, 300)
```
python 2_loo_l2reg.py --name_exp mnist_mlp --dataset MNIST --model large_mlp --n_exp 100 --deltas_start 0 --deltas_stop 3 --optimizer sgd --epochs 500 --lr 1e-2 --lrmin 1e-3 --bs 256 --bs_jacs 50
```
**Fig. 4 (b):** FMNIST, LeNet5
```
python 2_loo_l2reg.py --name_exp fmnist_lenet --dataset FMNIST --model lenet --n_exp 100 --deltas_start 1 --deltas_stop 3 --optimizer adam --epochs 300 --lr 1e-1 --lrmin 1e-3 --bs 256 --bs_jacs 500
```

## Estimated deviations correlate with the truth ##

- `3_validation_sensitivities.py`: This script reproduces the results of Fig. 2.
- Plot with `python plot_validation_sensitivities.py --name_exp XXX`

**Fig. 2 (a):** MNIST, MLP (500, 300)
```
python 3_validation_sensitivities.py --name_exp mnist_mlp --dataset MNIST --model large_mlp --delta 100 --epochs 500 --lr 1e-2 --lrmin 1e-3 --n_retrain 1000 --epochs_retrain 300 --lr_retrain 1e-3 --lrmin_retrain 1e-4 --bs 256 --bs_jacs 50
```
**Fig. 2 (b):** FMNIST, LeNet5
```
python 3_validation_sensitivities.py --name_exp fmnist_lenet --dataset FMNIST --model lenet --delta 100 --epochs 300 --lr 1e-1 --lrmin 1e-3 --n_retrain 1000 --epochs_retrain 200 --lr_retrain 1e-3 --lrmin_retrain 1e-4 --bs 256 --bs_jacs 500
```
**Fig. 2 (c):** CIFAR10, CNN
```
python 3_validation_sensitivities.py --name_exp cifar10_cnn --dataset CIFAR10 --model cnn_deepobs --delta 250 --epochs 500 --lr 1e-2 --lrmin 1e-4 --n_retrain 100 --epochs_retrain 300 --lr_retrain 1e-4 --lrmin_retrain 1e-5 --bs 256 --bs_jacs 20
```

## Evolution of sensitivities during training ##

- `4_evolving_sensitivities.py`: This script reproduces the results of Figs. 3 (b), 11 (b) and 11 (c).
- Plot with `python plot_evolving_sensitivities.py --name_exp XXX`
- In the paper we have used an earlier version of the iBLR optimizer (for this experiment only). The updated version used in this repository is linked in the section "dependencies" and reproduces the results of the paper qualitatively.

**Fig. 3 (b):** FMNIST, LeNet
```
python 4_evolving_sensitivities.py --name_exp fmnist_lenet --dataset FMNIST --model lenet --hess_init 0.1 --epochs 100 --lr 0.1 --lrmin 0 --bs 256 --delta 60 --bs_jacs 500
```
**Fig. 11 (b):** MNIST, MLP (500, 300)
```
python 4_evolving_sensitivities.py --name_exp mnist_mlp --dataset MNIST --model large_mlp --hess_init 0.1 --epochs 100 --lr 0.1 --lrmin 0 --bs 256 --delta 30 --bs_jacs 50
```
**Fig. 11 (c):** CIFAR10, ResNet20
```
python 4_evolving_sensitivities.py --name_exp cifar10_resnet20 --dataset CIFAR10 --model resnet20_frn --hess_init 0.1 --epochs 300 --lr 0.1 --lrmin 0 --bs 512 --delta 35 --bs_jacs 300
```

## Predicting the effect of class removal on generalization ##
- `5_loco.py`: This script reproduces Fig. 3 (a).
- Plot with `python plot_loco.py --names_exp XXX YYY` 

**Fig. 3 (a):** FMNIST, LeNet & MLP (32, 16)
```
python 5_loco.py --name_exp fmnist_lenet --dataset FMNIST --model lenet --lr 1e-1 --lrmin 1e-3 --bs 256 --epochs 300 --delta 100 --lr_retrain 1e-4 --lrmin_retrain 1e-5 --epochs_retrain 1000 --bs_jacs 500
python 5_loco.py --name_exp fmnist_mlp --dataset FMNIST --model small_mlp --lr 1e-2 --lrmin 1e-3 --bs 256 --epochs 300 --delta 100 --lr_retrain 1e-5 --lrmin_retrain 1e-6 --epochs_retrain 1000 --bs_jacs 500
```

## Troubleshooting

Please open an issue in this repository or contact [Peter](mailto:peter.nickl@riken.jp).

## How to cite

```
@inproceedings{NicklXu2023mpe,
 author = {Nickl, Peter and Xu, Lu and Tailor, Dharmesh and M\"{o}llenhoff, Thomas and Khan, Mohammad Emtiyaz},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {A. Oh and T. Neumann and A. Globerson and K. Saenko and M. Hardt and S. Levine},
 pages = {26923--26949},
 publisher = {Curran Associates, Inc.},
 title = {The Memory-Perturbation Equation: Understanding Model's Sensitivity to Data},
 url = {https://proceedings.neurips.cc/paper_files/paper/2023/file/550ab405d0addd3de5b70e57b44878df-Paper-Conference.pdf},
 volume = {36},
 year = {2023}
}
```
