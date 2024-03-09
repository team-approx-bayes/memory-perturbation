import torchvision
import torchvision.transforms as transforms
import torch

transform_usps = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2469,), (0.2989,)),
])
transform_mnist = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
transform_fmnist = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,)),
])
transform_cifar10 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

def get_dataset(name_dataset, return_transform=False):
    if name_dataset == 'MNIST':
        ds_train, ds_test = load_mnist()
        transform = transform_mnist
    elif name_dataset == 'FMNIST':
        ds_train, ds_test = load_fmnist()
        transform = transform_fmnist
    elif name_dataset == 'CIFAR10':
        ds_train, ds_test = load_cifar10()
        transform = transform_cifar10
    else:
        raise NotImplementedError
    if return_transform:
        return ds_train, ds_test, transform
    else:
        return ds_train, ds_test

def load_cifar10():
    trainset = torchvision.datasets.CIFAR10(
        root='../data', train=True, download=True, transform=transform_cifar10)
    testset = torchvision.datasets.CIFAR10(
        root='../data', train=False, download=True, transform=transform_cifar10)
    testset.targets = torch.asarray(testset.targets)
    return trainset, testset

def load_usps():
    trainset = torchvision.datasets.USPS(
        root='../data', train=True, download=True, transform=transform_usps)
    testset = torchvision.datasets.USPS(
        root='../data', train=False, download=True, transform=transform_usps)
    return trainset, testset

def load_mnist():
    trainset = torchvision.datasets.MNIST(
        root='../data', train=True, download=True, transform=transform_mnist)
    testset = torchvision.datasets.MNIST(
        root='../data', train=False, download=True, transform=transform_mnist)
    return trainset, testset

def load_fmnist():
    trainset = torchvision.datasets.FashionMNIST(
        root='../data', train=True, download=True, transform=transform_fmnist)
    testset = torchvision.datasets.FashionMNIST(
        root='../data', train=False, download=True, transform=transform_fmnist)
    return trainset, testset