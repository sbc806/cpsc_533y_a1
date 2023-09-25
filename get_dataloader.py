import functools
from os import replace

import numpy as np
import torch
from torch.utils import data
from torch.utils.data import DataLoader, Subset

from utils import mnist_helper


def get_dataloader(config, mode):
    """Set up input pipeline and get dataloader."""
    assert mode in ["train" , "test", "single_batch"]
    # this is hacky, but we sample the single batch samples from training set
    if mode ==  "single_batch":
        dataset = MnistptsDataset(config, "train")
    else:
        dataset = MnistptsDataset(config, mode)
    loader = functools.partial(
        DataLoader,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    loader_list = []

    # TODO: Use PyTorch Subset module to divide the training dataset into train
    # and validation. To do this, first find out how many items should go into
    # training and how many into validation using the configuration. Then,
    # shuffle the dataset randomly, and create two `Subset`s. Note that the
    # training dataset should be shuffled randomly everytime you go through the
    # dataset, while the validation set does not need to be.
    # TODO: We also want to create a dataset that contains only a single mini
    # batch of samples that will be used to perform sanity check on the
    # training code.    
    if mode == "train":
        ratio_tr_data = config.ratio_tr_data
        num_all = len(dataset)
        num_tr = int(ratio_tr_data * num_all)

        print(f"Number of training samples: {num_tr}")
        print(f"Number of valid samples: {num_all - num_tr}")
        
        # Create the two subsets after shuffling
        weights = torch.ones(num_all)
        random_ordering = torch.multinomial(weights, num_all, replacement=False)
        train_subset = Subset(dataset, random_ordering[0:num_tr])
        val_subset = Subset(dataset, random_ordering[num_tr:])
        # Create the data loaders
        train_loader = loader(dataset=train_subset, shuffle=True)
        val_loader = loader(dataset=val_subset, shuffle=False)
        loader_list.append(train_loader)
        loader_list.append(val_loader)
    elif mode == "test":
        num_all = len(dataset)

        print(f"Number of test samples: {num_all}")
        
        test_loader = loader(dataset=dataset, shuffle=False)
        loader_list.append(test_loader)
    elif mode == 'single_batch':
        num_all = config.batch_size

        print(f"Number of single batch samples: {num_all}")

        # Get a random single batch from the dataset
        weights = torch.ones(len(dataset))
        random_indices = torch.multinomial(weights, num_all, replacement=False)
        batch_subset = Subset(dataset, random_indices)
        batch_loader = loader(dataset=batch_subset, shuffle=False)
        loader_list.append(batch_loader)
    else:
        raise NotImplementedError
    
    return loader_list


class MnistptsDataset(data.Dataset):
    """Dataset for Mnist point clouds."""

    def __init__(self, config, mode):
        """Define immutable variables for multi-threaded loading.

        Args:
            config (config_dict):  hyperparamter configuration.
            mode: type of datset split.
        """

        assert mode in ["train", "test"]

        self.mode = mode
        self.config = config
        self.num_pts = config.num_pts
        self.random_sample = config.random_sample
        print(f"loading {mode} datasets.")
        
        # TODO: Our dataset is small. Load the entire dataset into memory to
        # avoid excessive disk access!
        self.pts_list, self.labels = mnist_helper.load_mnistpts(config.data_mnistpts_dir, mode)
        self.pts_list = torch.tensor(self.pts_list).astype(torch.float32)
        self.labels = torch.tensor(self.labels).astype(torch.int64)

    def __len__(self):
        """Return the length of dataset."""
        # TODO: return the length of dataset.
        return len(self.labels)

    def random_sampling(self, pts, num_pts):
        """Sampling points from point cloud.

        Args:
            pts (array): Nx2, point cloud.
        Returns:
            pts_sampled (array):  num_ptsX2, sampled point cloud.
        """

        # TODO: Sample points from point cloud. Importantly, we will sample
        # **without** replacement here to simulate how many actual point cloud
        # data behaves. If random sample is False, we simply sample the first
        # K points (K=num_pts)
        # Note: Random state might improperly be shared among threads. This is
        #   especially true if you use numpy to sample. Use PyTorch!
        if self.random_sample:
            # pass
            weights = torch.ones(pts.shape[0])
            sample_indices = torch.multinomial(weights, num_pts)
            pts_sampled = pts[sample_indices]
        else:
            # pass
            pts_sampled = pts[0: num_pts]
        
        return pts_sampled

    def __getitem__(self, index):
        """Get item"""

        # TODO: get item from dataset.
        # Note that we expect: pc (np.float32 type), label(np.int)
        individual_pc = self.pts_list[index]
        individual_pc_label = self.labels[index]
        
        data = {"pc": self.random_sampling(individual_pc, self.num_pts),
                "label": individual_pc_label}
        return data
    
    def normalize_data(self):
        if self.mode == "train":
            mean = torch.mean(self.pts_list, dim=0)
            var = torch.var(self.pts_list, dim=0)
        else:
            train_pts_list = mnist_helper.load_mnistpts(self.config.data_mnistpts_dir, "train")
            mean = torch.mean(train_pts_list, dim=0)
            var = torch.var(train_pts_list, dim=0)
        
        self.pts_list = self.pts_list - mean
        self.pts_list = self.pts_list / var
        