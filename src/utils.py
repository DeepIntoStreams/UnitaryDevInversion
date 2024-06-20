import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from functools import partial
from fbm import FBM
from typing import Callable, Tuple

from src.development_layer import development_layer


def fBM_data(num_samples: int, dim: int, length: int,
             h: float) -> torch.Tensor:
    """
    Create fBM paths of shape (num_samples, length + 1, dim)
    """
    fbm_paths = []
    for _ in range(num_samples * dim):
        f = FBM(n=length, hurst=h, method="daviesharte")
        fbm_paths.append(f.fbm())
    data = torch.FloatTensor(np.array(fbm_paths)).reshape(
        num_samples, dim, length + 1).permute(0, 2, 1)
    return data


def add_time(x: torch.Tensor) -> torch.Tensor:
    """
    Prepend an extra time dimension.
    """
    num_samples = x.shape[0]
    length = x.shape[1]
    t = torch.linspace(1 / length, 1,
                       length).reshape(1, -1, 1).repeat(num_samples, 1,
                                                        1).to(x.device)
    return torch.cat([t, x], dim=-1)


class char_func_path(nn.Module):
    def __init__(
        self,
        num_samples,
        hidden_size,
        lie_group,
        input_size,
        if_add_time: bool,
        init_range: float = 1,
    ):
        """
        Class for computing path charateristic function.

        Args:
            num_samples (int): Number of samples.
            hidden_size (int): Hidden size.
            input_size (int): Input size.
            if_add_time (bool): Whether to add time dimension to the input.
            init_range (float, optional): Range for weight initialization. Defaults to 1.
        """
        super(char_func_path, self).__init__()
        self.num_samples = num_samples
        self.hidden_size = hidden_size
        self.lie_group = lie_group
        self.input_size = input_size
        if if_add_time:
            self.input_size = input_size + 1
        else:
            self.input_size = input_size + 0
        self.development = development_layer(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            lie_group=self.lie_group,
            channels=self.num_samples,
            include_initial=True,
            # return_sequence=False,
            init_range=init_range,
        )
        for param in self.development.parameters():
            param.requires_grad = True
        self.if_add_time = if_add_time

    def reset_parameters(self):
        pass

    @staticmethod
    def HS_norm(X: torch.tensor, Y: torch.Tensor):
        """
        Hilbert-Schmidt norm computation.

        Args:
            X (torch.Tensor): Complex-valued tensor of shape (C, m, m).
            Y (torch.Tensor): Tensor of the same shape as X.

        Returns:
            torch.float: Hilbert-Schmidt norm of X and Y.
        """
        if len(X.shape) == 4:
            m = X.shape[-1]
            X = X.reshape(-1, m, m)

        else:
            pass
        D = torch.bmm(X, torch.conj(Y).permute(0, 2, 1))
        return (torch.einsum("bii->b", D)).mean().real

    def distance_measure(self,
                         X1: torch.tensor,
                         X2: torch.tensor,
                         Lambda=0.1) -> torch.float:
        """
        Distance measure given by the Hilbert-Schmidt inner product.

        Args:
            X1 (torch.tensor): Time series samples with shape (N_1, T, d).
            X2 (torch.tensor): Time series samples with shape (N_2, T, d).
            Lambda (float, optional): Scaling factor for additional distance measure on the initial time point,
            this is found helpful for learning distribution of initial time point.
              Defaults to 0.1.

        Returns:
            torch.float: Distance measure between two batches of samples.
        """
        # print(X1.shape)
        if self.if_add_time:
            X1 = add_time(X1)
            X2 = add_time(X2)
        else:
            pass
        # print(X1.shape)
        dev1, dev2 = self.development(X1), self.development(X2)
        N, T, d = X1.shape

        # initial_dev = self.unitary_development_initial()
        CF1, CF2 = dev1.mean(0), dev2.mean(0)

        if Lambda != 0:
            initial_incre_X1 = torch.cat([
                torch.zeros((N, 1, d)).to(X1.device), X1[:, 0, :].unsqueeze(1)
            ],
                                         dim=1)
            initial_incre_X2 = torch.cat([
                torch.zeros((N, 1, d)).to(X1.device), X2[:, 0, :].unsqueeze(1)
            ],
                                         dim=1)
            initial_CF_1 = self.development(initial_incre_X1).mean(0)
            initial_CF_2 = self.development(initial_incre_X2).mean(0)
            return self.HS_norm(CF1 - CF2, CF1 - CF2) + Lambda * self.HS_norm(
                initial_CF_1 - initial_CF_2, initial_CF_1 - initial_CF_2)
        else:
            return self.HS_norm(CF1 - CF2, CF1 - CF2)


class Permutation_test:
    """
    Class for training the empirical distances (i.e., RPCFD, PCFD, OPCFD) 
    and conducting the permutation test on the optimized distances.
    """
    def __init__(self, train_X: torch.Tensor, train_Y: torch.Tensor,
                 test_X: torch.Tensor, test_Y: torch.Tensor, num_exp: int,
                 num_perm: int, sample_size: int, device: str) -> None:
        """
        Attributes
        ----------
        train_X : torch.Tensor
            First half of the training set.
        train_Y : torch.Tensor
            Second half of the training set.
        test_X : torch.Tensor
            First half of the test set.
        test_Y : torch.Tensor
            Second half of the test set.
        num_exp : int
            Number of repeated experiments to estimate the test power or type I error.
        num_perm : int
            Number of permutations sampled to approximate the distribution of T(P_X, P_Y), with T being a statistic.
        sample_size : int
            Number of samples used to calculate the expectations.
        device : str
            Location of the data.
        """
        self.train_X = train_X
        self.train_Y = train_Y
        self.test_X = test_X
        self.test_Y = test_Y
        self.num_exp = num_exp
        self.num_perm = num_perm
        self.sample_size = sample_size
        self.device = device

    def permutation_test(
        self, test_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ) -> Tuple[float, float]:
        """
        Conduct the permutation test on the statistic test_func.
        """
        with torch.no_grad():
            true_rej = 0
            false_rej = 0
            for _ in tqdm(range(self.num_exp)):
                sample_idx = torch.randperm(self.test_X.shape[0])
                X = self.test_X[sample_idx[:self.sample_size]].to(self.device)
                Y1 = self.test_Y[sample_idx[:self.sample_size]].to(self.device)
                Y2 = self.test_Y[sample_idx[self.sample_size:2 *
                                            self.sample_size]].to(self.device)

                H0_stats = []
                H1_stats = []
                for _ in range(self.num_perm):
                    idx = torch.randperm(2 * self.sample_size)
                    combined = torch.cat([Y1, Y2])
                    H0_stats.append(
                        test_func(combined[idx[:self.sample_size]],
                                  combined[idx[self.sample_size:]]).cpu().
                        detach().numpy().item())
                    combined = torch.cat([X, Y1])
                    H1_stats.append(
                        test_func(combined[idx[:self.sample_size]],
                                  combined[idx[self.sample_size:]]).cpu().
                        detach().numpy().item())

                if test_func(Y1, Y2) > np.quantile(H0_stats, 0.95):
                    false_rej += 1
                if test_func(X, Y1) > np.quantile(H1_stats, 0.95):
                    true_rej += 1
            power = true_rej / self.num_exp
            type1_error = false_rej / self.num_exp
        return power, type1_error

    def run(self, distance: str, num_samples: int, hidden_size: int,
            batch_size: int, max_iter: int) -> pd.DataFrame:
        """
        Optimize the distance, and then conduct the permutation test.

        Args
        ----
        distance : str
            The distance used, must be "RPCFD", "PCFD", or "OPCFD".
        num_samples : int
            Number of linear maps to approximate the empirical distance, i.e., K
        hidden_size : int
            Matrix size of the image of the linear maps, i.e., k
        batch_size : int
            Batch size when optimizing the empirical distance with SGD.
        max_iter : int
            Maximum number of iterations when optimizing the empirical distance with SGD.

        Return
        ------
        A dataframe that contains optimization and test results.
        """
        assert distance in [
            "RPCFD", "PCFD", "OPCFD"
        ], "distance shuold be 'RPCFD', 'PCFD', or 'OPCFD'."
        power_trace = []
        type1_error_trace = []
        time_trace = []

        train_X_dl = DataLoader(self.train_X.to(self.device),
                                batch_size,
                                shuffle=True)
        train_Y_dl = DataLoader(self.train_Y.to(self.device),
                                batch_size,
                                shuffle=True)
        distance_func = char_func_path(num_samples=num_samples,
                                       hidden_size=hidden_size,
                                       lie_group=distance,
                                       input_size=self.train_X.shape[-1],
                                       if_add_time=True,
                                       init_range=1).to(self.device)
        char_optimizer = torch.optim.Adam(
            distance_func.parameters(),
            betas=(0, 0.9),
            lr=0.5 if distance == "RPCFD" else 0.05)

        start = time.time()
        for _ in tqdm(range(max_iter)):
            X = next(iter(train_X_dl))
            Y = next(iter(train_Y_dl))
            char_optimizer.zero_grad()
            char_loss = -distance_func.distance_measure(X, Y, Lambda=0)
            char_loss.backward()
            char_optimizer.step()
        end = time.time()

        power, type1_error = self.permutation_test(
            partial(distance_func.distance_measure, Lambda=0))

        power_trace.append(power)
        type1_error_trace.append(type1_error)
        time_trace.append(end - start)

        return pd.DataFrame({
            "power": power_trace,
            "type1_error": type1_error_trace,
            "opt_time": time_trace,
        })
