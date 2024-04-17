import numpy as np
import pandas as pd
import torch
from fbm import FBM
from tqdm import tqdm
from sigkernel.sigkernel import SigKernel


def FBM_data(num_samples, dim, length, h):
    fbm_paths = []
    for i in range(num_samples * dim):
        f = FBM(n=length, hurst=h, method='daviesharte')
        fbm_paths.append(f.fbm())
    data = torch.FloatTensor(np.array(fbm_paths)).reshape(
        num_samples, dim, length + 1).permute(0, 2, 1)
    return data


def get_time_vector(size: int, length: int) -> torch.Tensor:
    return torch.linspace(1 / length, 1, length).reshape(1, -1,
                                                         1).repeat(size, 1, 1)


def AddTime(x):
    t = get_time_vector(x.shape[0], x.shape[1]).to(x.device)
    return torch.cat([t, x], dim=-1)


class LaplaceKernel():
    """Laplace kernel k: R^d x R^d -> R, modified from the sigkernel package"""
    def __init__(self, sigma):
        self.sigma = sigma

    def batch_kernel(self, X, Y):
        """Input:
                  - X: torch tensor of shape (batch, length_X, dim),
                  - Y: torch tensor of shape (batch, length_Y, dim)
           Output:
                  - matrix k(X^i_s,Y^i_t) of shape (batch, length_X, length_Y)
        """
        A = X.shape[0]
        M = X.shape[1]
        N = Y.shape[1]
        Xs = torch.sum(X**2, dim=2)
        Ys = torch.sum(Y**2, dim=2)
        dist = -2. * torch.bmm(X, Y.permute(0, 2, 1))
        dist += torch.reshape(Xs, (A, M, 1)) + torch.reshape(Ys, (A, 1, N))
        dist = torch.clamp(dist, min=0)
        return torch.exp(-torch.sqrt(dist) / self.sigma)

    def Gram_matrix(self, X, Y):
        """Input:
                  - X: torch tensor of shape (batch_X, length_X, dim),
                  - Y: torch tensor of shape (batch_Y, length_Y, dim)
           Output:
                  - matrix k(X^i_s,Y^j_t) of shape (batch_X, batch_Y, length_X, length_Y)
        """
        A = X.shape[0]
        B = Y.shape[0]
        M = X.shape[1]
        N = Y.shape[1]
        Xs = torch.sum(X**2, dim=2)
        Ys = torch.sum(Y**2, dim=2)
        dist = -2. * torch.einsum('ipk,jqk->ijpq', X, Y)
        dist += torch.reshape(Xs,
                              (A, 1, M, 1)) + torch.reshape(Ys, (1, B, 1, N))
        dist = torch.clamp(dist, min=0)
        return torch.exp(-torch.sqrt(dist) / self.sigma)


class Compare_test_metrics:
    def __init__(self, X, Y, device):
        self.X = X
        self.Y = Y
        self.device = device

    def permutation_test(self, test_func, num_exp, num_perm, sample_size):
        with torch.no_grad():
            true_rej = 0  # power
            false_rej = 0  # type 1 error
            for _ in tqdm(range(num_exp)):
                sample_idx = torch.randperm(self.X.shape[0])
                X = self.X[sample_idx[:sample_size]].to(self.device)
                Y1 = self.Y[sample_idx[:sample_size]].to(self.device)
                Y2 = self.Y[sample_idx[sample_size:2 * sample_size]].to(
                    self.device)

                H0_stats = []
                H1_stats = []
                for _ in range(num_perm):
                    idx = torch.randperm(2 * sample_size)
                    combined = torch.cat([Y1, Y2])
                    H0_stats.append(
                        test_func(combined[idx[:sample_size]], combined[
                            idx[sample_size:]]).cpu().detach().numpy().item())
                    combined = torch.cat([X, Y1])
                    H1_stats.append(
                        test_func(combined[idx[:sample_size]], combined[
                            idx[sample_size:]]).cpu().detach().numpy().item())

                if test_func(Y1, Y2) > np.quantile(H0_stats, 0.95):
                    false_rej += 1
                if test_func(X, Y1) > np.quantile(H1_stats, 0.95):
                    true_rej += 1
            power = true_rej / num_exp
            type1_error = false_rej / num_exp
        return power, type1_error

    def run_HT(self, num_exp, num_perm, sample_size, tag, static_kernel,
               dyadic_order):

        models = []
        powers = []
        type1_errors = []
        tags = []

        kernel = SigKernel(static_kernel=static_kernel,
                           dyadic_order=dyadic_order)
        power, t1error = self.permutation_test(
            kernel.compute_mmd,
            num_exp,
            num_perm,
            sample_size,
        )
        models.append('sig mmd')
        powers.append(power)
        type1_errors.append(t1error)
        tags.append(tag)

        return pd.DataFrame({
            'model': models,
            'power': powers,
            'type1 error': type1_errors,
            'tag': tags
        })