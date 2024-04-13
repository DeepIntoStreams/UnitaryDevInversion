import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
import time
from utils import FBM_data, subsample, AddTime
from sigkernel.sigkernel import SigKernel
from sigkernel.static_kernels import LinearKernel, RBFKernel


class LaplaceKernel():
    """Laplace kernel k: R^d x R^d -> R"""
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
    def __init__(self, X, Y1, Y2, device):
        self.X = X
        self.Y1 = Y1
        self.Y2 = Y2
        self.device = device

    def permutation_test(self, test_func, num_exp, num_perm, sample_size):
        with torch.no_grad():
            true_rej = 0  # power
            false_rej = 0  # type 1 error
            for i in tqdm(range(num_exp)):
                X = subsample(self.X, sample_size)
                Y1 = subsample(self.Y1, sample_size)
                Y2 = subsample(self.Y2, sample_size)
                X = X.to(self.device)
                Y1 = Y1.to(self.device)
                Y2 = Y2.to(self.device)

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

    def run_HT(self,
               num_exp=20,
               num_perm=500,
               sample_size=200,
               tag=None,
               static_kernel=None,
               dyadic_order=None):
        models = []
        powers = []
        type1_errors = []
        tags = []

        # sig mmd
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


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    linear_kernel = LinearKernel()
    #     sigma = 0.1
    #     laplace_kernel = LaplaceKernel(sigma=sigma)
    #     rbf_kernel = RBFKernel(sigma=sigma)

    df_list = []
    for h in [
            0.2, 0.25, 0.3, 0.35, 0.4, 0.425, 0.45, 0.475, 0.5, 0.525, 0.55,
            0.575, 0.6, 0.65, 0.7, 0.75, 0.8
    ]:
        print('h =', h)

        X = AddTime(FBM_data(10000, dim=3, length=50, h=0.5)).double()
        Y1 = AddTime(FBM_data(10000, dim=3, length=50, h=h)).double()
        Y2 = AddTime(FBM_data(10000, dim=3, length=50, h=h)).double()

        df = Compare_test_metrics(X, Y1, Y2, device).run_HT(
            num_exp=20,
            num_perm=500,
            sample_size=200,
            tag=h,
            static_kernel=linear_kernel,
#             static_kernel=rbf_kernel,
#             static_kernel=laplace_kernel,
            dyadic_order=0,
        )
        print(df)
        df_list.append(df)
        df = pd.concat(df_list)

        if not os.path.exists('numerical_results'):
            os.mkdir('numerical_results')
            df.to_csv(f'numerical_results/sig mmd h={h}.csv')