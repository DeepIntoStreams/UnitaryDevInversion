import os
import numpy as np
import pandas as pd
from random import shuffle
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from functools import partial
from tqdm import tqdm
from src.PCFGAN.PCFGAN import char_func_path
from src.fbm_dl import FBM_data
from src.utils import tri_diag


class Compare_test_metrics:
    def __init__(self, X, Y, device):
        self.X = X
        self.Y = Y
        self.device = device

    def permutation_test(self, test_func, num_perm, sample_size):
        with torch.no_grad():
            X = self.subsample(self.X, sample_size)
            Y = self.subsample(self.Y, sample_size)
            X = X.to(self.device)
            Y = Y.to(self.device)

            n, m = X.shape[0], Y.shape[0]
            combined = torch.cat([X, Y])
            H0_stats = []
            H1_stats = []

            for i in tqdm(range(num_perm)):
                idx = torch.randperm(n + m)
                H0_stats.append(
                    test_func(combined[idx[:n]],
                              combined[idx[n:]]).cpu().detach().numpy())
                H1_stats.append(
                    test_func(
                        self.subsample(self.X, sample_size).to(self.device),
                        self.subsample(self.Y, sample_size).to(self.device),
                    ).cpu().detach().numpy())
            Q_a = np.quantile(np.array(H0_stats), q=0.95)
            Q_b = np.quantile(np.array(H1_stats), q=0.05)

            power = 1 - (Q_a > np.array(H1_stats)).sum() / num_perm
            type1_error = (Q_b < np.array(H0_stats)).sum() / num_perm
        return power, type1_error

    def run_HT(
        self,
        num_run,
        train_X,
        train_Y,
        sample_size=200,
        num_permutations=500,
        tag=None,
        num_samples=64,
        hidden_size=4,
    ):
        model = []
        power = []
        type1_error = []
        tags = []

        train_X_dl = DataLoader(train_X.to(self.device), 128, shuffle=True)
        train_Y_dl = DataLoader(train_Y.to(self.device), 128, shuffle=True)

        initial_char_func = char_func_path(num_samples=num_samples,
                                           hidden_size=hidden_size,
                                           input_size=train_X.shape[-1],
                                           add_time=True,
                                           init_range=1).to(self.device)

        untrained_power, untrained_t1error = self.permutation_test(
            partial(initial_char_func.distance_measure, Lambda=0),
            num_permutations,
            sample_size,
        )
        model.append('Random')
        power.append(untrained_power)
        type1_error.append(untrained_t1error)
        tags.append(tag)

        initial_char_func_diag = char_func_path(
            num_samples=num_samples,
            hidden_size=hidden_size,
            input_size=train_X.shape[-1],
            add_time=True,
            init_range=1,
        ).to(self.device)
        for param in initial_char_func_diag.parameters():
            param.requires_grad = False
            param = tri_diag(param)
        untrained_power_diag, untrained_t1error_diag = self.permutation_test(
            partial(initial_char_func_diag.distance_measure, Lambda=0),
            num_permutations,
            sample_size,
        )
        model.append('Restricted Random')
        power.append(untrained_power_diag)
        type1_error.append(untrained_t1error_diag)
        tags.append(tag)

        sig_inv = char_func_path(
            num_samples=num_samples,
            hidden_size=hidden_size,
            input_size=train_X.shape[-1],
            add_time=True,
            init_range=1,
        ).to(self.device)
        for param in sig_inv.parameters():
            param.requires_grad = False
            param = param * 0
            n = 0
            for i in range(4):
                for j in range(4):
                    for k in range(4):
                        param[i, n, 0, 1] = complex(1., 0)
                        param[j, n, 1, 2] = complex(1., 0)
                        param[k, n, 2, 3] = complex(1., 0)
                        n += 1
        sig_inv_power, sig_inv_t1error = self.permutation_test(
            partial(sig_inv.distance_measure, Lambda=0),
            num_permutations,
            sample_size,
        )
        model.append('Signature Inversion')
        power.append(sig_inv_power)
        type1_error.append(sig_inv_t1error)
        tags.append(tag)

        return pd.DataFrame({
            'model': model,
            'power': power,
            'type1 error': type1_error,
            'tag': tags
        })

    def subsample(self, data, sample_size):
        idx = torch.randint(low=0, high=data.shape[0], size=[sample_size])
        return data[idx]


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    df_list = []
    h_list = [0.3, 0.34, 0.38, 0.42, 0.46, 0.5, 0.54, 0.58, 0.62, 0.66, 0.7]

    X = FBM_data(10000, dim=3, length=50, h=0.5)
    train_X = FBM_data(5000, dim=3, length=50, h=0.5)

    for h in h_list:
        print('h =', h)
        Y = FBM_data(10000, dim=3, length=50, h=h)
        train_Y = FBM_data(5000, dim=3, length=50, h=h)

        for _ in range(5):
            df = Compare_test_metrics(X, Y, device).run_HT(
                num_run=np.nan,
                train_X=train_X,
                train_Y=train_Y,
                tag=h,
                num_samples=64,
                hidden_size=4,
            )
            print(df)
            df_list.append(df)
    df = pd.concat(df_list)
    if not os.path.exists('numerical_results'):
        os.mkdir('numerical_results')
    df.to_csv('numerical_results/ht_fbm.csv')