import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from functools import partial
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from src.char_func import char_func_path
from src.utils import FBM_data, subsample


class Compare_test_metrics:
    def __init__(self, X, Y1, Y2, device):
        self.X = X
        self.Y1 = Y1
        self.Y2 = Y2
        self.device = device

    def permutation_test(self, test_func, num_exp, num_perm, sample_size):
        with torch.no_grad():
            start = time.time()

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
            end = time.time()

        return power, type1_error, end - start

    def run_HT(self,
               train_X,
               train_Y,
               num_exp=20,
               num_perm=500,
               sample_size=200,
               tag=None,
               num_samples_u=None,
               hidden_size_u=None,
               num_samples_so=None,
               hidden_size_so=None,
               num_samples_d=None,
               hidden_size_d=None,
               batch_size=None,
               max_iter=2000):
        model = []
        power = []
        type1_error = []
        times = []
        infer_times = []
        tags = []

        ##########################################################################################################################################
        # 1. PCFD
        train_X_dl = DataLoader(train_X.to(self.device),
                                batch_size,
                                shuffle=True)
        train_Y_dl = DataLoader(train_Y.to(self.device),
                                batch_size,
                                shuffle=True)

        initial_char_func = char_func_path(num_samples=num_samples_u,
                                           hidden_size=hidden_size_u,
                                           lie_group='unitary',
                                           input_size=self.X.shape[-1],
                                           add_time=True,
                                           init_range=1).to(self.device)

        char_optimizer = torch.optim.Adam(initial_char_func.parameters(),
                                          betas=(0, 0.9),
                                          lr=0.05)

        start_u = time.time()
        for i in tqdm(range(max_iter)):
            X = next(iter(train_X_dl))
            Y = next(iter(train_Y_dl))
            char_optimizer.zero_grad()
            char_loss = -initial_char_func.distance_measure(X, Y, Lambda=0)
            char_loss.backward()
            char_optimizer.step()
        end_u = time.time()

        untrained_power, untrained_t1error, infer_time = self.permutation_test(
            partial(initial_char_func.distance_measure, Lambda=0),
            num_exp,
            num_perm,
            sample_size,
        )
        model.append('PCFD')
        power.append(untrained_power)
        type1_error.append(untrained_t1error)
        times.append(end_u - start_u)
        infer_times.append(infer_time)
        tags.append(tag)
        ##########################################################################################################################################
        # 2. OPCFD
        train_X_dl = DataLoader(train_X.to(self.device),
                                batch_size,
                                shuffle=True)
        train_Y_dl = DataLoader(train_Y.to(self.device),
                                batch_size,
                                shuffle=True)

        initial_char_func_so = char_func_path(num_samples=num_samples_so,
                                              hidden_size=hidden_size_so,
                                              lie_group='orthogonal',
                                              input_size=self.X.shape[-1],
                                              add_time=True,
                                              init_range=1).to(self.device)

        char_optimizer_so = torch.optim.Adam(initial_char_func_so.parameters(),
                                             betas=(0, 0.9),
                                             lr=0.05)

        start_so = time.time()
        for i in tqdm(range(max_iter)):
            X = next(iter(train_X_dl))
            Y = next(iter(train_Y_dl))
            char_optimizer_so.zero_grad()
            char_loss_so = -initial_char_func_so.distance_measure(
                X, Y, Lambda=0)
            char_loss_so.backward()
            char_optimizer_so.step()
        end_so = time.time()

        untrained_power_so, untrained_t1error_so, infer_time = self.permutation_test(
            partial(initial_char_func_so.distance_measure, Lambda=0),
            num_exp,
            num_perm,
            sample_size,
        )
        model.append('OPCFD')
        power.append(untrained_power_so)
        type1_error.append(untrained_t1error_so)
        times.append(end_so - start_so)
        infer_times.append(infer_time)
        tags.append(tag)
        ##########################################################################################################################################
        # 3. RPCFD
        train_X_dl = DataLoader(train_X.to(self.device),
                                batch_size,
                                shuffle=True)
        train_Y_dl = DataLoader(train_Y.to(self.device),
                                batch_size,
                                shuffle=True)

        initial_char_func_diag = char_func_path(
            num_samples=num_samples_d,
            hidden_size=hidden_size_d,
            lie_group='orthogonal_diag',
            input_size=self.X.shape[-1],
            add_time=True,
            init_range=1,
        ).to(self.device)

        char_optimizer_diag = torch.optim.Adam(
            initial_char_func_diag.parameters(), betas=(0, 0.9), lr=0.5)

        start_diag = time.time()
        for i in tqdm(range(max_iter)):
            X = next(iter(train_X_dl))
            Y = next(iter(train_Y_dl))
            char_optimizer_diag.zero_grad()
            char_loss_diag = -initial_char_func_diag.distance_measure(
                X, Y, Lambda=0)
            char_loss_diag.backward()
            char_optimizer_diag.step()
        end_diag = time.time()

        untrained_power_diag, untrained_t1error_diag, infer_time = self.permutation_test(
            partial(initial_char_func_diag.distance_measure, Lambda=0),
            num_exp,
            num_perm,
            sample_size,
        )
        model.append('RPCFD')
        power.append(untrained_power_diag)
        type1_error.append(untrained_t1error_diag)
        times.append(end_diag - start_diag)
        infer_times.append(infer_time)
        tags.append(tag)

        return pd.DataFrame({
            'model': model,
            'power': power,
            'type1 error': type1_error,
            'opt time': times,
            'infer time': infer_times,
            'tag': tags
        })


if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    max_iter = 500
    df_list = []
    for i in range(10):
        for h in [
                0.2, 0.25, 0.3, 0.35, 0.4, 0.425, 0.45, 0.475, 0.5, 0.525,
                0.55, 0.575, 0.6, 0.65, 0.7, 0.75, 0.8
        ]:
            for dim in [5]:
                print('h =', h)
                print('dim =', dim)
                print('trial =', i + 1)

                train_X = FBM_data(10000, dim=3, length=50, h=0.5)
                train_Y = FBM_data(10000, dim=3, length=50, h=h)

                X = FBM_data(10000, dim=3, length=50, h=0.5)
                Y1 = FBM_data(10000, dim=3, length=50, h=h)
                Y2 = FBM_data(10000, dim=3, length=50, h=h)

                df = Compare_test_metrics(X, Y1, Y2, device).run_HT(
                    train_X=train_X,
                    train_Y=train_Y,
                    num_exp=20,
                    num_perm=500,
                    sample_size=200,
                    tag=h,
                    num_samples_u=8,
                    hidden_size_u=dim,
                    num_samples_so=8,
                    hidden_size_so=dim,
                    num_samples_d=8,
                    hidden_size_d=dim,
                    batch_size=1024,
                    max_iter=max_iter,
                )
                print(df)
                df_list.append(df)
                df = pd.concat(df_list)

                if not os.path.exists('numerical_results'):
                    os.mkdir('numerical_results')
                df.to_csv(
                    f'numerical_results/trial={i + 1} h={h} dim={dim}.csv')