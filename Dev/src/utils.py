import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from functools import partial
from fbm import FBM
from src.char_func import char_func_path


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

    def run_HT(self, train_X, train_Y, num_exp, num_perm, sample_size, tag,
               num_samples_u, hidden_size_u, num_samples_so, hidden_size_so,
               num_samples_d, hidden_size_d, batch_size, max_iter):

        model = []
        power = []
        type1_error = []
        times = []
        tags = []
        ##########################################################################################################################################
        # 1. PCFD
        train_X_dl = DataLoader(train_X.to(self.device),
                                batch_size,
                                shuffle=True)
        train_Y_dl = DataLoader(train_Y.to(self.device),
                                batch_size,
                                shuffle=True)

        PCFD_char_func = char_func_path(num_samples=num_samples_u,
                                        hidden_size=hidden_size_u,
                                        lie_group='unitary',
                                        input_size=self.X.shape[-1],
                                        add_time=True,
                                        init_range=1).to(self.device)

        char_optimizer = torch.optim.Adam(PCFD_char_func.parameters(),
                                          betas=(0, 0.9),
                                          lr=0.05)
        start = time.time()
        for _ in tqdm(range(max_iter)):
            X = next(iter(train_X_dl))
            Y = next(iter(train_Y_dl))
            char_optimizer.zero_grad()
            char_loss = -PCFD_char_func.distance_measure(X, Y, Lambda=0)
            char_loss.backward()
            char_optimizer.step()
        end = time.time()

        PCFD_power, PCFD_t1error = self.permutation_test(
            partial(PCFD_char_func.distance_measure, Lambda=0),
            num_exp,
            num_perm,
            sample_size,
        )
        model.append('PCFD')
        power.append(PCFD_power)
        type1_error.append(PCFD_t1error)
        times.append(end - start)
        tags.append(tag)
        ##########################################################################################################################################
        # 2. OPCFD
        train_X_dl = DataLoader(train_X.to(self.device),
                                batch_size,
                                shuffle=True)
        train_Y_dl = DataLoader(train_Y.to(self.device),
                                batch_size,
                                shuffle=True)

        OPCFD_char_func = char_func_path(num_samples=num_samples_so,
                                         hidden_size=hidden_size_so,
                                         lie_group='orthogonal',
                                         input_size=self.X.shape[-1],
                                         add_time=True,
                                         init_range=1).to(self.device)

        char_optimizer = torch.optim.Adam(OPCFD_char_func.parameters(),
                                          betas=(0, 0.9),
                                          lr=0.05)

        start = time.time()
        for _ in tqdm(range(max_iter)):
            X = next(iter(train_X_dl))
            Y = next(iter(train_Y_dl))
            char_optimizer.zero_grad()
            char_loss = -OPCFD_char_func.distance_measure(X, Y, Lambda=0)
            char_loss.backward()
            char_optimizer.step()
        end_so = time.time()

        OPCFD_power, OPCFD_t1error = self.permutation_test(
            partial(OPCFD_char_func.distance_measure, Lambda=0),
            num_exp,
            num_perm,
            sample_size,
        )
        model.append('OPCFD')
        power.append(OPCFD_power)
        type1_error.append(OPCFD_t1error)
        times.append(end - start)
        tags.append(tag)
        ##########################################################################################################################################
        # 3. RPCFD
        train_X_dl = DataLoader(train_X.to(self.device),
                                batch_size,
                                shuffle=True)
        train_Y_dl = DataLoader(train_Y.to(self.device),
                                batch_size,
                                shuffle=True)

        RPCFD_char_func = char_func_path(
            num_samples=num_samples_d,
            hidden_size=hidden_size_d,
            lie_group='orthogonal_diag',
            input_size=self.X.shape[-1],
            add_time=True,
            init_range=1,
        ).to(self.device)

        char_optimizer = torch.optim.Adam(RPCFD_char_func.parameters(),
                                          betas=(0, 0.9),
                                          lr=0.5)

        start = time.time()
        for _ in tqdm(range(max_iter)):
            X = next(iter(train_X_dl))
            Y = next(iter(train_Y_dl))
            char_optimizer.zero_grad()
            char_loss = -RPCFD_char_func.distance_measure(X, Y, Lambda=0)
            char_loss.backward()
            char_optimizer.step()
        end = time.time()

        RPCFD_power, RPCFD_t1error = self.permutation_test(
            partial(RPCFD_char_func.distance_measure, Lambda=0),
            num_exp,
            num_perm,
            sample_size,
        )
        model.append('RPCFD')
        power.append(RPCFD_power)
        type1_error.append(RPCFD_t1error)
        times.append(end - start)
        tags.append(tag)

        return pd.DataFrame({
            'model': model,
            'power': power,
            'type1 error': type1_error,
            'opt time': times,
            'tag': tags
        })