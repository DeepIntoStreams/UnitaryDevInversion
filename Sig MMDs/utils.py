import numpy as np
import torch
from fbm import FBM


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


def subsample(data, sample_size):
    idx = torch.randint(low=0, high=data.shape[0], size=[sample_size])
    return data[idx]