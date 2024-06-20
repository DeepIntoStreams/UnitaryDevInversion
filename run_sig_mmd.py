import os
import numpy as np
import pandas as pd
import torch

from sigkernel.static_kernels import LinearKernel, RBFKernel
from sigkernel.sigkernel import SigKernel
from src.utils import fBM_data, add_time, Permutation_test

if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    if not os.path.exists("numerical_results"):
        os.mkdir("numerical_results")

    linear_kernel = SigKernel(static_kernel=LinearKernel(), dyadic_order=0)
    rbf_kernel = SigKernel(static_kernel=RBFKernel(sigma=0.1), dyadic_order=0)
    for i in range(5):
        for h in [
                0.2, 0.25, 0.3, 0.35, 0.4, 0.425, 0.45, 0.475, 0.5, 0.525,
                0.55, 0.575, 0.6, 0.65, 0.7, 0.75, 0.8
        ]:
            print(f"repeat: {i + 1}")
            print(f"h: {h}")

            test_X = add_time(fBM_data(10000, dim=3, length=50,
                                       h=0.5)).double()
            test_Y = add_time(fBM_data(10000, dim=3, length=50, h=h)).double()
            perm_test = Permutation_test(train_X=None,
                                         train_Y=None,
                                         test_X=test_X,
                                         test_Y=test_Y,
                                         num_exp=20,
                                         num_perm=500,
                                         sample_size=200,
                                         device=device)

            print("linear kernel:")
            df = perm_test.permutation_test(linear_kernel.compute_mmd)
            print(df)
            df.to_csv(
                f'numerical_results/repeat={i + 1}_h={h}_linear_kernel.csv')

            print("rbf kernel:")
            df = perm_test.permutation_test(rbf_kernel.compute_mmd)
            print(df)
            df.to_csv(f'numerical_results/repeat={i + 1}_h={h}_rbf_kernel.csv')
