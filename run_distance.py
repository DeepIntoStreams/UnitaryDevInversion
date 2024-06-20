import os
import numpy as np
import pandas as pd
import torch

from src.utils import fBM_data, Permutation_test

if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    if not os.path.exists("numerical_results"):
        os.mkdir("numerical_results")

    K = 8  # linear maps in the empirical distances
    k = 5  # matrix order of the linear maps' image
    for i in range(5):
        for h in [
                0.2, 0.25, 0.3, 0.35, 0.4, 0.425, 0.45, 0.475, 0.5, 0.525,
                0.55, 0.575, 0.6, 0.65, 0.7, 0.75, 0.8
        ]:
            train_X = fBM_data(10000, dim=3, length=50, h=0.5)
            train_Y = fBM_data(10000, dim=3, length=50, h=h)
            test_X = fBM_data(10000, dim=3, length=50, h=0.5)
            test_Y = fBM_data(10000, dim=3, length=50, h=h)
            perm_test = Permutation_test(train_X=train_X,
                                         train_Y=train_Y,
                                         test_X=test_X,
                                         test_Y=test_Y,
                                         num_exp=20,
                                         num_perm=500,
                                         sample_size=200,
                                         device=device)

            for distance in ["RPCFD", "PCFD", "OPCFD"]:
                print(f"repeat: {i + 1}")
                print(f"h: {h}")
                print(f"distance: {distance}")
                df = perm_test.run(distance,
                                   num_samples=K,
                                   hidden_size=k,
                                   batch_size=1024,
                                   max_iter=500)
                print(df)
                df.to_csv(
                    f"numerical_results/repeat={i + 1}_h={h}_distance={distance}.csv"
                )
