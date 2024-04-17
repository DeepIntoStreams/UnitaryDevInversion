import os
import numpy as np
import pandas as pd
import torch
from src.utils import FBM_data, Compare_test_metrics

if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    K = 8
    k = 5
    df_list = []
    for i in range(5):
        for h in [
                0.2, 0.25, 0.3, 0.35, 0.4, 0.425, 0.45, 0.475, 0.5, 0.525,
                0.55, 0.575, 0.6, 0.65, 0.7, 0.75, 0.8
        ]:
            print('trial =', i + 1)
            print('h =', h)

            train_X = FBM_data(10000, dim=3, length=50, h=0.5)
            train_Y = FBM_data(10000, dim=3, length=50, h=h)
            test_X = FBM_data(10000, dim=3, length=50, h=0.5)
            test_Y = FBM_data(10000, dim=3, length=50, h=h)

            df = Compare_test_metrics(test_X, test_Y, device).run_HT(
                train_X=train_X,
                train_Y=train_Y,
                num_exp=20,
                num_perm=500,
                sample_size=200,
                tag=h,
                num_samples_u=K,
                hidden_size_u=k,
                num_samples_so=K,
                hidden_size_so=k,
                num_samples_d=K,
                hidden_size_d=k,
                batch_size=1024,
                max_iter=500,
            )
            print(df)
            df_list.append(df)
            df = pd.concat(df_list)

            if not os.path.exists('numerical_results'):
                os.mkdir('numerical_results')
            df.to_csv(f'numerical_results/trial={i + 1} h={h}.csv')