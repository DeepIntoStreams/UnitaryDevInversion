import torch
import torch.nn as nn
from src.utils import AddTime
from src.development_layer import development_layer


class char_func_path(nn.Module):

    def __init__(
        self,
        num_samples,
        hidden_size,
        lie_group,
        input_size,
        add_time: bool,
        init_range: float = 1,
    ):
        """
        Class for computing path charateristic function.

        Args:
            num_samples (int): Number of samples.
            hidden_size (int): Hidden size.
            input_size (int): Input size.
            add_time (bool): Whether to add time dimension to the input.
            init_range (float, optional): Range for weight initialization. Defaults to 1.
        """
        super(char_func_path, self).__init__()
        self.num_samples = num_samples
        self.hidden_size = hidden_size
        self.lie_group = lie_group
        self.input_size = input_size
        if add_time:
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
        self.add_time = add_time

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
        if self.add_time:
            X1 = AddTime(X1)
            X2 = AddTime(X2)
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