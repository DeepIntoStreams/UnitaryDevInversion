import torch
from torch import nn
import math
from functools import partial
from src.unitary import rescaled_matrix_exp


class orthogonal(nn.Module):

    """
    A PyTorch module for parametrizing a unitary Lie algebra using a general linear matrix.
    """

    def __init__(self, size):
        """
        Initializes the unitary module.

        Args:
            size (torch.size): Size of the tensor to be parametrized.
        """
        super().__init__()
        self.size = size

    @staticmethod
    def frame(X: torch.tensor) -> torch.tensor:
        """
        Parametrizes the unitary Lie algebra from the general linear matrix X.

        Args:
            X (torch.Tensor): A tensor of shape (...,2n,2n).

        Returns:
            torch.Tensor: A tensor of shape (...,2n,2n).
        """
        X = (X - X.transpose(-2, -1)) / 2

        return X

    def forward(self, X: torch.tensor) -> torch.tensor:
        """
        Applies the frame method to the input tensor X.

        Args:
            X (torch.Tensor): A tensor to be parametrized.

        Returns:
            torch.Tensor: A tensor parametrized in the unitary Lie algebra.

        Raises:
            ValueError: If the input tensor has fewer than 2 dimensions or the last two dimensions are not square.
        """
        if len(X.size()) < 2:
            raise ValueError("weights has dimension < 2")
        if X.size(-2) != X.size(-1):
            raise ValueError("not sqaured matrix")
        return self.frame(X)

    @staticmethod
    def in_lie_algebra(X, eps=1e-5):
        """
        Checks if a given tensor is in the Lie algebra.

        Args:
            X: The tensor to check.
            eps (float): Optional. The tolerance for checking closeness to zero.

        Returns:
            bool: True if the tensor is in the Lie algebra, False otherwise.
        """
        return (
            X.dim() >= 2
            and X.size(-2) == X.size(-1)
            and torch.allclose(X.transpose(-2, -1), -X, atol=eps)
        )

def orthogonal_lie_init_(tensor: torch.tensor, init_=None):
    """
    Fills in the input tensor in place with initialization on the unitary Lie algebra.

    Args:
        tensor (torch.Tensor): A multi-dimensional tensor.
        init_ (callable): Optional. A function that initializes the tensor according to some distribution.

    Raises:
        ValueError: If the tensor has less than 2 dimensions or the last two dimensions are not square.

    """
    if tensor.ndim < 2 or tensor.size(-1) != tensor.size(-2):
        raise ValueError(
            "Only tensors with 2 or more dimensions which are square in "
            "the last two dimensions are supported. "
            "Got a tensor of shape {}".format(tuple(tensor.size()))
        )

    n = tensor.size(-2)
    tensorial_size = tensor.size()[:-2]

    # Non-zero elements that we are going to set on the diagonal
    n_diag = n

    # set values for upper trianguler matrix
    off_diag = tensor.new(tensorial_size + (2 * n, n))
    if init_ is None:
        torch.nn.init.uniform_(off_diag, -math.pi, math.pi)

    else:
        init_(off_diag)

    upper_tri_real = torch.triu(off_diag[..., :n, :n], 1).real

    real_part = (upper_tri_real - upper_tri_real.transpose(-2, -1)) / torch.tensor(
        [2], device=tensor.device
    ).sqrt()

    with torch.no_grad():
        # First non-central diagonal
        x = real_part
        if orthogonal(n).in_lie_algebra(x):
            tensor.copy_(x)
            return tensor
        else:
            raise ValueError("initialize not in Lie")


class orthogonal_projection(nn.Module):
    def __init__(self, input_size, hidden_size, channels=1, init_range=1, **kwargs):
        """
        Projection module used to project the path increments to the Lie group path increments
        using trainable weights from the Lie algebra.

        Args:
            input_size (int): Input size.
            hidden_size (int): Size of the hidden Lie algebra matrix.
            channels (int, optional): Number of channels to produce independent Lie algebra weights. Defaults to 1.
            init_range (int, optional): Range for weight initialization. Defaults to 1.
        """
        self.__dict__.update(kwargs)

        A = torch.empty(
            input_size, channels, hidden_size, hidden_size, dtype=torch.float
        )
        self.channels = channels
        super(orthogonal_projection, self).__init__()
        self.param_map = orthogonal(hidden_size)
        self.A = nn.Parameter(A)

        self.triv = torch.linalg.matrix_exp
        self.init_range = init_range
        self.reset_parameters()

        self.hidden_size = hidden_size

    def reset_parameters(self):
        orthogonal_lie_init_(self.A, partial(nn.init.normal_, std=1))

    def forward(self, dX: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the projection module.

        Args:
            dX (torch.Tensor): Tensor of shape (N, input_size).

        Returns:
            torch.Tensor: Tensor of shape (N, channels, hidden_size, hidden_size).
        """
        A = self.param_map(self.A).permute(1, 2, -1, 0)  # C,m,m,in
        AX = A.matmul(dX.T).permute(-1, 0, 1, 2)  # ->C,m,m,N->N,C,m,m

        return rescaled_matrix_exp(self.triv, AX)