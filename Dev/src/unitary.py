from functools import partial

import numpy as np
import torch
import torch.nn as nn
import math


def unitary_lie_init_(tensor: torch.tensor, init_=None):
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
    diag = tensor.new(tensorial_size + (n_diag,))
    if init_ is None:
        torch.nn.init.uniform_(diag, -math.pi, math.pi)
    else:
        init_(diag)
    diag = diag.imag * torch.tensor([1j], device=tensor.device)
    # set values for upper trianguler matrix
    off_diag = tensor.new(tensorial_size + (2 * n, n))
    if init_ is None:
        torch.nn.init.uniform_(off_diag, -math.pi, math.pi)

    else:
        init_(off_diag)

    upper_tri_real = torch.triu(off_diag[..., :n, :n], 1).real.cfloat()
    upper_tri_complex = torch.triu(
        off_diag[..., n:, :n], 1
    ).imag.cfloat() * torch.tensor([1j], device=tensor.device)

    real_part = (upper_tri_real - upper_tri_real.transpose(-2, -1)) / torch.tensor(
        [2], device=tensor.device
    ).cfloat().sqrt()
    complex_part = (
        upper_tri_complex + upper_tri_complex.transpose(-2, -1)
    ) / torch.tensor([2], device=tensor.device).cfloat().sqrt()

    with torch.no_grad():
        # First non-central diagonal
        x = real_part + complex_part + torch.diag_embed(diag)
        if unitary(n).in_lie_algebra(x):
            tensor = tensor.cfloat()
            tensor.copy_(x)
            return tensor
        else:
            raise ValueError("initialize not in Lie")


class unitary(nn.Module):

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
        X = (X - torch.conj(X.transpose(-2, -1))) / 2

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
            and torch.allclose(torch.conj(X.transpose(-2, -1)), -X, atol=eps)
        )


class unitary_projection(nn.Module):
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
            input_size, channels, hidden_size, hidden_size, dtype=torch.cfloat
        )
        self.channels = channels
        super(unitary_projection, self).__init__()
        self.param_map = unitary(hidden_size)
        self.A = nn.Parameter(A)

        self.triv = torch.linalg.matrix_exp
        self.init_range = init_range
        self.reset_parameters()

        self.hidden_size = hidden_size

    def reset_parameters(self):
        unitary_lie_init_(self.A, partial(nn.init.normal_, std=1))

    def M_initialize(self, A):
        init_range = np.linspace(0, 10, self.channels + 1)
        for i in range(self.channels):
            A[:, i] = unitary_lie_init_(
                A[:, i], partial(nn.init.uniform_, a=init_range[i], b=init_range[i + 1])
            )
        return A

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


def matrix_power_two_batch(A, k):
    """
    Computes the matrix power of A for each element in k using batch processing.

    Args:
        A (torch.Tensor): Input tensor of shape (..., m, m).
        k (torch.Tensor): Exponent tensor of shape (...).

    Returns:
        torch.Tensor: Resulting tensor of shape (..., m, m).
    """
    orig_size = A.size()
    A, k = A.flatten(0, -3), k.flatten()
    ksorted, idx = torch.sort(k)
    # Abusing bincount...
    count = torch.bincount(ksorted)
    nonzero = torch.nonzero(count, as_tuple=False)
    A = torch.matrix_power(A, 2 ** ksorted[0])
    last = ksorted[0]
    processed = count[nonzero[0]]
    for exp in nonzero[1:]:
        new, last = exp - last, exp
        A[idx[processed:]] = torch.matrix_power(A[idx[processed:]], 2 ** new.item())
        processed += count[exp]
    return A.reshape(orig_size)


def rescaled_matrix_exp(f, A):
    """
    Computes the rescaled matrix exponential of A.
    By following formula exp(A) = (exp(A/k))^k

    Args:
        f (callable): Function to compute the matrix exponential.
        A (torch.Tensor): Input tensor of shape (..., m, m).

    Returns:
        torch.Tensor: Resulting tensor of shape (..., m, m).
    """
    normA = torch.max(torch.sum(torch.abs(A), axis=-2), axis=-1).values
    more = normA > 1
    s = torch.ceil(torch.log2(normA)).long()
    s = normA.new_zeros(normA.size(), dtype=torch.long)
    s[more] = torch.ceil(torch.log2(normA[more])).long()
    A_1 = torch.pow(0.5, s.float()).unsqueeze_(-1).unsqueeze_(-1).expand_as(A) * A
    return matrix_power_two_batch(f(A_1), s)
