from functools import partial

import torch
import torch.nn as nn
import math

from src.unitary import unitary_projection


class upper_triangular(nn.Module):
    def __init__(self):
        """
        real symplectic lie algebra matrices, parametrized in terms of
        by a general linear matrix with shape (2n,2n ).
        Args:
            size (torch.size): Size of the tensor to be parametrized
        """
        super().__init__()

    @ staticmethod
    def frame(X: torch.tensor) -> torch.tensor:
        """ parametrise real upper triangular lie algebra from the general linear matrix X

        Args:
            X (torch.tensor): (...,n)
        Returns:
            torch.tensor: (...,n,n)
        """
        # Create an n x n zero matrix
        n = X.shape[-1]

        matrix = torch.zeros((n, n)).to(X.device).to(X.dtype)

        # Fill the strictly upper triangular part of the matrix
        indices = torch.triu_indices(n, n, offset=1)
        # matrix[indices[0], indices[1]] = vec

        return X

    def forward(self, X: torch.tensor) -> torch.tensor:
        if len(X.size()) < 2:
            raise ValueError('weights has dimension < 2')
        if X.size(-2) != X.size(-1):
            raise ValueError('not squared matrix')
        return self.frame(X)

    @ staticmethod
    def in_lie_algebra(X, eps=1e-5):
        return (X.dim() >= 2
                and X.size(-2) == X.size(-1)
                and torch.allclose(torch.conj(X.transpose(-2, -1)), -X, atol=eps))


def upper_triangular_lie_init_(tensor: torch.tensor, init_=None):
    """
    Fills in the input tensor in place with initialization on the unitary Lie algebra.

    Args:
        tensor (torch.Tensor): A multi-dimensional tensor.
        init_ (callable): Optional. A function that initializes the tensor according to some distribution.

    Raises:
        ValueError: If the tensor has less than 2 dimensions or the last two dimensions are not square.

    """
    with torch.no_grad():
        if tensor.ndim < 2:
            raise ValueError(
                "Only tensors with 2 or more dimensions are supported. "
                "Got a tensor of shape {}".format(tuple(tensor.size()))
            )

        if init_ is None:
            torch.nn.init.uniform_(tensor, -math.pi, math.pi)
        else:
            init_(tensor)

    return tensor


class up_projection(unitary_projection):
    def __init__(self, input_size, hidden_size, channels=1, init_range=1, **kwargs):
        """this class is used to project the path increments to the Lie group path increments, with Lie algbra trainable weights.
        Args:
            input_size (int): input size
            hidden_size (int): size of the hidden Lie algbra matrix
            channels (int, optional): number of channels, produce independent Lie algebra weights. Defaults to 1.
            param (method, optional): parametrization method to map the GL matrix to required matrix Lie algebra. Defaults to sp.
            triv (function, optional): the trivialization map from the Lie algebra to its correpsonding Lie group. Defaults to expm.
        """
        super(up_projection, self).__init__(input_size, hidden_size, channels, init_range, **kwargs)

        print("Using upper triangular")
        A = torch.empty(input_size, channels, hidden_size-1, dtype=torch.float)
        self.channels = channels
        # self.size = hidden_size
        self.param_map = upper_triangular()
        # A = self.M_initialize(A)
        self.A = nn.Parameter(A)

        self.triv = torch.matrix_exp
        self.init_range = init_range
        self.reset_parameters()
        self.col_idx = col_idx(self.hidden_size)
        self.hidden_size = hidden_size

    def reset_parameters(self):
        # print(self.A[0])
        self.A = upper_triangular_lie_init_(self.A, partial(nn.init.normal_, std=1))
        # print(self.A[0])
        return

    def forward(self, dX: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dX (torch.tensor): (N,input_size)
        Returns:
            torch.tensor: (N,channels,hidden_size,hidden_size)
        """
        # A = self.A/torch.linalg.matrix_norm(self.A).mean()

        A = self.A.permute(1, 2, 0) # C,m,in
        #A = A/torch.linalg.matrix_norm(A).mean()
        AX = A.matmul(dX.T).permute(-1, 0, 1)  # ->C,m,N->N,C,m

        return upper_matrix_exp(AX)


def upper_matrix_exp(A, return_matrix = True):
    """
        Computes the rescaled matrix exponential of A.
        By following formula exp(A) = (exp(A/k))^k

        Args:
            f (callable): Function to compute the matrix exponential.
            A (torch.Tensor): Input tensor of shape (N, C, m).

        Returns:
            if matrix form:
            torch.Tensor: Resulting tensor of shape (..., m, m).
            if vector form:
            torch.Tensor: Resulting tensor of shape (..., m). Arranged in diagonal form
    """

    N, C, m = A.shape

    out_dim = (m + 1) * m // 2
    res = torch.zeros([N, C, out_dim]).to(A.device)
    res[:, :, :m] = A
    rolling_idx = m
    B_n = A
    for i in range(1, m):
        B_n = (B_n[:, :, :-1] * A[:, :, i:])
        #         print(B_n)
        #         print(i+1, 1/math.factorial(i+1))

        res[:, :, rolling_idx:rolling_idx + m - i] = 1 / math.factorial(i + 1) * B_n
        rolling_idx += m - i

    if return_matrix:
        # return vector_to_matrix(res)
        res_matrix = torch.eye(m + 1).repeat([N, C, 1, 1]).to(A.device)
        _, col_indices = torch.triu_indices(m + 1, m + 1, offset=1)
        #     print(col_indices)
        temp = 1 + torch.tensor(sum([[i] * (m - i) for i in range(m + 1)], []))
        #     print(temp)

        row_indices = col_indices - temp
        #     print(row_indices)
        res_matrix[:, :, row_indices, col_indices] = res

        return res_matrix
    else:
        return res


def vector_to_matrix(A, hidden_size):
    N, C, m = A.shape
    assert hidden_size*(hidden_size-1)//2 == m, "Dimension does not agree"
    res_matrix = torch.eye(hidden_size).repeat([N, C, 1, 1]).to(A.device)
    _, col_indices = torch.triu_indices(hidden_size, hidden_size, offset=1)
    #     print(col_indices)
    temp = 1 + torch.tensor(sum([[i] * (hidden_size - 1 - i) for i in range(hidden_size)], []))
    #     print(temp)

    row_indices = col_indices - temp
    #     print(row_indices)
    res_matrix[:, :, row_indices, col_indices] = A

    return res_matrix


def col_idx(matrix_size):
    """
    Transform a diagonally arranged matrix into 2D array, each element contains the first i-th columns
    Parameters
    ----------
    matrix_size: int

    Returns
    2D array
    -------

    """
    mm = matrix_size * (matrix_size - 1) // 2
    n = matrix_size - 1
    vector_idx = list(range(mm))
    nested_array = [vector_idx]
    temp = 0
    for j in range(n - 1):
        array = []
        temp = n - 1 - j
        array.append(temp)
        for i in range(n - 1, j, -1):
            temp += i
            array.append(temp)
        nested_array.append(list(set(nested_array[-1]) - set(array)))
    nested_array.reverse()
    nested_array.pop(-1)
    return nested_array
