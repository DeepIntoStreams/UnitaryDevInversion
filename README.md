# On the determination of path signature from its unitary development

## Introduction

This repository is the official implementation of the paper "On the determination of path signature from its unitary development". The file "run_distance.py" is for the experiments on development-based models, i.e. RPCFD, OPCFD and PCFD[[1]](#1). The file "run_sig_mmd.py" is for the experiments on signature-based mmds[[2]](#2).


## Environment setup

The repository is run on Python 3.7.16, which can be set up by the following commands:

```console
pip install -r requirements.txt
pip install git+https://github.com/crispitagorico/sigkernel.git
```

Our implementation for signature-based mmds relies on the Python package sigkernel[[3]](#3).


## References
<a id="1">[1]</a> 
H. Lou, S. Li, and H. Ni, PCF-GAN: generating sequential data via the characteristic function of measures on the path space, *ArXiv preprint* (2023), ArXiv: 2305.12511; accepted by *NeurIPS 2023*.

<a id="2">[2]</a> 
I. Chevyrev and H. Oberhauser, Signature moments to characterize laws of stochastic processes, *J. Mach. Learning Res.* **23** (2002), 1--42.

<a id="3">[3]</a> 
C. Salvi, T. Cass, J. Foster, T. Lyons, and W. Yang, The Signature Kernel Is the Solution of a Goursat PDE, *SIAM J. Math. Data Sci.* **3** (2021), 873--899.
