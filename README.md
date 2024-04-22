# On the determination of path signature from its unitary development

## Introduction

This repository is the official implementation of the paper "On the determination of path signature from its unitary development". The folder "Dev" is for the experiments on development-based models (RPCFD, OPCFD and PCFD[[1]](#1)). The folder "Sig MMDs" is for the experiments on signature-based mmds[[2]](#2).


## Environment setup

The repository can be run on Python 3.10. The environment of the folder "Dev" can be set up by the following commands.

```console
conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

The environment of the folder "Sig MMDs" can be set up by the following commands (our implementation relies on the Python package sigkernel[[3]](#3)).

```console
pip install iisignature
pip install Cython
pip install git+https://github.com/crispitagorico/sigkernel.git
pip install -r requirements.txt
```

## References
<a id="1">[1]</a> 
H. Lou, S. Li, and H. Ni, PCF-GAN: generating sequential data via the characteristic function of measures on the path space, *ArXiv preprint* (2023), ArXiv: 2305.12511; accepted by *NeurIPS 2023*.

<a id="2">[2]</a> 
I. Chevyrev and H. Oberhauser, Signature moments to characterize laws of stochastic processes, *J. Mach. Learning Res.* **23** (2002), 1--42.

<a id="3">[3]</a> 
C. Salvi, T. Cass, J. Foster, T. Lyons, and W. Yang, The Signature Kernel Is the Solution of a Goursat PDE, *SIAM J. Math. Data Sci.* **3** (2021), 873--899.
