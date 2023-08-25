The code can identify multidimensional stochastic diffusion processes

$$
d{\mathbf{x}}(t)=\mathbf a({\mathbf{x}},t)dt+\mathbf b({\mathbf{x}},t) d\mathbf{W}(t)
$$

and stochastic jump-diffusion processes

$$
d{\mathbf{x}}(t)=\mathbf a({\mathbf{x}},t)dt+\mathbf b({\mathbf{x}},t) d\mathbf{W}(t)+\mathbf{c }({\mathbf{x}},t) d\mathbf{J}(t)
$$

by directly computing the Kramers-Moyal coefficients from time series.

This package contains implementations of all the examples in the paper "A kernel-based learning framework for discovering the governing equations of stochastic jump-diffusion processes directly from data". 

The proposed method for computing the Kramers-Moyal coefficients is given in kmc.py. Make sure your Python environment contains the common libraries in the kmc.py file.

1D_diffusion.dat, 1D_jump.dat, 2D_jump.dat, and 2D_jump2.dat represent the data of the examples of the stochastic diffusion and the stochastic jump-diffusion cases in one and two dimensions, respectively. 

The ”.ipynb“ files with the same name as the data files correspond to the implementation programs for each example. You can try running these .ipynb files as demos.  These ipynb files follow the programming presentation of jupyter, make sure you have jupyter installed in your python environment before running them.
