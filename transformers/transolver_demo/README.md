# Transolver++ Demo (1D Helmholtz)

This demo trains the Transolver++ model on a simple 1D Helmholtz/Poisson family:

u''(x) + (a*pi)^2 u(x) = 0,  x in [0,1],  u(0)=u(1)=0,

with analytic solutions u(x)=sin(a*pi*x). The condition `a` is provided to the model as a global vector.

## Files
- `dataset.py` — synthetic dataset and config
- `train_demo.py` — training loop with MSE + boundary + residual losses

## Run
```
python -m transformers.transolver_demo.train_demo
```

Notes:
- Requires PyTorch. Install it per your CUDA setup: https://pytorch.org/get-started/locally/
- The model lives in `transformers/transolver.py`.
