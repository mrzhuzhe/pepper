import torch
import numpy as np

"""
x = torch.empty(5, 3)
print(x)


a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

"""

if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    x = torch.randn(1)
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)                       # or just use strings ``.to("cuda")``
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!
