"""Exit 0 if PyTorch can run the models on the NVIDIA GPU, 1 if not (used by run.bat / run.sh).

A GPU can be detected yet unusable when the installed PyTorch has no kernels for it, e.g. an
RTX 50-series card with a CUDA 12.6 build ("no kernel image is available").
"""

import sys

try:
    import torch

    if not torch.cuda.is_available():
        print("PyTorch cannot see the NVIDIA GPU.")
        sys.exit(1)
    x = torch.rand(1, 1, 8, 8, 8, device="cuda")
    torch.nn.functional.conv3d(x, torch.rand(2, 1, 3, 3, 3, device="cuda"))
    torch.cuda.synchronize()
    print("GPU ready:", torch.cuda.get_device_name(0))
except Exception as exc:  # noqa: BLE001
    print("The NVIDIA GPU is not usable with this PyTorch build:", str(exc).splitlines()[0])
    sys.exit(1)
