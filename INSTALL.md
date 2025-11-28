# Install / Development

Basic steps to install KAIR from the project root (the directory that contains `pyproject.toml`):

- Install in editable/development mode (recommended while modifying code):

```bash
pip install -e .
```

- Install as a standard package:

```bash
pip install .
```

Notes:
- The `pyproject.toml` lists runtime dependencies from `requirement.txt` (opencv-python, scikit-image, pillow, torchvision, hdf5storage, ninja, lmdb, requests, timm, einops).
- This repository often depends on torch (PyTorch) — PyTorch installs vary by platform/CUDA version; install the correct `torch` wheel before or after installing this package if not already installed.
