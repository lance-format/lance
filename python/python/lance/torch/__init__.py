# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from typing import Optional, Union

from lance.dependencies import torch


def preferred_device(device: Optional[str] = None):
    """Get the preferred device for computation.

    Parameters
    ----------
    device : str, optional
        Device to use for computation. If None, the device will be
        detected automatically based on the platform.

    Returns
    -------
    device : torch.device
        Device to use for computation.
    """
    if device is not None:
        if isinstance(device, str):
            device = torch.device(device)
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def _to_device(
    data: "torch.Tensor", device: "Union[str, torch.device]"
) -> "torch.Tensor":
    """Move ``data`` to ``device``.

    MPS has no float64, so float64 becomes float32 there. Results computed on
    such data, including ``ground_truth``, have float32 precision, so nearly tied
    neighbours can change order. Use ``device="cpu"`` when double precision is
    needed.
    """
    if torch.device(device).type == "mps" and data.dtype == torch.float64:
        data = data.type(torch.float32)
    return data.to(device)
