"""TrustQAT: trustworthiness-aware quantization-aware training.

The package is split into three layers:

- ``trustqat.core``: framework-free (numpy) metrics, Pareto selection, epoch records,
  temperature scaling and paired comparisons.
- ``trustqat.keras`` / ``trustqat.torch``: framework adapters with the training-time
  implementations of trustworthiness objectives.
- ``trustqat.backends``: QAT backends (HGQ2 ships with the package; others plug in
  through the ``trustqat.backends`` entry-point group).

Importing ``trustqat`` itself does not import Keras, PyTorch or any QAT library.
"""

from ._version import __version__

__all__ = ['__version__']
