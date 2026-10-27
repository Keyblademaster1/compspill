"""compspill: AutoSpill-style compensation (spillover) matrices from single-stain controls."""
__version__ = '0.1.0'

from .config import load_config  # noqa: E402
from .estimate import SpilloverResult, estimate_spillover  # noqa: E402
from .io import load_fcs, read_csv, read_flowjo_mtx, write_csv, write_flowjo_mtx  # noqa: E402
from .samples import sample_check  # noqa: E402

__all__ = ['estimate_spillover', 'SpilloverResult', 'sample_check', 'load_config', 'load_fcs',
           'read_csv', 'write_csv', 'read_flowjo_mtx', 'write_flowjo_mtx', '__version__']
