from .lol import register_lol_dataset
from .sice import register_sice_dataset

from . import builtin as _builtin # ensure all builtin datasets are registered
