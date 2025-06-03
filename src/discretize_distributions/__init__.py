from .discretize import discretize, discretize_gmms_the_old_way
from .optimal import info

from . import distributions
from . import schemes
from . import optimal

__all__ = [
    'discretize'
    'discretize_gmms_the_old_way',
    'distributions', 
    'schemes', 
    'info'
    ]

