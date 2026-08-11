# Pure/exceptions/__init__.py

from .StagnationException import StagnationException
from .StatusMismatchException import StatusMismatchException

__all__ = [
    "StagnationException",
    "StatusMismatchException",
]