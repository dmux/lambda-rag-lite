"""Detectores para encoding, tipo de arquivo, etc."""

from .encoding import EncodingDetector
from .file_type import FileTypeDetector

__all__ = [
    "EncodingDetector",
    "FileTypeDetector",
]
