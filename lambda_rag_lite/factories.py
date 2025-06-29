"""
Factory classes for creating loaders.

"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .loaders import MarkdownLoader, TextLoader

from .config import LoaderConfig
from .detectors.file_type import FileTypeDetector


class LoaderFactory:
    """
    Factory para criar loaders apropriados baseado no tipo de arquivo.

    Centraliza a lógica de seleção de loaders.
    """

    def __init__(self, config: LoaderConfig | None = None):
        """
        Inicializa a factory.

        Args:
            config: Configuração para os loaders
        """
        self.config = config or LoaderConfig()
        self.file_type_detector = FileTypeDetector(self.config)

    def get_loader_for_file(
        self, file_path: Path
    ) -> MarkdownLoader | TextLoader | None:
        """
        Retorna o loader apropriado para um arquivo.

        Args:
            file_path: Caminho do arquivo

        Returns:
            Loader apropriado ou None se não suportado
        """
        if not self.file_type_detector.is_supported_extension(file_path):
            return None

        ext = file_path.suffix.lower()

        # Import here to avoid circular imports
        from .loaders import MarkdownLoader, TextLoader

        # Usa mapeamento direto em vez de múltiplas condições
        if ext in self.config.markdown_extensions:
            return MarkdownLoader(file_path)

        if ext in self.config.text_extensions:
            return TextLoader(file_path)

        return None
