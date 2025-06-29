"""
Detector de tipo de arquivo.

"""

from __future__ import annotations

import mimetypes
from pathlib import Path

from ..config import LoaderConfig
from ..constants import (
    CODE_TYPES,
    CONFIG_TYPES,
    DATA_TYPES,
    DOCUMENT_TYPES,
    EXTENSION_TYPE_MAPPING,
    MIME_TYPE_MAPPING,
    SPECIAL_FILES,
)


class FileTypeDetector:
    """
    Detector de tipo de arquivo baseado em extensão e MIME type.

    Centraliza a lógica de detecção de tipos de arquivo que estava
    espalhada pelos loaders.
    """

    def __init__(self, config: LoaderConfig | None = None):
        """
        Inicializa o detector.

        Args:
            config: Configuração do loader (usa padrão se None)
        """
        self.config = config or LoaderConfig()
        # Usa mapeamento das constantes
        self.type_mapping = EXTENSION_TYPE_MAPPING

    def detect_file_type(self, file_path: Path, fallback: str = "unknown") -> str:
        """
        Detecta o tipo de um arquivo.

        Args:
            file_path: Caminho para o arquivo
            fallback: Tipo a retornar quando não consegue detectar (padrão: "unknown")

        Returns:
            String descrevendo o tipo do arquivo
        """
        # Verifica nome especial do arquivo
        filename_lower = file_path.name.lower()
        for special, type_name in SPECIAL_FILES.items():
            if special in filename_lower:
                return type_name

        # Detecção por extensão
        ext = file_path.suffix.lower()
        if ext in self.type_mapping:
            return self.type_mapping[ext]

        # Se arquivo existe, tenta detecção por MIME type
        if file_path.exists():
            mime_type = self._get_mime_type(file_path)
            if mime_type:
                mime_type_result = self._mime_to_type(mime_type)
                if mime_type_result != "unknown":
                    return mime_type_result

            # Verifica se é arquivo de texto
            return "text" if self._is_text_file(file_path) else fallback

        # Se arquivo não existe, retorna o fallback
        return fallback

    def is_supported_extension(self, file_path: Path) -> bool:
        """
        Verifica se a extensão do arquivo é suportada.

        Args:
            file_path: Caminho para o arquivo

        Returns:
            True se a extensão é suportada
        """
        ext = file_path.suffix.lower()
        return (
            ext in self.config.text_extensions or ext in self.config.markdown_extensions
        )

    def is_text_file(self, file_path: Path) -> bool:
        """
        Verifica se um arquivo é de texto.

        Args:
            file_path: Caminho para o arquivo

        Returns:
            True se o arquivo é de texto
        """
        return self._is_text_file(file_path)

    def _get_mime_type(self, file_path: Path) -> str | None:
        """Obtém MIME type do arquivo."""
        try:
            mime_type, _ = mimetypes.guess_type(str(file_path))
            return mime_type
        except Exception:
            return None

    def _mime_to_type(self, mime_type: str) -> str:
        """Converte MIME type para tipo interno."""
        return MIME_TYPE_MAPPING.get(
            mime_type, "text" if mime_type.startswith("text/") else "unknown"
        )

    def _is_text_file(self, file_path: Path) -> bool:
        """
        Verifica se arquivo é de texto lendo uma amostra.

        Args:
            file_path: Caminho para o arquivo

        Returns:
            True se parece ser arquivo de texto
        """
        try:
            # Lê primeiros 512 bytes
            with open(file_path, "rb") as f:
                sample = f.read(512)

            if not sample:
                return True  # Arquivo vazio é considerado texto

            # Verifica se contém muitos bytes não-texto
            non_text_chars = sum(
                1 for byte in sample if byte < 32 and byte not in (9, 10, 13)
            )

            # Se mais de 10% são caracteres não-texto, provavelmente é binário
            return (non_text_chars / len(sample)) < 0.1

        except Exception:
            return False

    def get_file_category(self, file_path: Path) -> str:
        """
        Retorna categoria geral do arquivo.

        Args:
            file_path: Caminho para o arquivo

        Returns:
            Categoria do arquivo (code, document, data, config, etc.)
        """
        file_type = self.detect_file_type(file_path)

        if file_type in CODE_TYPES:
            return "code"
        elif file_type in DOCUMENT_TYPES:
            return "document"
        elif file_type in DATA_TYPES:
            return "data"
        elif file_type in CONFIG_TYPES:
            return "config"
        else:
            return "other"
