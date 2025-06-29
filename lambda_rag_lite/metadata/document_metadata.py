"""
Gerenciador de metadados de documentos.

Centraliza a criação e manipulação de metadados para documentos
carregados pelos loaders.
"""

from __future__ import annotations

import mimetypes
from datetime import datetime
from pathlib import Path
from typing import Any

from ..config import LoaderConfig
from ..detectors.file_type import FileTypeDetector
from ..text_cleaning import format_file_size


class DocumentMetadataManager:
    """
    Gerenciador centralizado de metadados de documentos.

    Extrai e organiza metadados de arquivos de forma consistente
    para todos os loaders.
    """

    def __init__(
        self,
        config: LoaderConfig | None = None,
        file_type_detector: FileTypeDetector | None = None,
    ):
        """
        Inicializa o gerenciador.

        Args:
            config: Configuração do loader
            file_type_detector: Detector de tipo de arquivo
        """
        self.config = config or LoaderConfig()
        self.file_type_detector = file_type_detector or FileTypeDetector(config)

    def create_file_metadata(
        self,
        file_path: Path,
        encoding_used: str | None = None,
        additional_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Cria metadados completos para um arquivo.

        Args:
            file_path: Caminho para o arquivo
            encoding_used: Encoding usado para ler o arquivo
            additional_metadata: Metadados adicionais específicos

        Returns:
            Dicionário com metadados do arquivo
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

        # Metadados básicos sempre incluídos
        metadata: dict[str, Any] = {
            "source": str(file_path.absolute()),
            "filename": file_path.name,
            "file_stem": file_path.stem,
            "file_extension": file_path.suffix.lower(),
        }

        if self.config.include_file_metadata:
            # Adiciona metadados detalhados do arquivo
            stat_info = file_path.stat()

            metadata.update(
                {
                    "file_size": stat_info.st_size,
                    "file_size_human": format_file_size(stat_info.st_size),
                    "created_time": datetime.fromtimestamp(
                        stat_info.st_ctime
                    ).isoformat(),
                    "modified_time": datetime.fromtimestamp(
                        stat_info.st_mtime
                    ).isoformat(),
                    "file_type": self.file_type_detector.detect_file_type(file_path),
                    "file_category": self.file_type_detector.get_file_category(
                        file_path
                    ),
                }
            )

            # Adiciona MIME type se disponível
            mime_type = self._get_mime_type(file_path)
            if mime_type:
                metadata["mime_type"] = mime_type

            # Adiciona encoding usado
            if encoding_used:
                metadata["encoding"] = encoding_used

        # Adiciona metadados específicos do caminho
        metadata.update(self._extract_path_metadata(file_path))

        # Merge com metadados adicionais
        if additional_metadata:
            metadata.update(additional_metadata)

        return metadata

    def create_directory_metadata(
        self, directory_path: Path, additional_metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Cria metadados para um diretório.

        Args:
            directory_path: Caminho para o diretório
            additional_metadata: Metadados adicionais

        Returns:
            Dicionário com metadados do diretório
        """
        if not directory_path.exists():
            raise FileNotFoundError(f"Diretório não encontrado: {directory_path}")

        metadata = {
            "source": str(directory_path.absolute()),
            "directory_name": directory_path.name,
            "is_directory": True,
        }

        if self.config.include_file_metadata:
            stat_info = directory_path.stat()
            metadata.update(
                {
                    "created_time": datetime.fromtimestamp(
                        stat_info.st_ctime
                    ).isoformat(),
                    "modified_time": datetime.fromtimestamp(
                        stat_info.st_mtime
                    ).isoformat(),
                }
            )

        # Adiciona metadados específicos do caminho
        metadata.update(self._extract_path_metadata(directory_path))

        if additional_metadata:
            metadata.update(additional_metadata)

        return metadata

    def enhance_metadata_with_content_analysis(
        self,
        metadata: dict[str, Any],
        content: str,
        analysis_results: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Enriquece metadados com análise do conteúdo.

        Args:
            metadata: Metadados existentes
            content: Conteúdo do documento
            analysis_results: Resultados de análise de texto

        Returns:
            Metadados enriquecidos
        """
        enhanced_metadata = metadata.copy()

        # Adiciona estatísticas básicas do conteúdo
        if content:
            enhanced_metadata.update(
                {
                    "content_length": len(content),
                    "content_lines": len(content.split("\n")),
                    "content_words": len(content.split()),
                    "content_paragraphs": len(
                        [p for p in content.split("\n\n") if p.strip()]
                    ),
                }
            )

        # Adiciona resultados de análise se fornecidos
        if analysis_results:
            enhanced_metadata.update(analysis_results)

        return enhanced_metadata

    def _extract_path_metadata(self, path: Path) -> dict[str, Any]:
        """
        Extrai metadados do caminho do arquivo.

        Args:
            path: Caminho do arquivo/diretório

        Returns:
            Dicionário com metadados do caminho
        """
        metadata: dict[str, Any] = {}

        # Adiciona informações do diretório pai
        if path.parent != path:
            metadata["parent_directory"] = path.parent.name
            metadata["relative_path"] = str(path.relative_to(path.anchor))

        # Conta depth do diretório
        parts = path.parts
        metadata["path_depth"] = len(parts) - 1  # -1 para não contar a raiz

        # Adiciona partes do caminho como tags
        if len(parts) > 1:
            metadata["path_parts"] = list(parts[1:])  # Remove a raiz

        return metadata

    def _get_mime_type(self, file_path: Path) -> str | None:
        """Obtém MIME type do arquivo."""
        try:
            mime_type, _ = mimetypes.guess_type(str(file_path))
            return mime_type
        except Exception:
            return None
