"""
Módulo de loaders para diferentes tipos de documentos.

"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from typing import List, Optional, Union

from langchain_core.documents import Document

from .config import LoaderConfig
from .detectors.encoding import EncodingDetector
from .detectors.file_type import FileTypeDetector
from .factories import LoaderFactory
from .metadata.document_metadata import DocumentMetadataManager


class MarkdownLoader:
    """
    Loader para arquivos Markdown que busca recursivamente em diretórios.

    """

    def __init__(
        self,
        path: Union[str, Path],
        encoding: str = "utf-8",
        auto_detect_encoding: bool = True,
    ):
        """
        Inicializa o loader de Markdown.

        Args:
            path: Caminho para arquivo ou diretório
            encoding: Codificação padrão para leitura de arquivos
            auto_detect_encoding: Se deve tentar detectar encoding automaticamente
        """
        self.path = Path(path)

        # Cria configuração e componentes especializados
        self.config = LoaderConfig(
            encoding=encoding, auto_detect_encoding=auto_detect_encoding
        )
        self.encoding_detector = EncodingDetector(self.config)
        self.metadata_manager = DocumentMetadataManager(self.config)

    @property
    def encoding(self) -> str:
        """Retorna a codificação configurada."""
        return self.config.encoding

    @property
    def auto_detect_encoding(self) -> bool:
        """Retorna se a detecção automática de encoding está habilitada."""
        return self.config.auto_detect_encoding

    def load(self) -> List[Document]:
        """
        Carrega documentos Markdown.

        Returns:
            Lista de Documents LangChain

        Raises:
            FileNotFoundError: Se o caminho não existe
            ValueError: Se não encontrar arquivos Markdown
        """
        if not self.path.exists():
            raise FileNotFoundError(f"Caminho não encontrado: {self.path}")

        documents = []

        if self.path.is_file():
            if self.path.suffix.lower() == ".md":
                doc = self._load_single_file(self.path)
                if doc:
                    documents.append(doc)
        else:
            # Busca recursiva por arquivos .md
            for md_file in self.path.rglob("*.md"):
                doc = self._load_single_file(md_file)
                if doc:
                    documents.append(doc)

        if not documents:
            raise ValueError(f"Nenhum arquivo Markdown encontrado em: {self.path}")

        return documents

    def _load_single_file(self, file_path: Path) -> Optional[Document]:
        """
        Carrega um único arquivo Markdown.


        Args:
            file_path: Caminho para o arquivo

        Returns:
            Document ou None se houve erro
        """
        try:
            # Tenta usar o detector de encoding especializado
            try:
                content, encoding_used = self.encoding_detector.detect_and_read(
                    file_path
                )
            except Exception:
                # Fallback para leitura direta (útil para testes com mocks)
                content = file_path.read_text(encoding=self.config.encoding)
                encoding_used = self.config.encoding

            # Cria metadados básicos
            metadata = {
                "source": str(file_path.absolute()),
                "filename": file_path.name,
                "file_type": "markdown",
                "file_size": file_path.stat().st_size,
                "encoding": encoding_used,
            }

            return Document(page_content=content, metadata=metadata)

        except Exception:
            # Log error silently and continue
            return None


class TextLoader:
    """
    Loader genérico para arquivos de texto.

    """

    def __init__(
        self,
        path: Union[str, Path],
        encoding: str = "utf-8",
        auto_detect_encoding: bool = True,
        extensions: Optional[List[str]] = None,
    ):
        """
        Inicializa o loader de texto.

        Args:
            path: Caminho para arquivo ou diretório
            encoding: Codificação padrão
            auto_detect_encoding: Se deve detectar encoding automaticamente
            extensions: Lista de extensões para carregar (padrão: texto comum)
        """
        self.path = Path(path)

        # Cria configuração personalizada
        config = LoaderConfig(
            encoding=encoding, auto_detect_encoding=auto_detect_encoding
        )

        # Personaliza extensões se fornecidas
        if extensions is not None:
            config.text_extensions = {ext.lower() for ext in extensions}

        # Inicializa componentes especializados
        self.config = config
        self.encoding_detector = EncodingDetector(config)
        self.file_type_detector = FileTypeDetector(config)
        self.metadata_manager = DocumentMetadataManager(config, self.file_type_detector)

    @property
    def extensions(self) -> set[str]:
        """Retorna o conjunto de extensões suportadas."""
        return self.config.text_extensions

    def _detect_file_type(self, file_path: Path) -> str:
        """Detecta o tipo do arquivo baseado na extensão."""
        return self.file_type_detector.detect_file_type(file_path, fallback="text")

    def load(self) -> List[Document]:
        """
        Carrega arquivos de texto.

        Returns:
            Lista de Documents
        """
        if not self.path.exists():
            raise FileNotFoundError(f"Caminho não encontrado: {self.path}")

        return self._load_files()

    def _load_files(self) -> list[Document]:
        """Carrega arquivos baseado no tipo de caminho."""
        if self.path.is_file():
            return self._load_single_path()
        else:
            return self._load_directory_files()

    def _load_single_path(self) -> list[Document]:
        """Carrega um único arquivo."""
        if self.file_type_detector.is_supported_extension(self.path):
            doc = self._load_single_file(self.path)
            return [doc] if doc else []
        return []

    def _load_directory_files(self) -> list[Document]:
        """Carrega arquivos de um diretório."""
        documents = []
        for file_path in self.path.rglob("*"):
            if self._should_load_file(file_path):
                doc = self._load_single_file(file_path)
                if doc:
                    documents.append(doc)
        return documents

    def _should_load_file(self, file_path: Path) -> bool:
        """Verifica se um arquivo deve ser carregado."""
        return file_path.is_file() and self.file_type_detector.is_supported_extension(
            file_path
        )

    def _load_single_file(self, file_path: Path) -> Document | None:
        """
        Carrega um único arquivo de texto.

        """
        try:
            # Tenta usar o detector de encoding especializado
            try:
                content, encoding_used = self.encoding_detector.detect_and_read(
                    file_path
                )
            except Exception:
                # Fallback para leitura direta (útil para testes com mocks)
                content = file_path.read_text(encoding=self.config.encoding)
                encoding_used = self.config.encoding

            # Cria metadados básicos
            metadata = {
                "source": str(file_path.absolute()),
                "filename": file_path.name,
                "file_type": self._detect_file_type(file_path),
                "file_size": file_path.stat().st_size,
                "encoding": encoding_used,
                "extension": file_path.suffix.lower(),
            }

            return Document(page_content=content, metadata=metadata)

        except Exception:
            return None


class DirectoryLoader:
    """
    Loader que combina múltiplos loaders para diferentes tipos de arquivo.

    """

    def __init__(
        self,
        path: Union[str, Path],
        recursive: bool = True,
        show_progress: bool = False,
        config: LoaderConfig | None = None,
    ):
        """
        Inicializa o loader de diretório.

        Args:
            path: Caminho do diretório
            recursive: Se deve buscar recursivamente
            show_progress: Se deve mostrar progresso
            config: Configuração personalizada
        """
        self.path = Path(path)
        self.recursive = recursive
        self.show_progress = show_progress
        self.config = config or LoaderConfig()

        # Inicializa factory de loaders
        self.loader_factory = LoaderFactory(self.config)

    def load(self) -> List[Document]:
        """
        Carrega todos os arquivos suportados do diretório.

        Returns:
            Lista de Documents de todos os arquivos
        """
        self._validate_path()

        files = self._get_files()
        if self.show_progress and self._has_progress_library():
            files = self._wrap_with_progress(files)

        return self._process_files(files)

    def _validate_path(self) -> None:
        """Valida se o caminho existe e é um diretório."""
        if not self.path.exists():
            raise FileNotFoundError(f"Diretório não encontrado: {self.path}")

        if not self.path.is_dir():
            raise ValueError(f"Caminho não é um diretório: {self.path}")

    def _process_files(self, files) -> list[Document]:
        """Processa lista de arquivos e retorna documentos."""
        documents = []
        for file_path in files:
            try:
                loader = self.loader_factory.get_loader_for_file(file_path)
                if loader:
                    docs = loader.load()
                    documents.extend(docs)
            except Exception as e:
                # Log the error for debugging and continue with next file
                logging.debug(f"Failed to load file {file_path}: {e}")
                continue

        return documents

    def _get_loader_for_file(self, file_path: Path):
        """Obtém o loader apropriado para um arquivo."""
        return self.loader_factory.get_loader_for_file(file_path)

    def _get_files(self) -> list[Path]:
        """Obtém lista de arquivos para processar."""
        if self.recursive:
            files = list(self.path.rglob("*"))
        else:
            files = list(self.path.glob("*"))

        # Filtra apenas arquivos
        return [f for f in files if f.is_file()]

    def _has_progress_library(self) -> bool:
        """Verifica se biblioteca de progresso está disponível."""
        return importlib.util.find_spec("tqdm") is not None

    def _wrap_with_progress(self, files: list[Path]):
        """Envolve lista com barra de progresso se disponível."""
        if self._has_progress_library():
            try:
                from tqdm import tqdm

                return tqdm(files, desc="Processando arquivos")
            except ImportError:
                pass
        return files
