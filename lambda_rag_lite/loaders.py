"""
Módulo de loaders para diferentes tipos de documentos.

Fornece classes para carregar e processar documentos de diferentes formatos,
mantendo compatibilidade com a interface LangChain.
"""

from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import List, Optional, Union

from langchain_core.documents import Document


class MarkdownLoader:
    """
    Loader para arquivos Markdown que busca recursivamente em diretórios.

    Carrega todos os arquivos .md encontrados em um diretório e seus
    subdiretórios, criando Documents LangChain com metadados apropriados.
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
        self.encoding = encoding
        self.auto_detect_encoding = auto_detect_encoding

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
            # Tenta com a codificação especificada
            content = file_path.read_text(encoding=self.encoding)
        except UnicodeDecodeError:
            if self.auto_detect_encoding:
                # Tenta outras codificações comuns
                for encoding in ["utf-8", "latin-1", "cp1252"]:
                    try:
                        content = file_path.read_text(encoding=encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    # Se todas falharam, usa ignore para evitar crash
                    content = file_path.read_text(errors="ignore")
            else:
                return None
        except Exception:
            return None

        # Metadados do arquivo
        metadata = {
            "source": str(file_path.absolute()),
            "filename": file_path.name,
            "file_type": "markdown",
            "file_size": file_path.stat().st_size,
        }

        return Document(page_content=content, metadata=metadata)


class TextLoader:
    """
    Loader genérico para arquivos de texto.

    Carrega arquivos .txt, .md, .py, .js, e outros formatos de texto,
    detectando automaticamente o tipo baseado na extensão.
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
        self.encoding = encoding
        self.auto_detect_encoding = auto_detect_encoding

        # Extensões padrão de texto
        if extensions is None:
            self.extensions = {
                ".txt",
                ".md",
                ".py",
                ".js",
                ".ts",
                ".json",
                ".yaml",
                ".yml",
                ".html",
                ".htm",
                ".css",
                ".sql",
                ".sh",
                ".bash",
                ".zsh",
                ".csv",
                ".log",
                ".conf",
                ".cfg",
                ".ini",
            }
        else:
            self.extensions = {ext.lower() for ext in extensions}

    def load(self) -> List[Document]:
        """
        Carrega arquivos de texto.

        Returns:
            Lista de Documents
        """
        if not self.path.exists():
            raise FileNotFoundError(f"Caminho não encontrado: {self.path}")

        documents = []

        if self.path.is_file():
            if self.path.suffix.lower() in self.extensions:
                doc = self._load_single_file(self.path)
                if doc:
                    documents.append(doc)
        else:
            # Busca recursiva
            for file_path in self.path.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in self.extensions:
                    doc = self._load_single_file(file_path)
                    if doc:
                        documents.append(doc)

        return documents

    def _load_single_file(self, file_path: Path) -> Optional[Document]:
        """Carrega um único arquivo de texto."""
        try:
            content = file_path.read_text(encoding=self.encoding)
        except UnicodeDecodeError:
            if self.auto_detect_encoding:
                for encoding in ["utf-8", "latin-1", "cp1252"]:
                    try:
                        content = file_path.read_text(encoding=encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    content = file_path.read_text(errors="ignore")
            else:
                return None
        except Exception:
            return None

        # Detecta tipo de arquivo
        file_type = self._detect_file_type(file_path)

        metadata = {
            "source": str(file_path.absolute()),
            "filename": file_path.name,
            "file_type": file_type,
            "file_size": file_path.stat().st_size,
            "extension": file_path.suffix.lower(),
        }

        return Document(page_content=content, metadata=metadata)

    def _detect_file_type(self, file_path: Path) -> str:
        """
        Detecta o tipo de arquivo baseado na extensão.

        Args:
            file_path: Caminho do arquivo

        Returns:
            String descrevendo o tipo de arquivo
        """
        ext = file_path.suffix.lower()

        type_mapping = {
            ".md": "markdown",
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".html": "html",
            ".htm": "html",
            ".css": "css",
            ".json": "json",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".sql": "sql",
            ".sh": "shell",
            ".bash": "shell",
            ".zsh": "shell",
            ".csv": "csv",
            ".log": "log",
            ".txt": "text",
        }

        return type_mapping.get(ext, "text")


class DirectoryLoader:
    """
    Loader que combina múltiplos loaders para diferentes tipos de arquivo.

    Carrega automaticamente diferentes tipos de arquivo de um diretório,
    aplicando o loader apropriado baseado na extensão.
    """

    def __init__(
        self,
        path: Union[str, Path],
        recursive: bool = True,
        show_progress: bool = False,
    ):
        """
        Inicializa o loader de diretório.

        Args:
            path: Caminho do diretório
            recursive: Se deve buscar recursivamente
            show_progress: Se deve mostrar progresso (requer tqdm)
        """
        self.path = Path(path)
        self.recursive = recursive
        self.show_progress = show_progress

    def load(self) -> List[Document]:
        """
        Carrega todos os arquivos suportados do diretório.

        Returns:
            Lista de Documents de todos os arquivos
        """
        if not self.path.exists():
            raise FileNotFoundError(f"Diretório não encontrado: {self.path}")

        if not self.path.is_dir():
            raise ValueError(f"Caminho não é um diretório: {self.path}")

        documents = []

        # Busca arquivos
        if self.recursive:
            files = list(self.path.rglob("*"))
        else:
            files = list(self.path.glob("*"))

        # Filtra apenas arquivos
        files = [f for f in files if f.is_file()]

        # Aplica loader apropriado
        if self.show_progress:
            try:
                from tqdm import tqdm

                files = tqdm(files, desc="Carregando arquivos")
            except ImportError:
                pass  # tqdm não disponível

        for file_path in files:
            try:
                loader = self._get_loader_for_file(file_path)
                if loader:
                    docs = loader.load()
                    documents.extend(docs)
            except Exception:
                continue  # Ignora arquivos que não conseguiu carregar

        return documents

    def _get_loader_for_file(
        self, file_path: Path
    ) -> Optional[Union[MarkdownLoader, TextLoader]]:
        """
        Retorna o loader apropriado para um arquivo.

        Args:
            file_path: Caminho do arquivo

        Returns:
            Loader apropriado ou None se não suportado
        """
        ext = file_path.suffix.lower()

        if ext == ".md":
            return MarkdownLoader(file_path)
        elif ext in {
            ".txt",
            ".py",
            ".js",
            ".ts",
            ".json",
            ".yaml",
            ".yml",
            ".html",
            ".htm",
            ".css",
            ".sql",
            ".sh",
            ".bash",
            ".zsh",
            ".csv",
            ".log",
            ".conf",
            ".cfg",
            ".ini",
        }:
            return TextLoader(file_path)

        return None
