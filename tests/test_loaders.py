"""
Testes unitários para o módulo de loaders.
"""

from pathlib import Path

import pytest

from lambda_rag_lite.loaders import DirectoryLoader, MarkdownLoader, TextLoader


class TestMarkdownLoader:

    def test_init(self):
        """Testa inicialização do MarkdownLoader."""
        loader = MarkdownLoader("test.md")
        assert loader.path == Path("test.md")
        assert loader.encoding == "utf-8"
        assert loader.auto_detect_encoding is True

        # Com parâmetros customizados
        loader = MarkdownLoader(
            "test.md", encoding="latin-1", auto_detect_encoding=False
        )
        assert loader.encoding == "latin-1"
        assert loader.auto_detect_encoding is False

    def test_load_file_not_found(self, mocker):
        """Testa carregamento com arquivo não encontrado."""
        loader = MarkdownLoader("nonexistent.md")

        with pytest.raises(FileNotFoundError, match="Caminho não encontrado"):
            loader.load()

    def test_load_single_markdown_file(self, mocker):
        """Testa carregamento de um único arquivo Markdown."""
        mocker.patch.object(Path, "exists", return_value=True)
        mocker.patch.object(Path, "is_file", return_value=True)
        mocker.patch.object(Path, "read_text", return_value="# Test\nMarkdown content")
        mock_stat = mocker.patch.object(Path, "stat")
        mock_stat.return_value.st_size = 100
        mocker.patch.object(
            Path, "suffix", new_callable=mocker.PropertyMock, return_value=".md"
        )
        mocker.patch.object(Path, "absolute", return_value="/path/test.md")
        mocker.patch.object(
            Path, "name", new_callable=mocker.PropertyMock, return_value="test.md"
        )

        loader = MarkdownLoader("test.md")
        docs = loader.load()

        assert len(docs) == 1
        assert docs[0].page_content == "# Test\nMarkdown content"
        assert "source" in docs[0].metadata
        assert docs[0].metadata["filename"] == "test.md"
        assert docs[0].metadata["file_type"] == "markdown"

    def test_load_directory_with_markdown_files(self, mocker):
        """Testa carregamento de diretório com arquivos Markdown."""
        mocker.patch.object(Path, "exists", return_value=True)
        mocker.patch.object(Path, "is_file", return_value=False)

        # Mock dos arquivos encontrados
        file1 = mocker.Mock()
        file1.read_text.return_value = "Content 1"
        file1.absolute.return_value = "/path/file1.md"
        file1.name = "file1.md"
        file1.stat.return_value.st_size = 100

        file2 = mocker.Mock()
        file2.read_text.return_value = "Content 2"
        file2.absolute.return_value = "/path/file2.md"
        file2.name = "file2.md"
        file2.stat.return_value.st_size = 200

        mocker.patch.object(Path, "rglob", return_value=[file1, file2])

        loader = MarkdownLoader("docs/")
        docs = loader.load()

        assert len(docs) == 2
        assert docs[0].page_content == "Content 1"
        assert docs[1].page_content == "Content 2"
        assert docs[0].metadata["filename"] == "file1.md"
        assert docs[1].metadata["filename"] == "file2.md"

    def test_load_directory_no_markdown_files(self, mocker):
        """Testa carregamento de diretório sem arquivos Markdown."""
        mocker.patch.object(Path, "exists", return_value=True)
        mocker.patch.object(Path, "is_file", return_value=False)
        mocker.patch.object(Path, "rglob", return_value=[])

        loader = MarkdownLoader("empty_docs/")

        with pytest.raises(ValueError, match="Nenhum arquivo Markdown encontrado"):
            loader.load()

    def test_load_with_encoding_error(self, mocker):
        """Testa carregamento com erro de encoding."""
        mocker.patch.object(Path, "exists", return_value=True)
        mocker.patch.object(Path, "is_file", return_value=True)
        mock_stat = mocker.patch.object(Path, "stat")
        mock_stat.return_value = mocker.Mock(st_size=100)
        mocker.patch.object(
            Path,
            "read_text",
            side_effect=[
                UnicodeDecodeError("utf-8", b"", 0, 1, "invalid"),
                "Content with different encoding",
            ],
        )
        mocker.patch.object(
            Path, "suffix", new_callable=mocker.PropertyMock, return_value=".md"
        )
        mocker.patch.object(Path, "absolute", return_value="/path/test.md")
        mocker.patch.object(
            Path, "name", new_callable=mocker.PropertyMock, return_value="test.md"
        )

        loader = MarkdownLoader("test.md")
        docs = loader.load()

        assert len(docs) == 1
        assert docs[0].page_content == "Content with different encoding"

    def test_load_with_encoding_error_no_auto_detect(self, mocker):
        """Testa carregamento com erro de encoding e auto-detect desabilitado."""
        mocker.patch.object(Path, "exists", return_value=True)
        mocker.patch.object(Path, "is_file", return_value=True)
        mocker.patch.object(
            Path,
            "read_text",
            side_effect=UnicodeDecodeError("utf-8", b"", 0, 1, "invalid"),
        )
        mocker.patch.object(
            Path, "suffix", new_callable=mocker.PropertyMock, return_value=".md"
        )
        loader = MarkdownLoader("test.md", auto_detect_encoding=False)
        with pytest.raises(ValueError, match="Nenhum arquivo Markdown encontrado"):
            loader.load()

    def test_load_with_general_exception(self, mocker):
        """Testa carregamento com exceção geral."""
        mocker.patch.object(Path, "exists", return_value=True)
        mocker.patch.object(Path, "is_file", return_value=True)
        mocker.patch.object(Path, "read_text", side_effect=OSError("Permission denied"))
        mocker.patch.object(
            Path, "suffix", new_callable=mocker.PropertyMock, return_value=".md"
        )
        loader = MarkdownLoader("test.md")
        with pytest.raises(ValueError, match="Nenhum arquivo Markdown encontrado"):
            loader.load()

    def test_load_with_all_encodings_fail(self, mocker):
        """Testa carregamento quando todas as codificações falham."""
        mocker.patch.object(Path, "exists", return_value=True)
        mocker.patch.object(Path, "is_file", return_value=True)
        mocker.patch.object(
            Path, "suffix", new_callable=mocker.PropertyMock, return_value=".md"
        )
        mocker.patch.object(Path, "absolute", return_value="/path/test.md")
        mocker.patch.object(
            Path, "name", new_callable=mocker.PropertyMock, return_value="test.md"
        )
        mock_stat = mocker.patch.object(Path, "stat")
        mock_stat.return_value = mocker.Mock(st_size=100)
        mock_read = mocker.patch.object(Path, "read_text")
        mock_read.side_effect = [
            UnicodeDecodeError("utf-8", b"", 0, 1, "invalid"),  # Primeira tentativa
            UnicodeDecodeError("utf-8", b"", 0, 1, "invalid"),  # utf-8
            UnicodeDecodeError("latin-1", b"", 0, 1, "invalid"),  # latin-1
            UnicodeDecodeError("cp1252", b"", 0, 1, "invalid"),  # cp1252
            "conteúdo fallback",  # errors="ignore"
        ]
        loader = MarkdownLoader("test.md", auto_detect_encoding=True)
        docs = loader.load()
        assert len(docs) == 1
        assert docs[0].page_content == "conteúdo fallback"

    def test_load_with_general_exception_in_detect_encoding(self, mocker):
        """Testa exceção geral durante carregamento."""
        mocker.patch.object(Path, "exists", return_value=True)
        mocker.patch.object(Path, "is_file", return_value=True)
        mocker.patch.object(
            Path, "suffix", new_callable=mocker.PropertyMock, return_value=".md"
        )
        mock_read = mocker.patch.object(Path, "read_text")
        mock_read.side_effect = Exception("Erro geral")
        loader = MarkdownLoader("test.md")
        with pytest.raises(ValueError, match="Nenhum arquivo Markdown encontrado"):
            loader.load()


class TestTextLoader:

    def test_init_default_extensions(self):
        """Testa inicialização com extensões padrão."""
        loader = TextLoader("test.txt")
        assert loader.path == Path("test.txt")
        assert ".txt" in loader.extensions
        assert ".py" in loader.extensions
        assert ".md" in loader.extensions

    def test_init_custom_extensions(self):
        """Testa inicialização com extensões customizadas."""
        custom_extensions = [".custom", ".ext"]
        loader = TextLoader("test.txt", extensions=custom_extensions)
        assert loader.extensions == {".custom", ".ext"}

    def test_load_single_file(self, mocker):
        """Testa carregamento de um único arquivo."""
        mocker.patch.object(Path, "exists", return_value=True)
        mocker.patch.object(Path, "is_file", return_value=True)
        mocker.patch.object(Path, "read_text", return_value="Test content")
        mock_stat = mocker.patch.object(Path, "stat")
        mock_stat.return_value.st_size = 50
        mocker.patch.object(
            Path, "suffix", new_callable=mocker.PropertyMock, return_value=".txt"
        )
        mocker.patch.object(Path, "absolute", return_value="/path/test.txt")
        mocker.patch.object(
            Path, "name", new_callable=mocker.PropertyMock, return_value="test.txt"
        )

        loader = TextLoader("test.txt")
        docs = loader.load()

        assert len(docs) == 1
        assert docs[0].page_content == "Test content"
        assert "source" in docs[0].metadata
        assert docs[0].metadata["filename"] == "test.txt"
        assert docs[0].metadata["file_type"] == "text"
        assert docs[0].metadata["extension"] == ".txt"

    def test_load_directory(self, mocker):
        """Testa carregamento de diretório."""
        mocker.patch.object(Path, "exists", return_value=True)
        mocker.patch.object(
            Path, "is_file", side_effect=lambda: False
        )  # Path é diretório

        # Mock dos arquivos encontrados
        file1 = mocker.Mock()
        file1.is_file.return_value = True
        file1.suffix = ".py"
        file1.read_text.return_value = "print('hello')"
        file1.absolute.return_value = "/path/test.py"
        file1.name = "test.py"
        file1.stat.return_value.st_size = 50

        mocker.patch.object(Path, "rglob", return_value=[file1])

        loader = TextLoader("src/")
        docs = loader.load()

        assert len(docs) == 1
        assert docs[0].page_content == "print('hello')"
        assert docs[0].metadata["file_type"] == "python"

    def test_detect_file_type(self):
        """Testa detecção de tipo de arquivo."""
        loader = TextLoader("test.txt")

        # Testa diferentes extensões
        assert loader._detect_file_type(Path("test.py")) == "python"
        assert loader._detect_file_type(Path("test.js")) == "javascript"
        assert loader._detect_file_type(Path("test.html")) == "html"
        assert loader._detect_file_type(Path("test.sql")) == "sql"
        assert loader._detect_file_type(Path("test.unknown")) == "text"

    def test_load_file_not_found(self, mocker):
        """Testa carregamento com arquivo não encontrado."""
        mocker.patch.object(Path, "exists", return_value=False)

        loader = TextLoader("nonexistent.txt")

        with pytest.raises(FileNotFoundError):
            loader.load()


class TestDirectoryLoader:

    def test_init(self):
        """Testa inicialização do DirectoryLoader."""
        loader = DirectoryLoader("docs/")
        assert loader.path == Path("docs/")
        assert loader.recursive is True
        assert loader.show_progress is False

        # Com parâmetros customizados
        loader = DirectoryLoader("docs/", recursive=False, show_progress=True)
        assert loader.recursive is False
        assert loader.show_progress is True

    def test_load_directory_not_found(self, mocker):
        """Testa carregamento com diretório não encontrado."""
        mocker.patch.object(Path, "exists", return_value=False)

        loader = DirectoryLoader("nonexistent/")

        with pytest.raises(FileNotFoundError, match="Diretório não encontrado"):
            loader.load()

    def test_load_not_directory(self, mocker):
        """Testa carregamento com caminho que não é diretório."""
        mocker.patch.object(Path, "exists", return_value=True)
        mocker.patch.object(Path, "is_dir", return_value=False)

        loader = DirectoryLoader("file.txt")

        with pytest.raises(ValueError, match="Caminho não é um diretório"):
            loader.load()

    def test_load_recursive(self, mocker):
        """Testa carregamento recursivo."""
        mocker.patch.object(Path, "exists", return_value=True)
        mocker.patch.object(Path, "is_dir", return_value=True)

        # Mock dos arquivos encontrados
        md_file = mocker.Mock()
        md_file.is_file.return_value = True
        md_file.suffix = ".md"

        py_file = mocker.Mock()
        py_file.is_file.return_value = True
        py_file.suffix = ".py"

        mocker.patch.object(Path, "rglob", return_value=[md_file, py_file])

        # Mock dos loaders
        mock_md_loader = mocker.patch("lambda_rag_lite.loaders.MarkdownLoader")
        mock_txt_loader = mocker.patch("lambda_rag_lite.loaders.TextLoader")
        mock_md_loader.return_value.load.return_value = [
            mocker.Mock(page_content="MD content")
        ]
        mock_txt_loader.return_value.load.return_value = [
            mocker.Mock(page_content="PY content")
        ]

        loader = DirectoryLoader("/some/dir", recursive=True)
        docs = loader.load()
        assert any(doc.page_content == "MD content" for doc in docs)
        assert any(doc.page_content == "PY content" for doc in docs)

    def test_load_non_recursive(self, mocker):
        """Testa carregamento não recursivo."""
        mocker.patch.object(Path, "exists", return_value=True)
        mocker.patch.object(Path, "is_dir", return_value=True)

        md_file = mocker.Mock()
        md_file.is_file.return_value = True
        md_file.suffix = ".md"

        mocker.patch.object(Path, "glob", return_value=[md_file])
        mock_md_loader = mocker.patch("lambda_rag_lite.loaders.MarkdownLoader")
        mock_md_loader.return_value.load.return_value = [
            mocker.Mock(page_content="MD content")
        ]

        loader = DirectoryLoader("/some/dir", recursive=False)
        docs = loader.load()
        assert any(doc.page_content == "MD content" for doc in docs)

    def test_get_loader_for_file(self):
        """Testa seleção de loader para arquivo."""
        loader = DirectoryLoader("docs/")

        # Teste arquivo Markdown
        md_loader = loader._get_loader_for_file(Path("test.md"))
        assert md_loader is not None

        # Teste arquivo Python
        py_loader = loader._get_loader_for_file(Path("test.py"))
        assert py_loader is not None

        # Teste arquivo não suportado
        unsupported_loader = loader._get_loader_for_file(Path("test.bin"))
        assert unsupported_loader is None

    def test_load_with_progress(self, mocker):
        """Testa carregamento com barra de progresso."""
        mocker.patch.object(Path, "exists", return_value=True)
        mocker.patch.object(Path, "is_dir", return_value=True)
        mocker.patch.object(Path, "rglob", return_value=[])

        # Teste sem tqdm (caso padrão)
        loader = DirectoryLoader("docs/", show_progress=True)
        docs = loader.load()

        assert len(docs) == 0

    def test_load_with_progress_no_tqdm(self, mocker):
        """Testa carregamento com progresso mas sem tqdm."""
        mocker.patch.object(Path, "exists", return_value=True)
        mocker.patch.object(Path, "is_dir", return_value=True)
        mocker.patch.object(Path, "rglob", return_value=[])

        # Simula ImportError quando tenta importar tqdm
        with mocker.patch("builtins.__import__", side_effect=ImportError):
            loader = DirectoryLoader("docs/", show_progress=True)
            docs = loader.load()

            assert len(docs) == 0

    def test_load_with_loader_exception(self, mocker):
        """Testa carregamento quando loader gera exceção - deve ignorar arquivo com erro."""
        mocker.patch.object(Path, "exists", return_value=True)
        mocker.patch.object(Path, "is_dir", return_value=True)

        # Mock de arquivo que causará exceção
        bad_file = mocker.Mock()
        bad_file.is_file.return_value = True
        bad_file.suffix = ".md"

        mocker.patch.object(Path, "rglob", return_value=[bad_file])
        mock_md_loader = mocker.patch("lambda_rag_lite.loaders.MarkdownLoader")
        mock_md_loader.return_value.load.side_effect = Exception("Loader error")

        loader = DirectoryLoader("/some/dir", recursive=True)
        # DirectoryLoader ignora arquivos que não consegue carregar, então não deve levantar exceção
        docs = loader.load()
        assert docs == []  # Lista vazia, pois o arquivo com erro foi ignorado
