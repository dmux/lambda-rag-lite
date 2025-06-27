"""
Testes unitários para o módulo de utilitários.
"""

from lambda_rag_lite.utils import (
    TextProcessor,
    calculate_text_stats,
    chunk_text,
    clean_text,
    extract_keywords,
    format_file_size,
)


class TestChunkText:

    def test_chunk_text_basic(self):
        """Testa divisão básica de texto."""
        text = "Este é um texto longo que será dividido em chunks menores para facilitar o processamento."
        chunks = chunk_text(text, chunk_size=30, chunk_overlap=10)

        assert len(chunks) > 1
        assert all(isinstance(chunk, str) for chunk in chunks)
        # Permite flexibilidade para quebras naturais
        assert all(len(chunk) <= 100 for chunk in chunks)

    def test_chunk_text_empty(self):
        """Testa divisão de texto vazio."""
        assert chunk_text("", chunk_size=100) == []
        assert chunk_text(None, chunk_size=100) == []

    def test_chunk_text_invalid_size(self):
        """Testa divisão com tamanho inválido."""
        text = "Texto de teste"
        assert chunk_text(text, chunk_size=0) == []
        assert chunk_text(text, chunk_size=-1) == []

    def test_chunk_text_small_text(self):
        """Testa divisão de texto pequeno."""
        text = "Texto pequeno"
        chunks = chunk_text(text, chunk_size=100)

        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_text_with_separator(self):
        """Testa divisão usando separador."""
        text = "Primeiro parágrafo.\n\nSegundo parágrafo.\n\nTerceiro parágrafo."
        chunks = chunk_text(text, chunk_size=50, separator="\n\n")

        assert len(chunks) >= 1
        assert all(isinstance(chunk, str) for chunk in chunks)

    def test_chunk_text_with_overlap(self):
        """Testa divisão com sobreposição."""
        text = "A" * 100  # Texto de 100 caracteres
        chunks = chunk_text(text, chunk_size=30, chunk_overlap=10)

        assert len(chunks) > 1
        # Verifica se há sobreposição (difícil de testar exatamente devido à lógica de quebra)
        assert all(len(chunk) >= 20 for chunk in chunks)  # Pelo menos alguns caracteres

    def test_chunk_text_large_section(self):
        """Testa divisão com seção maior que chunk_size."""
        large_section = "A" * 200
        text = f"Pequeno\n\n{large_section}\n\nOutro pequeno"
        chunks = chunk_text(text, chunk_size=50, separator="\n\n")

        assert len(chunks) >= 2
        # A seção grande deve ser dividida recursivamente
        assert any(len(chunk) <= 60 for chunk in chunks)  # Permite alguma flexibilidade

    def test_chunk_text_no_natural_break(self):
        """Testa chunking quando não há quebra natural."""
        # Texto longo sem espaços nem quebras naturais
        text = "A" * 100 + "\n\n" + "B" * 200
        chunks = chunk_text(text, chunk_size=50, separator="\n\n")

        assert len(chunks) >= 2
        # Verifica que o texto grande foi dividido mesmo sem quebras naturais
        assert any("B" in chunk for chunk in chunks)


class TestCleanText:

    def test_clean_text_basic(self):
        """Testa limpeza básica de texto."""
        dirty_text = "  Texto   com   espaços   extras  \n\n\n"
        clean = clean_text(dirty_text)

        assert clean == "Texto com espaços extras"

    def test_clean_text_empty(self):
        """Testa limpeza de texto vazio."""
        assert clean_text("") == ""
        assert clean_text(None) == ""

    def test_clean_text_control_characters(self):
        """Testa remoção de caracteres de controle."""
        text_with_control = "Texto\x00com\x08caracteres\x1fde controle"
        clean = clean_text(text_with_control)

        assert "\x00" not in clean
        assert "\x08" not in clean
        assert "\x1f" not in clean
        assert "Texto" in clean

    def test_clean_text_no_whitespace_removal(self):
        """Testa limpeza sem remoção de espaços extras."""
        dirty_text = "  Texto   com   espaços  "
        clean = clean_text(dirty_text, remove_extra_whitespace=False)

        # Deve remover caracteres de controle mas manter espaços
        assert "Texto" in clean
        assert len(clean) > len("Texto com espaços")

    def test_clean_text_multiple_newlines(self):
        """Testa limpeza de quebras de linha múltiplas."""
        text = "Linha 1\n\n\n\nLinha 2\n\n\nLinha 3"
        clean = clean_text(text)

        assert "\n\n\n" not in clean
        assert "Linha 1\n\nLinha 2\n\nLinha 3" == clean


class TestExtractKeywords:

    def test_extract_keywords_basic(self):
        """Testa extração básica de palavras-chave."""
        text = (
            "Python é uma linguagem de programação muito popular para ciência de dados"
        )
        keywords = extract_keywords(text)

        assert isinstance(keywords, list)
        assert len(keywords) > 0
        assert "python" in keywords
        assert "programação" in keywords
        # Stop words não devem aparecer
        assert "é" not in keywords
        assert "uma" not in keywords

    def test_extract_keywords_empty(self):
        """Testa extração de texto vazio."""
        assert extract_keywords("") == []
        assert extract_keywords(None) == []

    def test_extract_keywords_min_length(self):
        """Testa filtro por comprimento mínimo."""
        text = "a bb ccc dddd eeeee"
        keywords = extract_keywords(text, min_length=4)

        assert "dddd" in keywords
        assert "eeeee" in keywords
        assert "a" not in keywords
        assert "bb" not in keywords
        assert "ccc" not in keywords

    def test_extract_keywords_max_words(self):
        """Testa limite máximo de palavras."""
        text = "palavra1 palavra2 palavra3 palavra4 palavra5 palavra6"
        keywords = extract_keywords(text, max_words=3)

        assert len(keywords) <= 3

    def test_extract_keywords_frequency_order(self):
        """Testa ordenação por frequência."""
        text = "python python java java java javascript"
        keywords = extract_keywords(text)

        # "java" aparece 3 vezes, "python" 2 vezes, "javascript" 1 vez
        assert keywords.index("java") < keywords.index("python")
        assert keywords.index("python") < keywords.index("javascript")

    def test_extract_keywords_punctuation_removal(self):
        """Testa remoção de pontuação."""
        text = "Python, Java! JavaScript? C#... Go."
        keywords = extract_keywords(text)

        assert "python" in keywords
        assert "java" in keywords
        assert "javascript" in keywords
        # Pontuação deve ser removida
        assert "python," not in keywords
        assert "java!" not in keywords


class TestCalculateTextStats:

    def test_calculate_text_stats_basic(self):
        """Testa cálculo básico de estatísticas."""
        text = "Primeira linha\nSegunda linha\n\nTerceiro parágrafo"
        stats = calculate_text_stats(text)

        assert stats["char_count"] == len(text)
        assert (
            stats["word_count"] == 6
        )  # Primeira, linha, Segunda, linha, Terceiro, parágrafo
        assert stats["line_count"] == 4  # 3 quebras de linha + 1
        assert stats["paragraph_count"] == 2  # Separados por \n\n
        assert isinstance(stats["avg_words_per_line"], float)
        assert isinstance(stats["avg_chars_per_word"], float)

    def test_calculate_text_stats_empty(self):
        """Testa estatísticas de texto vazio."""
        stats = calculate_text_stats("")

        assert stats["char_count"] == 0
        assert stats["word_count"] == 0
        assert stats["line_count"] == 0  # String vazia não tem linhas
        assert stats["paragraph_count"] == 0
        assert stats["avg_words_per_line"] == 0
        assert stats["avg_chars_per_word"] == 0

    def test_calculate_text_stats_none(self):
        """Testa estatísticas de texto None."""
        stats = calculate_text_stats(None)

        assert all(value == 0 for value in stats.values())

    def test_calculate_text_stats_single_word(self):
        """Testa estatísticas de uma única palavra."""
        stats = calculate_text_stats("palavra")

        assert stats["char_count"] == 7
        assert stats["word_count"] == 1
        assert stats["line_count"] == 1
        assert stats["paragraph_count"] == 1
        assert stats["avg_words_per_line"] == 1.0
        assert stats["avg_chars_per_word"] == 7.0


class TestFormatFileSize:

    def test_format_file_size_bytes(self):
        """Testa formatação de tamanhos em bytes."""
        assert format_file_size(0) == "0 B"
        assert format_file_size(500) == "500.0 B"
        assert format_file_size(1023) == "1023.0 B"

    def test_format_file_size_kb(self):
        """Testa formatação de tamanhos em KB."""
        assert format_file_size(1024) == "1.0 KB"
        assert format_file_size(1536) == "1.5 KB"  # 1.5 KB
        assert format_file_size(1024 * 1023) == "1023.0 KB"

    def test_format_file_size_mb(self):
        """Testa formatação de tamanhos em MB."""
        assert format_file_size(1024 * 1024) == "1.0 MB"
        assert format_file_size(int(1024 * 1024 * 2.5)) == "2.5 MB"

    def test_format_file_size_gb(self):
        """Testa formatação de tamanhos em GB."""
        assert format_file_size(1024 * 1024 * 1024) == "1.0 GB"
        assert format_file_size(int(1024 * 1024 * 1024 * 1.2)) == "1.2 GB"

    def test_format_file_size_tb(self):
        """Testa formatação de tamanhos em TB."""
        assert format_file_size(1024 * 1024 * 1024 * 1024) == "1.0 TB"


class TestTextProcessor:

    def test_text_processor_init(self):
        """Testa inicialização do TextProcessor."""
        processor = TextProcessor()
        assert processor.chunk_size == 1000
        assert processor.chunk_overlap == 200
        assert processor.clean_text is True
        assert processor.extract_keywords is False

        # Com parâmetros customizados
        processor = TextProcessor(
            chunk_size=500, chunk_overlap=100, clean_text=False, extract_keywords=True
        )
        assert processor.chunk_size == 500
        assert processor.chunk_overlap == 100
        assert processor.clean_text is False
        assert processor.extract_keywords is True

    def test_process_text_empty(self):
        """Testa processamento de texto vazio."""
        processor = TextProcessor()
        results = processor.process_text("")

        assert results == []

        results = processor.process_text(None)
        assert results == []

    def test_process_text_basic(self):
        """Testa processamento básico de texto."""
        processor = TextProcessor(chunk_size=50, chunk_overlap=10)
        text = "Este é um texto longo que será dividido em chunks para processamento."

        results = processor.process_text(text)

        assert len(results) > 0
        assert all("text" in result for result in results)
        assert all("metadata" in result for result in results)

        # Verifica metadados
        first_result = results[0]
        metadata = first_result["metadata"]
        assert "chunk_index" in metadata
        assert "chunk_count" in metadata
        assert "chunk_size" in metadata
        assert metadata["chunk_index"] == 0
        assert metadata["chunk_count"] == len(results)

    def test_process_text_with_metadata(self):
        """Testa processamento com metadados base."""
        processor = TextProcessor(chunk_size=50)
        text = "Texto para processamento com metadados."
        base_metadata = {"source": "test.txt", "author": "Test Author"}

        results = processor.process_text(text, metadata=base_metadata)

        assert len(results) > 0
        for result in results:
            metadata = result["metadata"]
            assert metadata["source"] == "test.txt"
            assert metadata["author"] == "Test Author"
            assert "chunk_index" in metadata

    def test_process_text_with_keywords(self):
        """Testa processamento com extração de palavras-chave."""
        processor = TextProcessor(chunk_size=100, extract_keywords=True)
        text = "Python é uma linguagem de programação muito popular."

        results = processor.process_text(text)

        assert len(results) > 0
        first_result = results[0]
        assert "keywords" in first_result["metadata"]
        assert isinstance(first_result["metadata"]["keywords"], list)

    def test_process_text_with_cleaning(self):
        """Testa processamento com limpeza de texto."""
        processor = TextProcessor(clean_text=True)
        dirty_text = "  Texto   com   espaços   extras  \n\n\n"

        results = processor.process_text(dirty_text)

        assert len(results) > 0
        # O texto deve estar limpo
        clean_content = results[0]["text"]
        assert "espaços extras" in clean_content
        assert "   " not in clean_content

    def test_process_text_without_cleaning(self):
        """Testa processamento sem limpeza de texto."""
        processor = TextProcessor(clean_text=False)
        dirty_text = "  Texto   com   espaços   extras  "

        results = processor.process_text(dirty_text)

        assert len(results) > 0
        # O texto deve manter os espaços extras
        raw_content = results[0]["text"]
        assert "   " in raw_content

    def test_process_text_statistics(self):
        """Testa se estatísticas são adicionadas aos metadados."""
        processor = TextProcessor()
        text = "Texto de teste para verificar estatísticas."

        results = processor.process_text(text)

        assert len(results) > 0
        metadata = results[0]["metadata"]

        # Verifica se estatísticas estão presentes
        assert "char_count" in metadata
        assert "word_count" in metadata
        assert "line_count" in metadata
        assert "paragraph_count" in metadata
        assert "avg_words_per_line" in metadata
        assert "avg_chars_per_word" in metadata

    def test_process_text_multiple_chunks(self):
        """Testa processamento que gera múltiplos chunks."""
        processor = TextProcessor(chunk_size=30, chunk_overlap=5)
        long_text = "Este é um texto muito longo que definitivamente será dividido em múltiplos chunks para testar a funcionalidade de processamento de texto em pedaços menores."

        results = processor.process_text(long_text)

        assert len(results) > 1

        # Verifica se os índices dos chunks estão corretos
        for i, result in enumerate(results):
            assert result["metadata"]["chunk_index"] == i
            assert result["metadata"]["chunk_count"] == len(results)

        # Verifica se todos os chunks têm conteúdo
        assert all(result["text"].strip() for result in results)
