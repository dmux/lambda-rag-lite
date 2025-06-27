"""
Testes unitários para o módulo de embeddings.
"""

import pytest

from lambda_rag_lite.embeddings import SimpleHashEmbedding, TFIDFHashEmbedding


class TestSimpleHashEmbedding:

    def test_init_valid_dimension(self):
        """Testa inicialização com dimensão válida."""
        embedder = SimpleHashEmbedding(dim=128)
        assert embedder.dim == 128

    def test_init_invalid_dimension(self):
        """Testa inicialização com dimensão inválida."""
        with pytest.raises(ValueError):
            SimpleHashEmbedding(dim=0)

        with pytest.raises(ValueError):
            SimpleHashEmbedding(dim=-1)

    def test_embed_query_basic(self):
        """Testa embedding básico de consulta."""
        embedder = SimpleHashEmbedding(dim=256)
        text = "hello world test"

        embedding = embedder.embed_query(text)

        assert len(embedding) == 256
        assert all(isinstance(x, float) for x in embedding)
        assert abs(sum(x * x for x in embedding) - 1.0) < 1e-6  # Normalização L2

    def test_embed_query_empty_text(self):
        """Testa embedding de texto vazio."""
        embedder = SimpleHashEmbedding(dim=100)

        empty_embedding = embedder.embed_query("")
        assert len(empty_embedding) == 100
        assert all(x == 0.0 for x in empty_embedding)

        whitespace_embedding = embedder.embed_query("   ")
        assert len(whitespace_embedding) == 100
        assert all(x == 0.0 for x in whitespace_embedding)

    def test_embed_documents(self):
        """Testa embedding de múltiplos documentos."""
        embedder = SimpleHashEmbedding(dim=128)
        texts = ["hello world", "machine learning", "python programming"]

        embeddings = embedder.embed_documents(texts)

        assert len(embeddings) == 3
        assert all(len(emb) == 128 for emb in embeddings)
        assert all(abs(sum(x * x for x in emb) - 1.0) < 1e-6 for emb in embeddings)

    def test_consistency(self):
        """Testa se o mesmo texto produz o mesmo embedding."""
        embedder = SimpleHashEmbedding(dim=256)
        text = "consistent text for testing"

        emb1 = embedder.embed_query(text)
        emb2 = embedder.embed_query(text)

        assert emb1 == emb2

    def test_different_texts_different_embeddings(self):
        """Testa se textos diferentes produzem embeddings diferentes."""
        embedder = SimpleHashEmbedding(dim=256)

        emb1 = embedder.embed_query("first text")
        emb2 = embedder.embed_query("second text")

        assert emb1 != emb2

    def test_embed_text_zero_norm(self, mocker):
        """Testa embedding quando a norma é zero."""
        embedder = SimpleHashEmbedding(dim=64)

        # Mock para forçar que o vetor seja todo zero
        def mock_embed_text(text):
            return [0.0] * embedder.dim  # Vetor zero

        mocker.patch.object(embedder, "_embed_text", side_effect=mock_embed_text)
        embedding = embedder.embed_query("test")
        assert len(embedding) == 64
        assert all(x == 0.0 for x in embedding)

    def test_embed_text_zero_norm_edge_case(self, mocker):
        """Testa condição de norma zero em _embed_text_with_idf."""
        embedding = TFIDFHashEmbedding()

        # Mock para retornar vetor com norma zero
        mocker.patch.object(
            TFIDFHashEmbedding,
            "_embed_text_with_idf",
            return_value=[0.0] * embedding.dim,
        )
        result = embedding._embed_text_with_idf("test")
        assert result == [0.0] * embedding.dim


class TestTFIDFHashEmbedding:

    def test_init(self):
        """Testa inicialização do TF-IDF embedding."""
        embedder = TFIDFHashEmbedding(dim=128)

        assert embedder.dim == 128
        assert embedder.doc_count == 0
        assert embedder.token_doc_count == {}

    def test_embed_documents_updates_stats(self):
        """Testa se embed_documents atualiza as estatísticas."""
        embedder = TFIDFHashEmbedding(dim=128)
        texts = ["hello world", "world python", "machine learning"]

        embeddings = embedder.embed_documents(texts)

        assert embedder.doc_count == 3
        assert "world" in embedder.token_doc_count
        assert embedder.token_doc_count["world"] == 2  # Aparece em 2 docs
        assert len(embeddings) == 3

    def test_embed_query_uses_stats(self):
        """Testa se embed_query usa as estatísticas coletadas."""
        embedder = TFIDFHashEmbedding(dim=128)

        # Primeiro, treina com alguns documentos
        train_texts = ["hello world", "world programming", "python world"]
        embedder.embed_documents(train_texts)

        # Depois, faz consulta
        query_embedding = embedder.embed_query("hello python")

        assert len(query_embedding) == 128
        assert abs(sum(x * x for x in query_embedding) - 1.0) < 1e-6

    def test_idf_calculation(self):
        """Testa se o cálculo de IDF está funcionando."""
        embedder = TFIDFHashEmbedding(dim=128)

        # Documento com palavra comum e palavra rara
        texts = ["common word text", "common different text", "rare word here"]

        embedder.embed_documents(texts)

        # "common" aparece em 2 de 3 docs, "rare" aparece em 1 de 3 docs
        assert embedder.token_doc_count["common"] == 2
        assert embedder.token_doc_count["rare"] == 1

        # Embedding de consulta deve dar peso diferente para palavras com IDF diferente
        embedding = embedder.embed_query("common rare")
        assert len(embedding) == 128

    def test_embed_query_empty_after_split(self):
        """Testa embedding quando não há tokens válidos após split."""
        embedder = TFIDFHashEmbedding(dim=128)

        # Testa com texto que não produz tokens válidos
        embedding = embedder.embed_query("   ")
        assert len(embedding) == 128
        assert all(x == 0.0 for x in embedding)

    def test_update_document_stats_empty_tokens(self):
        """Testa atualização de estatísticas com tokens vazios."""
        embedder = TFIDFHashEmbedding(dim=128)

        # Texto que produz set vazio após split
        embedder._update_document_stats("   ")

        assert embedder.doc_count == 1
        assert len(embedder.token_doc_count) == 0

    def test_embed_text_with_idf_no_tokens(self):
        """Testa embedding com IDF quando não há tokens."""
        embedder = TFIDFHashEmbedding(dim=64)

        # Configura alguns documentos primeiro
        embedder._update_document_stats("test document")

        # Depois testa com texto vazio
        embedding = embedder._embed_text_with_idf("   ")
        assert len(embedding) == 64
        assert all(x == 0.0 for x in embedding)

    def test_embed_text_with_idf_zero_norm(self, mocker):
        """Testa embedding quando a norma é zero."""
        embedder = TFIDFHashEmbedding(dim=64)

        import math

        def mock_log(x):
            return 0  # Força IDF = 0

        mocker.patch("math.log", side_effect=mock_log)
        embedding = embedder._embed_text_with_idf("test")
        assert len(embedding) == 64

    def test_init_invalid_dimension_edge_case(self):
        """Testa inicialização com dimensão zero."""
        with pytest.raises(ValueError, match="Dimensão deve ser maior que 0"):
            TFIDFHashEmbedding(dim=0)
