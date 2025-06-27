"""
Testes unitários para o módulo de vector store.
"""

from langchain_core.documents import Document

from lambda_rag_lite.embeddings import SimpleHashEmbedding
from lambda_rag_lite.vectorstore import PurePythonVectorStore


class TestPurePythonVectorStore:

    def setup_method(self):
        """Setup para cada teste."""
        self.embedder = SimpleHashEmbedding(dim=128)
        self.vector_store = PurePythonVectorStore(self.embedder)

    def test_init(self):
        """Testa inicialização do vector store."""
        assert self.vector_store.embedding_function == self.embedder
        assert len(self.vector_store.vectors) == 0
        assert len(self.vector_store.texts) == 0
        assert len(self.vector_store.metadatas) == 0

    def test_from_texts(self):
        """Testa criação a partir de textos."""
        texts = ["hello world", "machine learning", "python programming"]
        metadatas = [{"id": 1}, {"id": 2}, {"id": 3}]

        vs = PurePythonVectorStore.from_texts(texts, self.embedder, metadatas)

        assert len(vs) == 3
        assert vs.texts == texts
        assert vs.metadatas == metadatas
        assert len(vs.vectors) == 3

    def test_from_documents(self):
        """Testa criação a partir de Documents."""
        docs = [
            Document(page_content="hello world", metadata={"id": 1}),
            Document(page_content="machine learning", metadata={"id": 2}),
        ]

        vs = PurePythonVectorStore.from_documents(docs, self.embedder)

        assert len(vs) == 2
        assert vs.texts[0] == "hello world"
        assert vs.metadatas[0] == {"id": 1}

    def test_add_texts(self):
        """Testa adição de textos."""
        initial_texts = ["first text"]
        self.vector_store.add_texts(initial_texts)

        assert len(self.vector_store) == 1

        new_texts = ["second text", "third text"]
        ids = self.vector_store.add_texts(new_texts)

        assert len(self.vector_store) == 3
        assert ids == ["1", "2"]  # IDs dos textos adicionados
        assert self.vector_store.texts[-1] == "third text"

    def test_add_texts_with_metadata(self):
        """Testa adição de textos com metadados."""
        texts = ["text one", "text two"]
        metadatas = [{"source": "doc1"}, {"source": "doc2"}]

        self.vector_store.add_texts(texts, metadatas)

        assert len(self.vector_store) == 2
        assert self.vector_store.metadatas[0] == {"source": "doc1"}
        assert self.vector_store.metadatas[1] == {"source": "doc2"}

    def test_similarity_search(self):
        """Testa busca por similaridade."""
        texts = [
            "python programming language",
            "machine learning algorithms",
            "web development with javascript",
        ]

        self.vector_store.add_texts(texts)

        results = self.vector_store.similarity_search("python coding", k=2)

        assert len(results) == 2
        assert all(isinstance(doc, Document) for doc in results)
        # O primeiro resultado deve ser mais relevante para "python"
        assert "python" in results[0].page_content.lower()

    def test_similarity_search_with_score(self):
        """Testa busca com scores."""
        texts = ["python language", "java language", "javascript code"]
        self.vector_store.add_texts(texts)

        results = self.vector_store.similarity_search_with_score("python", k=2)

        assert len(results) == 2
        assert all(isinstance(result, tuple) for result in results)
        assert all(len(result) == 2 for result in results)

        doc, score = results[0]
        assert isinstance(doc, Document)
        assert isinstance(score, float)
        assert 0 <= score <= 1  # Similaridade coseno normalizada

    def test_similarity_search_empty_store(self):
        """Testa busca em vector store vazio."""
        results = self.vector_store.similarity_search("any query")
        assert results == []

        results_with_score = self.vector_store.similarity_search_with_score("any query")
        assert results_with_score == []

    def test_max_marginal_relevance_search(self):
        """Testa busca com Maximum Marginal Relevance."""
        # Textos similares para testar diversidade
        texts = [
            "python programming language syntax",
            "python coding best practices",
            "python development tools",
            "java programming language",
            "javascript web development",
        ]

        self.vector_store.add_texts(texts)

        # MMR deve retornar resultados mais diversos
        mmr_results = self.vector_store.max_marginal_relevance_search(
            "python programming", k=3, fetch_k=5
        )

        assert len(mmr_results) <= 3
        assert all(isinstance(doc, Document) for doc in mmr_results)

    def test_max_marginal_relevance_search_empty_store(self):
        """Testa MMR em vector store vazio."""
        results = self.vector_store.max_marginal_relevance_search("any query")
        assert results == []

    def test_max_marginal_relevance_search_no_candidates(self, mocker):
        """Testa MMR quando não há candidatos."""
        # Mock para simular similarity_search_with_score retornando lista vazia
        mocker.patch.object(
            self.vector_store, "similarity_search_with_score", return_value=[]
        )
        results = self.vector_store.max_marginal_relevance_search("query")
        assert results == []

    def test_max_marginal_relevance_search_single_candidate(self):
        """Testa MMR com apenas um candidato."""
        self.vector_store.add_texts(["single text"])

        results = self.vector_store.max_marginal_relevance_search("text", k=3)
        assert len(results) == 1

    def test_delete_documents(self):
        """Testa remoção de documentos."""
        texts = ["doc1", "doc2", "doc3"]
        ids = self.vector_store.add_texts(texts)

        assert len(self.vector_store) == 3

        # Remove o segundo documento (índice 1)
        success = self.vector_store.delete(["1"])

        assert success is True
        assert len(self.vector_store) == 2
        assert self.vector_store.texts == ["doc1", "doc3"]

    def test_delete_invalid_ids(self):
        """Testa remoção com IDs inválidos."""
        self.vector_store.add_texts(["text1", "text2"])

        # IDs inválidos
        result = self.vector_store.delete(["invalid", "999", "-1"])
        assert result is False  # Retorna False quando nenhum ID é válido
        assert len(self.vector_store) == 2  # Nenhum documento removido

    def test_delete_none_ids(self):
        """Testa remoção com IDs None."""
        self.vector_store.add_texts(["text1"])

        result = self.vector_store.delete(None)
        assert result is None
        assert len(self.vector_store) == 1

    def test_delete_empty_ids(self):
        """Testa remoção com lista de IDs vazia."""
        self.vector_store.add_texts(["text1"])

        result = self.vector_store.delete([])
        assert result is None
        assert len(self.vector_store) == 1

    def test_cosine_similarity(self):
        """Testa cálculo de similaridade coseno."""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        vec3 = [1.0, 0.0, 0.0]

        # Vetores ortogonais
        sim1 = self.vector_store._cosine_similarity(vec1, vec2)
        assert abs(sim1 - 0.0) < 1e-6

        # Vetores idênticos
        sim2 = self.vector_store._cosine_similarity(vec1, vec3)
        assert abs(sim2 - 1.0) < 1e-6

        # Vetores de tamanhos diferentes
        sim3 = self.vector_store._cosine_similarity(vec1, [1.0, 0.0])
        assert sim3 == 0.0

    def test_cosine_similarity_zero_norms(self):
        """Testa similaridade coseno com vetores de norma zero."""
        # Vetores com norma zero
        vec1 = [0.0, 0.0, 0.0]
        vec2 = [1.0, 2.0, 3.0]

        sim = self.vector_store._cosine_similarity(vec1, vec2)
        assert sim == 0.0

        # Ambos com norma zero
        sim = self.vector_store._cosine_similarity(vec1, vec1)
        assert sim == 0.0

    def test_len_method(self):
        """Testa método __len__."""
        assert len(self.vector_store) == 0

        self.vector_store.add_texts(["one", "two", "three"])
        assert len(self.vector_store) == 3

        self.vector_store.delete(["1"])
        assert len(self.vector_store) == 2

    def test_metadata_isolation(self):
        """Testa se metadados são isolados (não compartilhados por referência)."""
        base_metadata = {"shared": "value"}
        texts = ["text1", "text2"]
        metadatas = [base_metadata.copy(), base_metadata.copy()]

        self.vector_store.add_texts(texts, metadatas)

        # Modifica metadados originais
        base_metadata["shared"] = "modified"

        # Metadados no vector store não devem ser afetados
        stored_docs = self.vector_store.get_all_documents()
        assert stored_docs[0].metadata["shared"] == "value"
        assert stored_docs[1].metadata["shared"] == "value"

    def test_get_by_ids(self):
        """Testa recuperação por IDs."""
        texts = ["first", "second", "third"]
        metadatas = [{"n": 1}, {"n": 2}, {"n": 3}]

        self.vector_store.add_texts(texts, metadatas)

        docs = self.vector_store.get_by_ids(["0", "2"])

        assert len(docs) == 2
        assert docs[0].page_content == "first"
        assert docs[1].page_content == "third"
        assert docs[0].metadata == {"n": 1}

    def test_get_by_ids_invalid_ids(self):
        """Testa recuperação com IDs inválidos."""
        self.vector_store.add_texts(["text1", "text2"])

        # IDs inválidos
        docs = self.vector_store.get_by_ids(["invalid", "999", "-1"])
        assert len(docs) == 0

    def test_get_all_documents(self):
        """Testa recuperação de todos os documentos."""
        texts = ["doc1", "doc2"]
        metadatas = [{"id": 1}, {"id": 2}]

        self.vector_store.add_texts(texts, metadatas)

        all_docs = self.vector_store.get_all_documents()

        assert len(all_docs) == 2
        assert all(isinstance(doc, Document) for doc in all_docs)
        assert all_docs[0].page_content == "doc1"
        assert all_docs[1].metadata == {"id": 2}

    def test_similarity_search_with_score_ordering(self):
        """Testa se os resultados estão ordenados por score decrescente."""
        texts = ["python", "java", "javascript"]
        self.vector_store.add_texts(texts)

        results = self.vector_store.similarity_search_with_score("python", k=3)

        # Scores devem estar em ordem decrescente
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_add_texts_empty_list(self):
        """Testa adição de lista vazia de textos."""
        ids = self.vector_store.add_texts([])
        assert ids == []
        assert len(self.vector_store) == 0

    def test_max_marginal_relevance_search_no_fetch_k_candidates(self, mocker):
        """Testa MMR quando similarity_search_with_score não retorna candidatos suficientes."""
        self.vector_store.add_texts(["test"])

        # Mock similarity_search_with_score para retornar lista vazia
        mocker.patch.object(
            self.vector_store, "similarity_search_with_score", return_value=[]
        )
        results = self.vector_store.max_marginal_relevance_search(
            "query", k=1, fetch_k=1
        )
        assert results == []
