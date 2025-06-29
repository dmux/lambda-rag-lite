"""
Fornece uma implementação pura em Python de um vector store compatível
com a interface LangChain, usando similaridade coseno para busca.

"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore


class PurePythonVectorStore(VectorStore):
    """
    Vector store implementado em Python puro, compatível com LangChain.

    Esta implementação armazena vetores, textos e metadados em listas Python
    e usa similaridade coseno para busca de documentos similares.

    Attributes:
        embedding_function: Função para gerar embeddings dos textos
        vectors: Lista de vetores armazenados
        texts: Lista de textos originais
        metadatas: Lista de metadados associados aos textos
    """

    def __init__(self, embedding_function: Embeddings):
        """
        Inicializa o vector store.

        Args:
            embedding_function: Instância de Embeddings para converter textos
        """
        self.embedding_function = embedding_function
        self.vectors: List[List[float]] = []
        self.texts: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> "PurePythonVectorStore":
        """
        Cria uma instância do vector store a partir de textos.

        Args:
            texts: Lista de textos para indexar
            embedding: Função de embedding
            metadatas: Metadados opcionais para cada texto
            **kwargs: Argumentos adicionais (ignorados)

        Returns:
            Nova instância do PurePythonVectorStore
        """
        instance = cls(embedding_function=embedding)
        instance.add_texts(texts, metadatas)
        return instance

    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        embedding: Embeddings,
        **kwargs: Any,
    ) -> "PurePythonVectorStore":
        """
        Cria uma instância a partir de uma lista de Documents.

        Args:
            documents: Lista de documentos LangChain
            embedding: Função de embedding
            **kwargs: Argumentos adicionais

        Returns:
            Nova instância do PurePythonVectorStore
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return cls.from_texts(texts, embedding, metadatas, **kwargs)

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """
        Adiciona textos ao vector store.

        Args:
            texts: Textos para adicionar
            metadatas: Metadados opcionais
            **kwargs: Argumentos adicionais

        Returns:
            Lista de IDs dos textos adicionados (índices como strings)
        """
        texts_list = list(texts)
        if not texts_list:
            return []

        # Gera embeddings para todos os textos
        embeddings = self.embedding_function.embed_documents(texts_list)

        # Adiciona aos armazenamentos internos
        start_idx = len(self.texts)
        ids = []

        for i, (text, embedding) in enumerate(zip(texts_list, embeddings)):
            self.texts.append(text)
            self.vectors.append(embedding)

            # Adiciona metadados
            if metadatas and i < len(metadatas):
                self.metadatas.append(metadatas[i])
            else:
                self.metadatas.append({})

            ids.append(str(start_idx + i))

        return ids

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """
        Busca documentos similares à consulta.

        Args:
            query: Texto da consulta
            k: Número de documentos para retornar
            **kwargs: Argumentos adicionais

        Returns:
            Lista dos k documentos mais similares
        """
        docs_with_scores = self.similarity_search_with_score(query, k, **kwargs)
        return [doc for doc, _ in docs_with_scores]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """
        Busca documentos similares retornando scores.

        Args:
            query: Texto da consulta
            k: Número de documentos para retornar
            **kwargs: Argumentos adicionais

        Returns:
            Lista de tuplas (documento, score) ordenadas por similaridade
        """
        if not self.vectors:
            return []

        # Gera embedding da consulta
        query_embedding = self.embedding_function.embed_query(query)

        # Calcula similaridades
        similarities = [
            self._cosine_similarity(query_embedding, vector) for vector in self.vectors
        ]

        # Ordena por similaridade (decrescente) e pega os top k
        indexed_similarities = list(enumerate(similarities))
        indexed_similarities.sort(key=lambda x: x[1], reverse=True)
        top_k = indexed_similarities[:k]

        # Cria documentos com scores
        results = []
        for idx, score in top_k:
            doc = Document(
                page_content=self.texts[idx], metadata=self.metadatas[idx].copy()
            )
            results.append((doc, score))

        return results

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        """
        Busca usando Maximum Marginal Relevance para diversificar resultados.

        Args:
            query: Texto da consulta
            k: Número final de documentos para retornar
            fetch_k: Número de documentos candidatos para considerar
            lambda_mult: Parâmetro de balanceamento relevância vs diversidade
            **kwargs: Argumentos adicionais

        Returns:
            Lista de documentos diversificados
        """
        if not self.vectors:
            return []

        # Busca candidatos iniciais
        candidates = self.similarity_search_with_score(query, fetch_k, **kwargs)
        if not candidates:
            return []

        query_embedding = self.embedding_function.embed_query(query)

        # Implementa algoritmo MMR
        selected = []
        remaining = candidates.copy()

        while len(selected) < k and remaining:
            best_score = float("-inf")
            best_idx = 0

            for i, (doc, relevance) in enumerate(remaining):
                # Calcula similaridade máxima com documentos já selecionados
                if selected:
                    doc_embedding = self.embedding_function.embed_query(
                        doc.page_content
                    )
                    max_similarity = max(
                        self._cosine_similarity(
                            doc_embedding,
                            self.embedding_function.embed_query(sel_doc.page_content),
                        )
                        for sel_doc in selected
                    )
                else:
                    max_similarity = 0

                # Score MMR
                mmr_score = lambda_mult * relevance - (1 - lambda_mult) * max_similarity

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i

            selected.append(remaining.pop(best_idx)[0])

        return selected

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """
        Remove documentos do vector store.

        Args:
            ids: Lista de IDs para remover (índices como strings)
            **kwargs: Argumentos adicionais

        Returns:
            True se removeu com sucesso, None se não implementado
        """
        if not ids:
            return None

        # Converte IDs para índices e ordena em ordem decrescente
        indices_to_remove = []
        for id_str in ids:
            try:
                idx = int(id_str)
                if 0 <= idx < len(self.texts):
                    indices_to_remove.append(idx)
            except ValueError:
                continue

        # Remove em ordem decrescente para não afetar índices
        for idx in sorted(indices_to_remove, reverse=True):
            del self.texts[idx]
            del self.vectors[idx]
            del self.metadatas[idx]

        return len(indices_to_remove) > 0

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """
        Calcula similaridade coseno entre dois vetores.

        Args:
            a: Primeiro vetor
            b: Segundo vetor

        Returns:
            Similaridade coseno (-1 a 1)
        """
        if len(a) != len(b):
            return 0.0

        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def get_by_ids(self, ids: List[str]) -> List[Document]:
        """
        Recupera documentos pelos IDs.

        Args:
            ids: Lista de IDs (índices como strings)

        Returns:
            Lista de documentos correspondentes
        """
        documents = []
        for id_str in ids:
            try:
                idx = int(id_str)
                if 0 <= idx < len(self.texts):
                    doc = Document(
                        page_content=self.texts[idx],
                        metadata=self.metadatas[idx].copy(),
                    )
                    documents.append(doc)
            except ValueError:
                continue
        return documents

    def __len__(self) -> int:
        """Retorna o número de documentos armazenados."""
        return len(self.texts)

    def get_all_documents(self) -> List[Document]:
        """
        Retorna todos os documentos armazenados.

        Returns:
            Lista com todos os documentos
        """
        return [
            Document(page_content=text, metadata=metadata.copy())
            for text, metadata in zip(self.texts, self.metadatas)
        ]
