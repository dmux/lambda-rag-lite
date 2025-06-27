"""
Módulo de embeddings para Lambda RAG Lite.

Fornece implementações de embedding que não dependem de bibliotecas externas
pesadas, usando técnicas de hashing para converter texto em vetores.
"""

from __future__ import annotations

import hashlib
import math
from typing import List

from langchain_core.embeddings import Embeddings


class SimpleHashEmbedding(Embeddings):
    """
    Embedding baseado em hashing que transforma texto em vetores esparsos.

    Esta implementação usa SHA-1 para mapear tokens para índices em um vetor
    de dimensão fixa, criando uma representação numérica do texto sem
    dependências externas.

    Attributes:
        dim: Dimensão do vetor de saída (padrão: 256)
    """

    def __init__(self, dim: int = 256):
        """
        Inicializa o embedding com dimensão especificada.

        Args:
            dim: Dimensão do vetor de embedding (deve ser > 0)

        Raises:
            ValueError: Se dim <= 0
        """
        if dim <= 0:
            raise ValueError("Dimensão deve ser maior que 0")
        self.dim = dim

    def embed_query(self, text: str) -> List[float]:
        """
        Converte uma consulta em vetor de embedding.

        Args:
            text: Texto da consulta para ser convertido

        Returns:
            Lista de floats representando o vetor normalizado
        """
        return self._embed_text(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Converte uma lista de documentos em vetores de embedding.

        Args:
            texts: Lista de textos para serem convertidos

        Returns:
            Lista de vetores de embedding, um para cada texto
        """
        return [self._embed_text(text) for text in texts]

    def _embed_text(self, text: str) -> List[float]:
        """
        Método interno para converter texto em vetor usando hashing.

        Args:
            text: Texto para ser convertido

        Returns:
            Vetor normalizado representando o texto
        """
        if not text or not text.strip():
            return [0.0] * self.dim

        vec = [0.0] * self.dim

        # Tokenização simples e contagem por hashing
        tokens = text.lower().split()
        for token in tokens:
            # Usa SHA-1 para mapear token para índice
            hash_hex = hashlib.sha1(token.encode("utf-8")).hexdigest()[:8]
            idx = int(hash_hex, 16) % self.dim
            vec[idx] += 1.0

        # Normalização L2
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0:
            return vec  # Retorna vetor zero se não há tokens válidos

        return [v / norm for v in vec]


class TFIDFHashEmbedding(Embeddings):
    """
    Variação do SimpleHashEmbedding que incorpora frequência de documentos.

    Esta versão mantém estatísticas sobre frequência de tokens em documentos
    para melhorar a qualidade dos embeddings usando uma aproximação de TF-IDF.
    """

    def __init__(self, dim: int = 256):
        """
        Inicializa o embedding TF-IDF com hashing.

        Args:
            dim: Dimensão do vetor de embedding
        """
        if dim <= 0:
            raise ValueError("Dimensão deve ser maior que 0")
        self.dim = dim
        self.doc_count = 0
        self.token_doc_count = {}  # Contagem de documentos que contêm cada token

    def embed_query(self, text: str) -> List[float]:
        """Converte consulta usando estatísticas já coletadas."""
        return self._embed_text_with_idf(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Converte documentos atualizando estatísticas de frequência.

        Args:
            texts: Lista de textos para processar

        Returns:
            Lista de vetores de embedding
        """
        # Primeiro passo: coletar estatísticas
        for text in texts:
            self._update_document_stats(text)

        # Segundo passo: gerar embeddings
        return [self._embed_text_with_idf(text) for text in texts]

    def _update_document_stats(self, text: str) -> None:
        """Atualiza estatísticas de frequência de documentos."""
        self.doc_count += 1
        tokens_in_doc = set(text.lower().split())

        for token in tokens_in_doc:
            if token not in self.token_doc_count:
                self.token_doc_count[token] = 0
            self.token_doc_count[token] += 1

    def _embed_text_with_idf(self, text: str) -> List[float]:
        """Gera embedding considerando IDF dos tokens."""
        if not text or not text.strip():
            return [0.0] * self.dim

        vec = [0.0] * self.dim
        tokens = text.lower().split()

        # Contagem de frequência no documento
        token_freq = {}
        for token in tokens:
            token_freq[token] = token_freq.get(token, 0) + 1

        for token, tf in token_freq.items():
            # Calcula IDF (inverso da frequência do documento)
            df = self.token_doc_count.get(token, 1)
            idf = math.log(max(self.doc_count, 1) / df)

            # TF-IDF score
            tfidf = tf * idf

            # Mapeia para índice usando hash
            hash_hex = hashlib.sha1(token.encode("utf-8")).hexdigest()[:8]
            idx = int(hash_hex, 16) % self.dim
            vec[idx] += tfidf

        # Normalização L2
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0:
            return vec

        return [v / norm for v in vec]
