"""
Processador de texto.
"""

from __future__ import annotations

import re
from typing import Any

from ..config import ChunkingConfig, TextProcessingConfig
from ..strategies.chunking import TextChunker
from ..text_cleaning import calculate_text_stats, extract_keywords


class TextProcessor:
    """
    Processador avançado de texto com configurações personalizáveis.

    """

    def __init__(
        self,
        processing_config: TextProcessingConfig | None = None,
        chunking_config: ChunkingConfig | None = None,
    ):
        """
        Inicializa o processador de texto.

        Args:
            processing_config: Configuração de processamento de texto
            chunking_config: Configuração de chunking
        """
        self.processing_config = processing_config or TextProcessingConfig()
        self.chunking_config = chunking_config or ChunkingConfig()
        self.chunker = TextChunker()

    def process_text(
        self, text: str | None, metadata: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """
        Processa texto completo retornando chunks com metadados.

        Args:
            text: Texto para processar
            metadata: Metadados base para adicionar aos chunks

        Returns:
            Lista de dicionários com texto e metadados
        """
        if not text:
            return []

        # Limpa texto se configurado
        processed_text = self._clean_text_if_needed(text)

        # Divide em chunks
        chunks = self.chunker.chunk(processed_text, self.chunking_config)

        results = []
        base_metadata = metadata or {}

        for i, chunk in enumerate(chunks):
            chunk_metadata = base_metadata.copy()
            chunk_metadata.update(self._create_chunk_metadata(chunk, i, len(chunks)))

            # Adiciona palavras-chave se configurado
            if self.processing_config.extract_keywords:
                keywords = extract_keywords(
                    chunk,
                    self.processing_config.keyword_min_length,
                    self.processing_config.keyword_max_words,
                )
                chunk_metadata["keywords"] = keywords

            # Adiciona estatísticas se configurado
            if self.processing_config.calculate_stats:
                stats = calculate_text_stats(chunk)
                chunk_metadata.update(stats)

            results.append({"text": chunk, "metadata": chunk_metadata})

        return results

    def clean_text(
        self, text: str | None, remove_extra_whitespace: bool | None = None
    ) -> str:
        """
        Limpa e normaliza texto para melhor processamento.

        Args:
            text: Texto para limpar
            remove_extra_whitespace: Override da configuração padrão

        Returns:
            Texto limpo
        """
        if not text:
            return ""

        # Remove caracteres de controle exceto quebras de linha e tabs
        text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)

        should_remove_whitespace = (
            remove_extra_whitespace
            if remove_extra_whitespace is not None
            else self.processing_config.remove_extra_whitespace
        )

        if should_remove_whitespace:
            # Remove espaços múltiplos
            text = re.sub(r" +", " ", text)
            # Remove quebras de linha múltiplas
            text = re.sub(r"\n\s*\n", "\n\n", text)
            # Remove espaços no início e fim de linhas
            text = "\n".join(line.strip() for line in text.split("\n"))

        return text.strip()

    def extract_keywords(
        self,
        text: str | None,
        min_length: int | None = None,
        max_words: int | None = None,
    ) -> list[str]:
        """
        Extrai palavras-chave de um texto.

        Args:
            text: Texto para extrair palavras-chave
            min_length: Comprimento mínimo das palavras
            max_words: Número máximo de palavras para retornar

        Returns:
            Lista de palavras-chave
        """
        if not text:
            return []

        min_len = min_length or self.processing_config.keyword_min_length
        max_words_count = max_words or self.processing_config.keyword_max_words

        # Usa a função do módulo text_cleaning mas com configurações do processador
        return extract_keywords(text, min_len, max_words_count)

    def calculate_text_stats(self, text: str | None) -> dict[str, Any]:
        """
        Calcula estatísticas básicas de um texto.

        Args:
            text: Texto para analisar

        Returns:
            Dicionário com estatísticas
        """
        return calculate_text_stats(text)

    def _clean_text_if_needed(self, text: str) -> str:
        """Limpa texto se a configuração estiver habilitada."""
        if self.processing_config.clean_text:
            return self.clean_text(text)
        return text

    def _create_chunk_metadata(
        self, chunk: str, index: int, total_chunks: int
    ) -> dict[str, Any]:
        """Cria metadados básicos para um chunk."""
        return {
            "chunk_index": index,
            "chunk_count": total_chunks,
            "chunk_size": len(chunk),
        }
