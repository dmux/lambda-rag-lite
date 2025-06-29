"""
Utilitários e helpers para Lambda RAG Lite.

"""

from __future__ import annotations

from typing import List, Optional

from .processors.text_processor import TextProcessor as NewTextProcessor
from .strategies.chunking import chunk_text as new_chunk_text
from .text_cleaning import calculate_text_stats as calculate_text_stats_func
from .text_cleaning import clean_text as clean_text_func
from .text_cleaning import extract_keywords as extract_keywords_func
from .text_cleaning import format_file_size as format_file_size_func


def chunk_text(
    text: str | None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    separator: str = "\n\n",
) -> List[str]:
    """
    Divide texto em chunks menores com sobreposição.

    Args:
        text: Texto para dividir
        chunk_size: Tamanho máximo de cada chunk em caracteres
        chunk_overlap: Sobreposição entre chunks em caracteres
        separator: Separador preferencial para divisão

    Returns:
        Lista de chunks de texto
    """
    return new_chunk_text(text, chunk_size, chunk_overlap, separator)


def clean_text(text: str | None, remove_extra_whitespace: bool = True) -> str:
    """
    Limpa e normaliza texto para melhor processamento.

    Args:
        text: Texto para limpar
        remove_extra_whitespace: Se deve remover espaços extras

    Returns:
        Texto limpo
    """
    return clean_text_func(text, remove_extra_whitespace)


def extract_keywords(
    text: str | None, min_length: int = 3, max_words: int = 20
) -> List[str]:
    """
    Extrai palavras-chave simples de um texto.

    Args:
        text: Texto para extrair palavras-chave
        min_length: Comprimento mínimo das palavras
        max_words: Número máximo de palavras para retornar

    Returns:
        Lista de palavras-chave
    """
    return extract_keywords_func(text, min_length, max_words)


def calculate_text_stats(text: str | None) -> dict:
    """
    Calcula estatísticas básicas de um texto.

    Args:
        text: Texto para analisar

    Returns:
        Dicionário com estatísticas
    """
    return calculate_text_stats_func(text)


def format_file_size(size_bytes: int) -> str:
    """
    Formata tamanho de arquivo em formato legível.

    Args:
        size_bytes: Tamanho em bytes

    Returns:
        String formatada (ex: "1.5 MB")
    """
    return format_file_size_func(size_bytes)


class TextProcessor:
    """
    Classe para processamento avançado de texto com configurações personalizáveis.

    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        clean_text: bool = True,
        extract_keywords: bool = False,
    ):
        """
        Inicializa o processador de texto.

        Args:
            chunk_size: Tamanho dos chunks
            chunk_overlap: Sobreposição entre chunks
            clean_text: Se deve limpar o texto
            extract_keywords: Se deve extrair palavras-chave
        """
        # Cria configurações compatíveis para a nova implementação
        from .config import ChunkingConfig, TextProcessingConfig

        # Ajusta chunk_overlap se for maior que chunk_size
        if chunk_overlap >= chunk_size:
            chunk_overlap = max(0, chunk_size // 4)  # 25% do chunk_size

        chunking_config = ChunkingConfig(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

        processing_config = TextProcessingConfig(
            clean_text=clean_text, extract_keywords=extract_keywords
        )

        # Usa a nova implementação internamente
        self._processor = NewTextProcessor(processing_config, chunking_config)

    @property
    def chunk_size(self) -> int:
        """Retorna o tamanho do chunk configurado."""
        return self._processor.chunking_config.chunk_size

    @property
    def chunk_overlap(self) -> int:
        """Retorna a sobreposição configurada."""
        return self._processor.chunking_config.chunk_overlap

    @property
    def clean_text(self) -> bool:
        """Retorna se a limpeza de texto está habilitada."""
        return self._processor.processing_config.clean_text

    @property
    def extract_keywords(self) -> bool:
        """Retorna se a extração de palavras-chave está habilitada."""
        return self._processor.processing_config.extract_keywords

    def process_text(
        self, text: str | None, metadata: Optional[dict] = None
    ) -> List[dict]:
        """
        Processa texto completo retornando chunks com metadados.

        Args:
            text: Texto para processar
            metadata: Metadados base para adicionar aos chunks

        Returns:
            Lista de dicionários com texto e metadados
        """
        return self._processor.process_text(text, metadata)
