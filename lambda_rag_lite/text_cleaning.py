"""
Funções utilitárias para limpeza e formatação de texto.

"""

from __future__ import annotations

import re


def clean_text(text: str | None, remove_extra_whitespace: bool = True) -> str:
    """
    Limpa e normaliza texto para melhor processamento.

    Args:
        text: Texto para limpar
        remove_extra_whitespace: Se deve remover espaços extras

    Returns:
        Texto limpo
    """
    if not text:
        return ""

    # Remove caracteres de controle exceto quebras de linha e tabs
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)

    if remove_extra_whitespace:
        # Remove espaços múltiplos
        text = re.sub(r" +", " ", text)
        # Remove quebras de linha múltiplas
        text = re.sub(r"\n\s*\n", "\n\n", text)
        # Remove espaços no início e fim de linhas
        text = "\n".join(line.strip() for line in text.split("\n"))

    return text.strip()


def extract_keywords(
    text: str | None, min_length: int = 3, max_words: int = 20
) -> list[str]:
    """
    Extrai palavras-chave simples de um texto.

    Args:
        text: Texto para extrair palavras-chave
        min_length: Comprimento mínimo das palavras
        max_words: Número máximo de palavras para retornar

    Returns:
        Lista de palavras-chave
    """
    if not text:
        return []

    # Remove pontuação e converte para minúsculas
    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    words = cleaned.split()

    # Filtra palavras muito curtas e stop words básicas
    stop_words = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "he",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "that",
        "the",
        "to",
        "was",
        "will",
        "with",
        "o",
        "e",
        "de",
        "do",
        "da",
        "em",
        "um",
        "uma",
        "para",
        "com",
        "por",
        "no",
        "na",
        "os",
        "dos",
        "das",
    }

    keywords = []
    word_count = {}

    for word in words:
        if len(word) >= min_length and word not in stop_words and word.isalpha():
            word_count[word] = word_count.get(word, 0) + 1

    # Ordena por frequência
    sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    keywords = [word for word, _ in sorted_words[:max_words]]

    return keywords


def calculate_text_stats(text: str | None) -> dict:
    """
    Calcula estatísticas básicas de um texto.

    Args:
        text: Texto para analisar

    Returns:
        Dicionário com estatísticas
    """
    if not text:
        return {
            "char_count": 0,
            "word_count": 0,
            "line_count": 0,
            "paragraph_count": 0,
            "avg_words_per_line": 0,
            "avg_chars_per_word": 0,
        }

    char_count = len(text)
    words = text.split()
    word_count = len(words)
    lines = text.split("\n")
    line_count = len(lines)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    paragraph_count = len(paragraphs)

    avg_words_per_line = word_count / line_count if line_count > 0 else 0
    avg_chars_per_word = char_count / word_count if word_count > 0 else 0

    return {
        "char_count": char_count,
        "word_count": word_count,
        "line_count": line_count,
        "paragraph_count": paragraph_count,
        "avg_words_per_line": round(avg_words_per_line, 2),
        "avg_chars_per_word": round(avg_chars_per_word, 2),
    }


def format_file_size(size_bytes: int) -> str:
    """
    Formata tamanho de arquivo em formato legível.

    Args:
        size_bytes: Tamanho em bytes

    Returns:
        String formatada (ex: "1.5 MB")
    """
    if size_bytes == 0:
        return "0 B"

    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    size = float(size_bytes)

    while size >= 1024.0 and i < len(size_names) - 1:
        size /= 1024.0
        i += 1

    return f"{size:.1f} {size_names[i]}"
