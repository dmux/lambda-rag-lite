"""
Utilitários e helpers para Lambda RAG Lite.

Contém funções auxiliares para processamento de texto, configuração
e outras operações comuns.
"""

from __future__ import annotations

import re
from typing import List, Optional


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
    if not text or chunk_size <= 0:
        return []

    if len(text) <= chunk_size:
        return [text]

    chunks = []

    # Tenta dividir usando o separador preferencial primeiro
    if separator in text:
        sections = text.split(separator)
        current_chunk = ""

        for section in sections:
            # Se a seção sozinha é maior que chunk_size, divide ela
            if len(section) > chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""

                # Divide seção grande por caracteres para evitar recursão infinita
                sub_start = 0
                while sub_start < len(section):
                    sub_end = sub_start + chunk_size
                    if sub_end >= len(section):
                        chunks.append(section[sub_start:].strip())
                        break

                    # Procura quebra natural
                    break_point = sub_end
                    for i in range(
                        max(0, sub_end - 50), min(len(section), sub_end + 50)
                    ):
                        if section[i] in " \n\t":
                            break_point = i
                            break

                    chunks.append(section[sub_start:break_point].strip())
                    sub_start = max(sub_start + 1, break_point - chunk_overlap)

            # Se adicionar esta seção não excede o limite
            elif len(current_chunk) + len(section) + len(separator) <= chunk_size:
                if current_chunk:
                    current_chunk += separator + section
                else:
                    current_chunk = section

            # Se excede, salva chunk atual e inicia novo
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = section

        # Adiciona último chunk se não vazio
        if current_chunk:
            chunks.append(current_chunk.strip())

    else:
        # Divisão por caracteres quando separador não está presente
        start = 0
        while start < len(text):
            end = start + chunk_size

            if end >= len(text):
                chunks.append(text[start:].strip())
                break

            # Tenta encontrar uma quebra natural próxima ao fim
            break_point = end
            search_start = max(0, end - 50)
            search_end = min(len(text), end + 50)

            for i in range(search_start, search_end):
                if i < len(text) and text[i] in ".!?\n":
                    break_point = i + 1
                    break

            chunks.append(text[start:break_point].strip())
            start = (
                max(0, break_point - chunk_overlap)
                if chunk_overlap > 0
                else break_point
            )

    # Remove chunks vazios
    return [chunk for chunk in chunks if chunk.strip()]


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
        "a",
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
        "as",
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
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.clean_text = clean_text
        self.extract_keywords = extract_keywords

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
        if not text:
            return []

        # Limpa texto se configurado
        processed_text = clean_text(text) if self.clean_text else text

        # Divide em chunks
        chunks = chunk_text(processed_text, self.chunk_size, self.chunk_overlap)

        results = []
        base_metadata = metadata or {}

        for i, chunk in enumerate(chunks):
            chunk_metadata = base_metadata.copy()
            chunk_metadata.update(
                {
                    "chunk_index": i,
                    "chunk_count": len(chunks),
                    "chunk_size": len(chunk),
                }
            )

            # Adiciona palavras-chave se configurado
            if self.extract_keywords:
                keywords = extract_keywords(chunk)
                chunk_metadata["keywords"] = keywords

            # Adiciona estatísticas
            stats = calculate_text_stats(chunk)
            chunk_metadata.update(stats)

            results.append({"text": chunk, "metadata": chunk_metadata})

        return results
