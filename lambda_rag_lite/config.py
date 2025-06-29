"""
Configurações centralizadas para Lambda RAG Lite.

Contém dataclasses para configurações de chunking, processamento de texto,
e outras operações configuráveis.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .constants import MARKDOWN_EXTENSIONS, TEXT_EXTENSIONS


@dataclass
class ChunkingConfig:
    """Configuração para estratégias de chunking de texto."""

    chunk_size: int = 1000
    chunk_overlap: int = 200
    separator: str = "\n\n"
    natural_break_search_range: int = 50
    natural_break_chars: str = " \n\t"
    sentence_break_chars: str = ".!?\n"
    min_chunk_size: int = 10
    max_chunk_size: int = 4000

    def __post_init__(self):
        """Valida os parâmetros de configuração."""
        if self.chunk_size <= 0:
            raise ValueError("chunk_size deve ser maior que zero")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap não pode ser negativo")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap deve ser menor que chunk_size")
        if self.min_chunk_size <= 0:
            raise ValueError("min_chunk_size deve ser maior que zero")
        if self.max_chunk_size <= self.chunk_size:
            raise ValueError("max_chunk_size deve ser maior que chunk_size")


@dataclass
class TextProcessingConfig:
    """Configuração para processamento de texto."""

    clean_text: bool = True
    remove_extra_whitespace: bool = True
    extract_keywords: bool = False
    keyword_min_length: int = 3
    keyword_max_words: int = 20
    calculate_stats: bool = True

    # Stop words em português e inglês
    stop_words: set[str] = field(
        default_factory=lambda: {
            # Inglês
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
            # Português
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
            "que",
            "não",
            "se",
            "ou",
            "como",
            "mais",
            "mas",
            "ser",
            "ter",
            "esse",
            "sua",
            "seu",
            "ela",
            "ele",
            "quando",
            "onde",
            "porque",
            "ainda",
        }
    )


@dataclass
class LoaderConfig:
    """Configuração para loaders de documentos."""

    encoding: str = "utf-8"
    auto_detect_encoding: bool = True
    fallback_encodings: list[str] = field(
        default_factory=lambda: ["utf-8", "latin-1", "cp1252", "iso-8859-1"]
    )

    # Extensões suportadas por tipo
    text_extensions: set[str] = field(default_factory=lambda: TEXT_EXTENSIONS.copy())

    markdown_extensions: set[str] = field(
        default_factory=lambda: MARKDOWN_EXTENSIONS.copy()
    )

    # Tamanho máximo de arquivo (em bytes)
    max_file_size: int = 50 * 1024 * 1024  # 50MB

    # Se deve incluir metadados de arquivo
    include_file_metadata: bool = True

    # Se deve mostrar progresso
    show_progress: bool = False


@dataclass
class EmbeddingConfig:
    """Configuração para embeddings."""

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cpu"  # ou "cuda" se disponível
    batch_size: int = 32
    normalize_embeddings: bool = True
    trust_remote_code: bool = False


@dataclass
class VectorStoreConfig:
    """Configuração para vector store."""

    dimension: int = 384  # Dimensão padrão do all-MiniLM-L6-v2
    index_type: str = "flat"  # flat, hnsw, ivf
    distance_metric: str = "cosine"  # cosine, euclidean, dot_product

    # Parâmetros específicos do FAISS
    nlist: int = 100  # Para IVF
    nprobe: int = 10  # Para IVF
    m: int = 16  # Para HNSW
    ef_construction: int = 200  # Para HNSW
    ef_search: int = 100  # Para HNSW


# Configurações predefinidas para casos de uso comuns
class ConfigPresets:
    """Presets de configuração para casos de uso comuns."""

    @staticmethod
    def small_documents() -> ChunkingConfig:
        """Configuração otimizada para documentos pequenos."""
        return ChunkingConfig(
            chunk_size=500,
            chunk_overlap=50,
            separator="\n",
            natural_break_search_range=25,
        )

    @staticmethod
    def large_documents() -> ChunkingConfig:
        """Configuração otimizada para documentos grandes."""
        return ChunkingConfig(
            chunk_size=2000,
            chunk_overlap=400,
            separator="\n\n",
            natural_break_search_range=100,
        )

    @staticmethod
    def code_files() -> ChunkingConfig:
        """Configuração otimizada para arquivos de código."""
        return ChunkingConfig(
            chunk_size=1500,
            chunk_overlap=300,
            separator="\n\n",
            natural_break_search_range=75,
            sentence_break_chars="\n;{}",
        )

    @staticmethod
    def academic_papers() -> ChunkingConfig:
        """Configuração otimizada para papers acadêmicos."""
        return ChunkingConfig(
            chunk_size=1200,
            chunk_overlap=200,
            separator="\n\n",
            natural_break_search_range=60,
            sentence_break_chars=".!?\n",
        )

    @staticmethod
    def fast_processing() -> TextProcessingConfig:
        """Configuração para processamento rápido."""
        return TextProcessingConfig(
            clean_text=False, extract_keywords=False, calculate_stats=False
        )

    @staticmethod
    def comprehensive_processing() -> TextProcessingConfig:
        """Configuração para processamento completo."""
        return TextProcessingConfig(
            clean_text=True,
            extract_keywords=True,
            calculate_stats=True,
            keyword_max_words=50,
        )
