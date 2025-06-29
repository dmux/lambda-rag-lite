"""
Lambda RAG Lite - Uma biblioteca Python para RAG (Retrieval-Augmented Generation)
simples e eficiente, compatível com LangChain.

Esta biblioteca fornece uma implementação pura em Python de um vector store
que usa hashing para embeddings, ideal para casos onde você precisa de uma
solução leve sem dependências pesadas como NumPy ou bibliotecas de ML.
"""

# Novas classes especializadas
from .config import (
    ChunkingConfig,
    ConfigPresets,
    EmbeddingConfig,
    LoaderConfig,
    TextProcessingConfig,
    VectorStoreConfig,
)
from .detectors.encoding import EncodingDetector
from .detectors.file_type import FileTypeDetector
from .embeddings import SimpleHashEmbedding, TFIDFHashEmbedding
from .loaders import DirectoryLoader, LoaderFactory, MarkdownLoader, TextLoader
from .metadata.document_metadata import DocumentMetadataManager
from .processors.text_processor import TextProcessor as NewTextProcessor
from .strategies.chunking import ChunkingStrategy, TextChunker

# Importa as funções de compatibilidade
from .utils import TextProcessor  # Classe compatível
from .utils import (
    calculate_text_stats,
    chunk_text,
    clean_text,
    extract_keywords,
    format_file_size,
)
from .vectorstore import PurePythonVectorStore

__version__ = "0.2.0"
__all__ = [
    # Classes principais
    "SimpleHashEmbedding",
    "TFIDFHashEmbedding",
    "PurePythonVectorStore",
    "MarkdownLoader",
    "TextLoader",
    "DirectoryLoader",
    "LoaderFactory",
    # Configurações
    "ChunkingConfig",
    "TextProcessingConfig",
    "LoaderConfig",
    "EmbeddingConfig",
    "VectorStoreConfig",
    "ConfigPresets",
    # Novas classes especializadas
    "TextChunker",
    "ChunkingStrategy",
    "NewTextProcessor",
    "EncodingDetector",
    "FileTypeDetector",
    "DocumentMetadataManager",
    # Funções de compatibilidade
    "chunk_text",
    "clean_text",
    "extract_keywords",
    "calculate_text_stats",
    "format_file_size",
    "TextProcessor",  # Classe compatível
]
