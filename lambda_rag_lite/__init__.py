"""
Lambda RAG Lite - Uma biblioteca Python para RAG (Retrieval-Augmented Generation)
simples e eficiente, compatível com LangChain.

Esta biblioteca fornece uma implementação pura em Python de um vector store
que usa hashing para embeddings, ideal para casos onde você precisa de uma
solução leve sem dependências pesadas como NumPy ou bibliotecas de ML.
"""

from .embeddings import SimpleHashEmbedding, TFIDFHashEmbedding
from .loaders import DirectoryLoader, MarkdownLoader, TextLoader
from .vectorstore import PurePythonVectorStore

__version__ = "0.1.0"
__all__ = [
    "SimpleHashEmbedding",
    "TFIDFHashEmbedding",
    "PurePythonVectorStore",
    "MarkdownLoader",
    "TextLoader",
    "DirectoryLoader",
]
