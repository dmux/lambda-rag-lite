# Changelog

Todas as mudanças notáveis neste projeto serão documentadas neste arquivo.

O formato é baseado em [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
e este projeto adere ao [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-12-26

### Added

- Implementação inicial da biblioteca Lambda RAG Lite
- `SimpleHashEmbedding`: Embedding baseado em hashing SHA-1
- `TFIDFHashEmbedding`: Embedding com suporte a TF-IDF
- `PurePythonVectorStore`: Vector store compatível com LangChain
- `MarkdownLoader`: Loader para arquivos Markdown
- `TextLoader`: Loader genérico para arquivos de texto
- `DirectoryLoader`: Loader automático para diretórios
- Utilitários para processamento de texto (chunking, limpeza, etc.)
- Suporte a Maximum Marginal Relevance (MMR)
- Testes unitários completos
- Exemplos de uso e integração com LangChain
- Documentação completa no README.md

### Features

- 🪶 Ultra-leve: Sem dependências pesadas (NumPy, scikit-learn, etc.)
- 🔗 Compatível com LangChain: Implementa interfaces padrão
- 🚀 Puro Python: Funciona em qualquer ambiente Python 3.10+
- 🎯 Fácil de usar: API simples e intuitiva
- 📦 Pronto para produção: Inclui testes e documentação
- 🔍 Busca avançada: Suporte a MMR para diversificar resultados
- 📊 Flexível: Múltiplos tipos de embedding e loaders

### Technical Details

- Compatibilidade com Python 3.10+
- Implementa interfaces LangChain VectorStore e Embeddings
- Algoritmo de similaridade coseno otimizado
- Suporte a metadados e chunking inteligente
- Detecção automática de encoding para arquivos
- Configuração flexível via pyproject.toml
