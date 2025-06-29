# Changelog

Todas as mudanças notáveis neste projeto serão documentadas neste arquivo.

O formato é baseado em [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
e este projeto adere ao [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2024-12-29

### 🚀 Major Refactoring & Architecture Improvements

Esta versão representa uma refatoração significativa da arquitetura da biblioteca, com foco em modularização, robustez e manutenibilidade.

### ✨ Mudanças

#### Nova Arquitetura Modular

- **Configurações Centralizadas**: Sistema de configuração tipo-seguro com dataclasses
  - `ChunkingConfig`: Configuração para estratégias de chunking
  - `TextProcessingConfig`: Configuração para processamento de texto
  - `LoaderConfig`: Configuração para loaders de documentos
  - `EmbeddingConfig` e `VectorStoreConfig`: Configurações futuras
  - `ConfigPresets`: Presets predefinidos para casos de uso comuns

#### Estratégias de Chunking (Pattern Strategy)

- **`ChunkingStrategy`**: Interface abstrata para estratégias de chunking
- **`SeparatorChunkingStrategy`**: Chunking baseado em separadores preferenciais
- **`CharacterChunkingStrategy`**: Chunking baseado em contagem de caracteres
- **`TextChunker`**: Classe principal que unifica as estratégias
- Algoritmos melhorados de detecção de quebras naturais
- Cálculo otimizado de sobreposição entre chunks

#### Módulos Especializados

- **`EncodingDetector`**: Detecção inteligente de encoding de arquivos
- **`FileTypeDetector`**: Detecção de tipos de arquivo baseada em extensão e MIME type
- **`DocumentMetadataManager`**: Gestão centralizada de metadados de documentos
- **`LoaderFactory`**: Factory pattern para criação de loaders apropriados
- **`TextProcessor`**: Processador avançado de texto com configurações personalizáveis

#### Constantes Centralizadas

- Mapeamentos abrangentes de extensões de arquivo para tipos
- Suporte expandido para linguagens de programação
- Categorizações de arquivos (code, document, data, config)
- Mapeamentos de MIME types

### 🔧 Correções

#### Correções Críticas no Chunking

- **Corrigida lógica de sobreposição**: Chunks não geram mais sobreposições excessivas
- **Corrigida detecção de quebras de sentença**: Prioriza quebras naturais adequadamente
- **Eliminados chunks minúsculos**: Filtros inteligentes para evitar chunks inválidos
- **Corrigido cálculo do próximo início**: Evita loops infinitos e garante progresso

#### Melhorias nos Loaders

- **Detecção robusta de encoding**: Fallback inteligente com múltiplas estratégias
- **Tratamento de erros aprimorado**: Loaders não falham com arquivos problemáticos
- **Metadados enriquecidos**: Informações mais detalhadas sobre arquivos carregados

#### Refatoração da API

- **Compatibilidade mantida**: APIs antigas continuam funcionando
- **Funções de conveniência**: `chunk_text()`, `clean_text()`, etc. usam nova arquitetura
- **TextProcessor legado**: Reimplementado usando novos componentes internamente

#### Estrutura de Diretórios

```text
lambda_rag_lite/
├── config.py                  # Configurações centralizadas
├── constants.py               # Constantes e mapeamentos
├── text_cleaning.py          # Funções de limpeza de texto
├── factories.py              # Factory patterns
├── detectors/                # Detectores especializados
│   ├── encoding.py
│   └── file_type.py
├── metadata/                 # Gestão de metadados
│   └── document_metadata.py
├── processors/               # Processadores de texto
│   └── text_processor.py
└── strategies/               # Estratégias de chunking
    └── chunking.py
```

#### Performance e Robustez

- **Algoritmos otimizados**: Chunking mais eficiente e preciso
- **Tratamento de erros**: Recuperação graceful de falhas
- **Validação de entrada**: Validações robustas em todas as operações

#### Metadados e Análise

- **Metadados enriquecidos**: Informações detalhadas sobre arquivos e chunks
- **Estatísticas de texto**: Análise aprofundada do conteúdo
- **Detecção de tipo**: Classificação inteligente de arquivos

### 🔄 Deprecated

- Nenhuma funcionalidade foi depreciada nesta versão
- APIs antigas mantêm compatibilidade total

### 🚨 Migration Guide

Esta versão mantém total compatibilidade com a API anterior. Nenhuma mudança é necessária no código existente.

**Recomendações para novos projetos:**

```python
from lambda_rag_lite import (
    ChunkingConfig, TextChunker, TextProcessor as NewTextProcessor,
    EncodingDetector, ConfigPresets
)

# Use as novas classes para maior flexibilidade
config = ConfigPresets.large_documents()
chunker = TextChunker()
chunks = chunker.chunk(text, config)
```

---

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
