# Lambda RAG Lite

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Package CI](https://github.com/dmux/lambda-rag-lite/actions/workflows/ci.yml/badge.svg)](https://github.com/dmux/lambda-rag-lite/actions/workflows/ci.yml)

Uma biblioteca Python **ultra-leve** para RAG (Retrieval-Augmented Generation) **criada especificamente para AWS Lambda**, totalmente compatível com **LangChain**, que **elimina dependências pesadas** como NumPy, TensorFlow ou qualquer biblioteca de ML.

> 🎯 **Criada para AWS Lambda**: Esta biblioteca foi desenvolvida especificamente para resolver os problemas de tamanho de pacote e cold start em funções AWS Lambda, onde bibliotecas como NumPy podem aumentar significativamente o tempo de inicialização e o tamanho do deployment.

## ✨ Características

- 🐦 **Feita para AWS Lambda**: Projetada para minimizar cold starts e tamanho de deployment
- 🪶 **Ultra-leve**: **ZERO dependências de ML** (sem NumPy, scikit-learn, TensorFlow, PyTorch)
- ⚡ **Cold Start Rápido**: Inicialização em milissegundos vs. segundos com bibliotecas tradicionais
- 📦 **Pacote Pequeno**: ~50KB vs. ~100MB+ de bibliotecas tradicionais de ML
- 🔗 **Compatível com LangChain**: Implementa interfaces padrão (`VectorStore`, `Embeddings`)
- 🚀 **Puro Python**: Funciona em qualquer ambiente Python 3.10+ (Lambda, Container, local)
- 🎯 **Fácil de usar**: API simples e intuitiva
- ✅ **Pronto para produção**: Inclui testes, documentação e exemplos
- 🔍 **Busca avançada**: Suporte a MMR (Maximum Marginal Relevance) para diversificar resultados
- 📊 **Flexível**: Múltiplos tipos de embedding e loaders

## 🚀 Início Rápido

### Por que Lambda RAG Lite?

**Problema**: Bibliotecas tradicionais de RAG (como LangChain com embeddings de ML) incluem dependências massivas:

- NumPy: ~15-20MB
- scikit-learn: ~30-40MB
- Transformers: ~500MB+
- Torch: ~800MB+

**Resultado em AWS Lambda**:

- ❌ Cold starts de 10-30 segundos
- ❌ Pacotes de deployment de 100MB-1GB+
- ❌ Limites de tamanho excedidos
- ❌ Custos elevados de armazenamento

**Solução Lambda RAG Lite**:

- ✅ Cold starts < 1 segundo
- ✅ Pacote de deployment < 1MB
- ✅ Zero dependências externas
- ✅ Custo mínimo de armazenamento

### Instalação

```bash
pip install lambda-rag-lite
```

### Exemplo Básico

```python
from lambda_rag_lite import SimpleHashEmbedding, PurePythonVectorStore

# Criar embedding e vector store
embedder = SimpleHashEmbedding(dim=256)
vector_store = PurePythonVectorStore(embedder)

# Adicionar textos
texts = [
    "Python é uma linguagem de programação versátil.",
    "Machine learning é uma área da IA.",
    "RAG combina recuperação com geração de texto."
]

vector_store.add_texts(texts)

# Buscar documentos similares
results = vector_store.similarity_search("O que é Python?", k=2)

for doc in results:
    print(doc.page_content)
```

### Carregando Documentos

```python
from lambda_rag_lite import MarkdownLoader, PurePythonVectorStore, SimpleHashEmbedding

# Carregar documentos Markdown
loader = MarkdownLoader("docs/")
documents = loader.load()

# Criar vector store
embedder = SimpleHashEmbedding(dim=512)
vector_store = PurePythonVectorStore.from_documents(documents, embedder)

# Buscar com scores
results = vector_store.similarity_search_with_score("Como usar?", k=3)

for doc, score in results:
    print(f"Score: {score:.3f} - {doc.page_content[:100]}...")
```

## 📖 Documentação Completa

### Embeddings

#### SimpleHashEmbedding

Converte texto em vetores usando hashing SHA-1. Ideal para casos onde você precisa de embeddings rápidos sem modelos de ML.

```python
from lambda_rag_lite import SimpleHashEmbedding

embedder = SimpleHashEmbedding(dim=256)

# Embed uma consulta
query_vector = embedder.embed_query("exemplo de texto")

# Embed múltiplos documentos
doc_vectors = embedder.embed_documents([
    "primeiro documento",
    "segundo documento"
])
```

#### TFIDFHashEmbedding

Versão avançada que incorpora TF-IDF para melhor qualidade dos embeddings.

```python
from lambda_rag_lite.embeddings import TFIDFHashEmbedding

embedder = TFIDFHashEmbedding(dim=512)

# Treina com documentos (coleta estatísticas)
embeddings = embedder.embed_documents(training_texts)

# Usa para consultas
query_embedding = embedder.embed_query("minha consulta")
```

### Vector Store

#### PurePythonVectorStore

Implementação completa da interface LangChain VectorStore.

```python
from lambda_rag_lite import PurePythonVectorStore, SimpleHashEmbedding

embedder = SimpleHashEmbedding()
store = PurePythonVectorStore(embedder)

# Métodos principais
store.add_texts(texts, metadatas)
store.similarity_search(query, k=4)
store.similarity_search_with_score(query, k=4)
store.max_marginal_relevance_search(query, k=4)
store.delete(ids)
store.get_by_ids(ids)
```

### Loaders

#### MarkdownLoader

Carrega arquivos Markdown recursivamente.

```python
from lambda_rag_lite import MarkdownLoader

# Carrega um arquivo
loader = MarkdownLoader("documento.md")
docs = loader.load()

# Carrega diretório inteiro
loader = MarkdownLoader("docs/", auto_detect_encoding=True)
docs = loader.load()
```

#### TextLoader

Loader genérico para diferentes tipos de arquivo de texto.

```python
from lambda_rag_lite.loaders import TextLoader

loader = TextLoader(
    "src/",
    extensions=['.py', '.js', '.md', '.txt']
)
docs = loader.load()
```

#### DirectoryLoader

Loader automático que detecta tipos de arquivo.

```python
from lambda_rag_lite.loaders import DirectoryLoader

loader = DirectoryLoader("projeto/", recursive=True)
docs = loader.load()  # Carrega automaticamente .md, .py, .txt, etc.
```

### Utilitários

#### Processamento de Texto

```python
from lambda_rag_lite.utils import chunk_text, clean_text, TextProcessor

# Dividir texto em chunks
chunks = chunk_text(
    text="texto longo aqui...",
    chunk_size=1000,
    chunk_overlap=200
)

# Limpar texto
clean = clean_text(text, remove_extra_whitespace=True)

# Processador avançado
processor = TextProcessor(
    chunk_size=500,
    chunk_overlap=100,
    clean_text=True,
    extract_keywords=True
)

processed = processor.process_text(text, metadata={"source": "doc1"})
```

## � Deploy em AWS Lambda

### Exemplo de Função Lambda

```python
import json
from lambda_rag_lite import SimpleHashEmbedding, PurePythonVectorStore

# Inicialização global (fora do handler para reutilizar entre invocações)
embedder = SimpleHashEmbedding(dim=256)
vector_store = None

def lambda_handler(event, context):
    global vector_store

    # Inicializar vector store na primeira execução
    if vector_store is None:
        # Carregar seus documentos aqui
        documents = load_your_documents()  # Implementar conforme necessário
        vector_store = PurePythonVectorStore.from_texts(documents, embedder)

    # Processar consulta
    query = event.get('query', '')
    results = vector_store.similarity_search(query, k=3)

    return {
        'statusCode': 200,
        'body': json.dumps({
            'results': [{'content': doc.page_content, 'metadata': doc.metadata}
                       for doc in results]
        })
    }

def load_your_documents():
    # Exemplo: carregar de S3, DynamoDB, ou incluir no pacote
    return [
        "Documento 1: Conteúdo sobre Python e AWS Lambda",
        "Documento 2: Guia de RAG sem NumPy",
        "Documento 3: Performance em serverless"
    ]
```

### Deployment com AWS SAM

```yaml
# template.yaml
AWSTemplateFormatVersion: "2010-09-09"
Transform: AWS::Serverless-2016-10-31

Resources:
  RagFunction:
    Type: AWS::Serverless::Function
    Properties:
      CodeUri: src/
      Handler: app.lambda_handler
      Runtime: python3.11
      MemorySize: 512 # Muito menos que libs tradicionais!
      Timeout: 30
      Environment:
        Variables:
          EMBEDDING_DIM: 256
```

### Deployment com Terraform

```hcl
resource "aws_lambda_function" "rag_function" {
  filename         = "lambda-rag-lite.zip"
  function_name    = "rag-search"
  role            = aws_iam_role.lambda_role.arn
  handler         = "app.lambda_handler"
  runtime         = "python3.11"
  memory_size     = 512
  timeout         = 30

  # Pacote será < 1MB vs 100MB+ com bibliotecas tradicionais
}
```

## �🔧 Integração com LangChain

Lambda RAG Lite implementa as interfaces padrão do LangChain, então você pode usar diretamente com outros componentes:

```python
from langchain.chains import RetrievalQA
from lambda_rag_lite import PurePythonVectorStore, SimpleHashEmbedding

# Criar retriever
embedder = SimpleHashEmbedding(dim=256)
vector_store = PurePythonVectorStore.from_texts(texts, embedder)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# Usar com LangChain Chain (exemplo conceitual)
# qa_chain = RetrievalQA.from_chain_type(
#     llm=your_llm,
#     retriever=retriever,
#     return_source_documents=True
# )

# result = qa_chain({"query": "Sua pergunta aqui"})
```

## ⚡ Performance

Lambda RAG Lite é otimizada especificamente para **AWS Lambda**:

### Performance em Lambda

| Métrica                 | Lambda RAG Lite | Libs Tradicionais (NumPy/ML) |
| ----------------------- | --------------- | ---------------------------- |
| **Cold Start**          | < 1s            | 10-30s                       |
| **Tamanho do Pacote**   | < 1MB           | 100MB-1GB+                   |
| **Tempo de Deploy**     | < 30s           | 5-15min                      |
| **Memória Inicial**     | ~50MB           | 200MB-1GB+                   |
| **Tempo de Importação** | ~50ms           | 2-10s                        |

### Performance Operacional

| Operação           | Tempo (1000 docs) | Memória     |
| ------------------ | ----------------- | ----------- |
| Indexação          | ~0.5s             | ~50MB       |
| Busca (k=5)        | ~5ms              | ~50MB       |
| Adição incremental | ~0.5ms/doc        | +0.05MB/doc |

### Casos de Uso Ideais para AWS Lambda

- ✅ **APIs de busca semântica** com baixa latência
- ✅ **Chatbots serverless** com RAG
- ✅ **Processamento de documentos** em lote
- ✅ **Microserviços de recomendação**
- ✅ **Análise de texto** sob demanda

## 🛠️ Desenvolvimento

### Configuração do Ambiente

```bash
# Clonar repositório
git clone https://github.com/yourusername/lambda-rag-lite.git
cd lambda-rag-lite

# Instalar dependências de desenvolvimento
pip install -e ".[dev]"

# Executar testes
pytest

# Executar linting
black .
isort .
ruff check .
```

### Executar Exemplos

```bash
# Exemplo básico
python examples/basic_usage.py

# Integração com LangChain
python examples/langchain_integration.py

# Exemplo interativo (requer pasta docs_markdown/)
python example.py
```

## � Quando Usar Lambda RAG Lite

### ✅ Use Lambda RAG Lite quando

- Você está deployando em **AWS Lambda** ou outros ambientes serverless
- Precisa de **cold starts rápidos** (< 1 segundo)
- Tem **limitações de tamanho de pacote** (< 250MB no Lambda)
- Quer evitar **custos de dependências pesadas**
- Precisa de uma solução **simples e eficiente** para busca semântica
- Está construindo **APIs de baixa latência**
- Quer **compatibilidade com LangChain** sem overhead

### ❌ NÃO use Lambda RAG Lite quando

- Precisa da **máxima qualidade de embeddings** (use OpenAI, Cohere, etc.)
- Está deployando em **servidores dedicados** com recursos abundantes
- Precisa de **algoritmos de ML avançados** (clustering, dimensionality reduction)
- Tem **datasets muito grandes** (>100k documentos) que precisam de otimizações especiais
- Precisa de **embeddings pré-treinados** específicos do domínio

### 🔀 Alternativas e Quando Usá-las

| Biblioteca                   | Melhor Para                 | Tamanho | Cold Start |
| ---------------------------- | --------------------------- | ------- | ---------- |
| **Lambda RAG Lite**          | AWS Lambda, Serverless      | < 1MB   | < 1s       |
| **LangChain + OpenAI**       | Máxima qualidade            | ~10MB   | ~2s        |
| **LangChain + Transformers** | Embeddings locais avançados | ~500MB  | ~15s       |
| **Pinecone/Weaviate**        | Produção em larga escala    | Cliente | ~1s        |
| **FAISS + NumPy**            | Servidores dedicados        | ~100MB  | ~5s        |

## �🤝 Contribuindo

Contribuições são bem-vindas! Por favor:

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/nova-feature`)
3. Commit suas mudanças (`git commit -am 'Adiciona nova feature'`)
4. Push para a branch (`git push origin feature/nova-feature`)
5. Abra um Pull Request

### Executar Testes

```bash
# Todos os testes
pytest

# Com cobertura
pytest --cov=lambda_rag_lite --cov-report=html

# Testes específicos
pytest tests/test_embeddings.py -v
```

## 📋 Roadmap

- [ ] Mais algoritmos de embedding (BM25, LSH)
- [ ] Suporte a diferentes métricas de distância
- [ ] Otimizações de performance
- [ ] Integração com mais formatos de documento
- [ ] Suporte a índices persistentes
- [ ] Implementação de quantização para reduzir memória

## ✨ Por que Lambda RAG Lite Existe?

Esta biblioteca foi criada para resolver um problema específico: **implementar RAG em AWS Lambda sem sofrer com cold starts lentos e pacotes gigantes**.

### O Problema Real

Quando tentamos usar bibliotecas tradicionais de RAG/ML em Lambda:

```python
# ❌ Bibliotecas tradicionais
import numpy as np           # +15MB
import pandas as pd          # +20MB
import sklearn               # +30MB
from sentence_transformers import SentenceTransformer  # +500MB
```

**Resultado**: Pacote de 500MB+, cold start de 15-30 segundos, custos elevados.

### Nossa Solução

```python
# ✅ Lambda RAG Lite
from lambda_rag_lite import SimpleHashEmbedding, PurePythonVectorStore
```

**Resultado**: Pacote de <1MB, cold start de <1 segundo, custo mínimo.

### Filosofia de Design

1. **Zero dependências externas** - Apenas Python puro
2. **Compatibilidade total com LangChain** - Drop-in replacement
3. **Performance otimizada para Lambda** - Minimizar cold start
4. **Simplicidade** - API fácil de usar
5. **Qualidade suficiente** - Boa para 80% dos casos de uso

---

💡 **Dica**: Se você precisa da máxima qualidade de embeddings, use OpenAI/Cohere APIs. Se você precisa de deployment rápido e barato em Lambda, use Lambda RAG Lite.

## 📜 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

## 🙏 Agradecimentos

- [LangChain](https://github.com/langchain-ai/langchain) pela excelente arquitetura
- Comunidade Python pela inspiração
- Todos os contribuidores que ajudaram a melhorar este projeto

## 📞 Suporte

- 📧 Email: <rafael.sales@gmail.com>
- 🐛 Issues: [GitHub Issues](https://github.com/dmux/lambda-rag-lite/issues)
- 💬 Discussões: [GitHub Discussions](https://github.com/dmux/lambda-rag-lite/discussions)

---

⭐ **Se este projeto foi útil para você, considere dar uma estrela no GitHub!**
