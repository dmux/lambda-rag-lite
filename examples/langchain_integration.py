"""
Exemplo de integração com LangChain para criação de um sistema RAG completo.

Este exemplo mostra como usar Lambda RAG Lite com outros componentes
LangChain para criar um pipeline RAG funcional.
"""

from lambda_rag_lite import PurePythonVectorStore, SimpleHashEmbedding
from lambda_rag_lite.utils import TextProcessor


def create_knowledge_base():
    """Cria uma base de conhecimento de exemplo."""

    # Documentos de exemplo sobre programação
    documents = {
        "python_basics.md": """
# Python Básico

Python é uma linguagem de programação de alto nível, interpretada e de propósito geral.

## Características principais:
- Sintaxe simples e legível
- Tipagem dinâmica
- Orientada a objetos
- Grande comunidade e biblioteca padrão rica

## Exemplo de código:
```python
def hello_world():
    print("Hello, World!")
    
hello_world()
```
        """,
        "machine_learning.md": """
# Machine Learning

Machine Learning é um subcampo da inteligência artificial que permite que computadores aprendam sem serem explicitamente programados.

## Tipos de ML:
1. **Aprendizado Supervisionado**: Usa dados rotulados
2. **Aprendizado Não Supervisionado**: Encontra padrões em dados não rotulados
3. **Aprendizado por Reforço**: Aprende através de recompensas e punições

## Aplicações:
- Reconhecimento de imagem
- Processamento de linguagem natural
- Sistemas de recomendação
- Análise preditiva
        """,
        "langchain_intro.md": """
# LangChain

LangChain é um framework para desenvolver aplicações powered por modelos de linguagem grandes (LLMs).

## Componentes principais:
- **Models**: Interface para diferentes LLMs
- **Prompts**: Templates para formatação de prompts
- **Chains**: Combinação de componentes para workflows
- **Memory**: Persistência de estado entre interações
- **Agents**: Sistemas que podem usar ferramentas

## RAG com LangChain:
RAG (Retrieval-Augmented Generation) combina recuperação de informações com geração de texto.
O processo típico inclui:
1. Indexar documentos em um vector store
2. Recuperar documentos relevantes para uma consulta
3. Usar documentos como contexto para geração
        """,
    }

    return documents


def main():
    print("=== Sistema RAG com Lambda RAG Lite ===\n")

    # 1. Criar base de conhecimento
    knowledge_base = create_knowledge_base()

    # 2. Configurar processador de texto
    processor = TextProcessor(
        chunk_size=500, chunk_overlap=100, clean_text=True, extract_keywords=True
    )

    # 3. Criar embedding
    embedder = SimpleHashEmbedding(dim=512)  # Dimensão maior para melhor qualidade

    # 4. Processar documentos
    all_chunks = []
    all_metadatas = []

    for filename, content in knowledge_base.items():
        # Processa texto em chunks
        processed_chunks = processor.process_text(
            content, metadata={"source": filename, "document_type": "markdown"}
        )

        for chunk_data in processed_chunks:
            all_chunks.append(chunk_data["text"])
            all_metadatas.append(chunk_data["metadata"])

    print(
        f"📚 Processados {len(all_chunks)} chunks de {len(knowledge_base)} documentos"
    )

    # 5. Criar vector store
    vector_store = PurePythonVectorStore.from_texts(
        texts=all_chunks, embedding=embedder, metadatas=all_metadatas
    )

    print(f"✅ Vector store criado com {len(vector_store)} documentos\n")

    # 6. Função de RAG simples
    def rag_query(question: str, k: int = 3):
        """Executa consulta RAG."""

        # Recupera documentos relevantes
        relevant_docs = vector_store.similarity_search_with_score(question, k=k)

        if not relevant_docs:
            return "Não encontrei informações relevantes para sua pergunta."

        # Monta contexto
        context_parts = []
        for i, (doc, score) in enumerate(relevant_docs, 1):
            source = doc.metadata.get("source", "Unknown")
            context_parts.append(
                f"[Fonte {i}: {source}, Score: {score:.3f}]\n{doc.page_content}"
            )

        context = "\n\n".join(context_parts)

        # Simula resposta baseada no contexto (sem LLM real)
        response = f"""
Baseado nos documentos encontrados, aqui estão as informações relevantes:

CONTEXTO RECUPERADO:
{context}

RESPOSTA:
Com base nos documentos acima, posso fornecer informações relevantes sobre sua pergunta: "{question}"

Para uma resposta mais elaborada, você poderia integrar este contexto com um LLM como GPT, Claude, ou qualquer outro modelo compatível com LangChain.
        """

        return response

    # 7. Exemplos de consultas
    queries = [
        "O que é Python e quais são suas características?",
        "Como funciona o machine learning supervisionado?",
        "O que é RAG e como implementar com LangChain?",
        "Quais são os tipos de aprendizado de máquina?",
    ]

    for query in queries:
        print(f"🔍 PERGUNTA: {query}")
        print("=" * 60)

        response = rag_query(query)

        # Mostra apenas os documentos relevantes para economizar espaço
        relevant_docs = vector_store.similarity_search_with_score(query, k=2)

        print("DOCUMENTOS RELEVANTES:")
        for i, (doc, score) in enumerate(relevant_docs, 1):
            source = doc.metadata.get("source", "Unknown")
            chunk_idx = doc.metadata.get("chunk_index", 0)
            keywords = doc.metadata.get("keywords", [])

            print(f"\n{i}. Fonte: {source} (chunk {chunk_idx})")
            print(f"   Score: {score:.3f}")
            print(f"   Keywords: {', '.join(keywords[:5])}")
            print(f"   Conteúdo: {doc.page_content[:200]}...")

        print("\n" + "=" * 60 + "\n")

    # 8. Demonstra Maximum Marginal Relevance
    print("🎯 DEMONSTRAÇÃO MMR (Maximum Marginal Relevance)")
    print("=" * 60)

    query = "linguagem programação"

    print(f"Consulta: {query}\n")

    # Busca normal
    normal_results = vector_store.similarity_search_with_score(query, k=4)
    print("BUSCA NORMAL (por similaridade):")
    for i, (doc, score) in enumerate(normal_results, 1):
        source = doc.metadata.get("source", "Unknown")
        print(f"{i}. {source} - Score: {score:.3f}")
        print(f"   {doc.page_content[:100]}...")

    print()

    # Busca MMR
    mmr_results = vector_store.max_marginal_relevance_search(query, k=4, fetch_k=8)
    print("BUSCA MMR (diversificada):")
    for i, doc in enumerate(mmr_results, 1):
        source = doc.metadata.get("source", "Unknown")
        print(f"{i}. {source}")
        print(f"   {doc.page_content[:100]}...")

    print("\n" + "=" * 60)

    # 9. Estatísticas finais
    print("📊 ESTATÍSTICAS DO SISTEMA")
    print("=" * 30)
    print(f"Total de documentos originais: {len(knowledge_base)}")
    print(f"Total de chunks processados: {len(vector_store)}")
    print(f"Dimensão dos embeddings: {embedder.dim}")

    # Analisa distribuição de chunks por fonte
    source_count = {}
    for metadata in all_metadatas:
        source = metadata.get("source", "Unknown")
        source_count[source] = source_count.get(source, 0) + 1

    print("\nDistribuição de chunks por fonte:")
    for source, count in source_count.items():
        print(f"  {source}: {count} chunks")

    print("\n✅ Demonstração concluída!")
    print("\nPara integrar com um LLM real, você pode:")
    print("1. Usar langchain.llms com OpenAI, Anthropic, etc.")
    print("2. Criar um prompt template que inclua o contexto recuperado")
    print("3. Implementar chains mais sofisticadas para diferentes tipos de consulta")


if __name__ == "__main__":
    main()
