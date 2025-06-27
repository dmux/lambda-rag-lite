"""
Exemplo básico de uso da biblioteca Lambda RAG Lite.

Este exemplo demonstra como usar os componentes principais da biblioteca
para criar um sistema RAG simples.
"""

from pathlib import Path

from lambda_rag_lite import MarkdownLoader, PurePythonVectorStore, SimpleHashEmbedding


def main():
    # Exemplo 1: Uso básico com textos simples
    print("=== Exemplo 1: Textos Simples ===")

    # Cria embedding
    embedder = SimpleHashEmbedding(dim=256)

    # Textos de exemplo
    texts = [
        "Python é uma linguagem de programação de alto nível.",
        "Machine learning é uma área da inteligência artificial.",
        "RAG combina recuperação de informações com geração de texto.",
        "LangChain é um framework para desenvolver aplicações com LLMs.",
        "Vector stores permitem busca semântica em documentos.",
    ]

    # Cria vector store
    vector_store = PurePythonVectorStore.from_texts(texts=texts, embedding=embedder)

    # Consulta
    query = "O que é machine learning?"
    results = vector_store.similarity_search(query, k=3)

    print(f"Consulta: {query}")
    print("Resultados:")
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc.page_content}")

    print()

    # Exemplo 2: Com scores de similaridade
    print("=== Exemplo 2: Com Scores ===")

    results_with_scores = vector_store.similarity_search_with_score(query, k=3)

    for i, (doc, score) in enumerate(results_with_scores, 1):
        print(f"{i}. Score: {score:.3f} - {doc.page_content}")

    print()

    # Exemplo 3: Carregando arquivos Markdown (se existir pasta docs)
    print("=== Exemplo 3: Carregando Documentos ===")

    docs_dir = Path("docs")
    if docs_dir.exists():
        # Carrega documentos
        loader = MarkdownLoader(docs_dir)
        documents = loader.load()

        print(f"Carregados {len(documents)} documentos")

        # Cria novo vector store com documentos
        doc_vector_store = PurePythonVectorStore.from_documents(
            documents=documents, embedding=embedder
        )

        # Consulta nos documentos
        doc_query = "Como usar esta biblioteca?"
        doc_results = doc_vector_store.similarity_search(doc_query, k=2)

        print(f"Consulta: {doc_query}")
        for i, doc in enumerate(doc_results, 1):
            source = doc.metadata.get("source", "Unknown")
            content = (
                doc.page_content[:200] + "..."
                if len(doc.page_content) > 200
                else doc.page_content
            )
            print(f"{i}. Fonte: {source}")
            print(f"   Conteúdo: {content}")
    else:
        print("Pasta 'docs' não encontrada. Criando documentos de exemplo...")

        # Cria documentos de exemplo
        from langchain_core.documents import Document

        sample_docs = [
            Document(
                page_content="Lambda RAG Lite é uma biblioteca Python para RAG que não requer dependências pesadas como NumPy ou bibliotecas de ML.",
                metadata={"source": "intro.md", "topic": "introduction"},
            ),
            Document(
                page_content="O SimpleHashEmbedding usa hashing SHA-1 para converter tokens em índices de vetor, criando embeddings esparsos.",
                metadata={"source": "embeddings.md", "topic": "embeddings"},
            ),
            Document(
                page_content="O PurePythonVectorStore implementa a interface LangChain VectorStore usando apenas Python puro para armazenamento.",
                metadata={"source": "vectorstore.md", "topic": "storage"},
            ),
        ]

        doc_vector_store = PurePythonVectorStore.from_documents(
            documents=sample_docs, embedding=embedder
        )

        doc_query = "Como funciona o embedding?"
        doc_results = doc_vector_store.similarity_search_with_score(doc_query, k=2)

        print(f"Consulta: {doc_query}")
        for i, (doc, score) in enumerate(doc_results, 1):
            source = doc.metadata.get("source", "Unknown")
            print(f"{i}. Score: {score:.3f} - Fonte: {source}")
            print(f"   {doc.page_content}")

    print()

    # Exemplo 4: Maximum Marginal Relevance
    print("=== Exemplo 4: MMR (Diversidade) ===")

    # Adiciona textos similares para testar diversidade
    similar_texts = [
        "Python é uma linguagem muito popular para ciência de dados.",
        "Python tem uma sintaxe simples e legível.",
        "JavaScript é usado para desenvolvimento web frontend.",
        "Java é uma linguagem orientada a objetos.",
    ]

    vector_store.add_texts(similar_texts)

    # Compara busca normal vs MMR
    normal_results = vector_store.similarity_search("linguagem Python", k=3)
    mmr_results = vector_store.max_marginal_relevance_search("linguagem Python", k=3)

    print("Busca normal:")
    for i, doc in enumerate(normal_results, 1):
        print(f"{i}. {doc.page_content}")

    print("\nBusca com MMR (mais diversa):")
    for i, doc in enumerate(mmr_results, 1):
        print(f"{i}. {doc.page_content}")

    print()

    # Exemplo 5: Estatísticas do vector store
    print("=== Exemplo 5: Estatísticas ===")
    print(f"Total de documentos: {len(vector_store)}")
    print(f"Dimensão dos embeddings: {embedder.dim}")

    # Testa recuperação por IDs
    all_docs = vector_store.get_all_documents()
    print(f"Primeiros 3 documentos:")
    for i, doc in enumerate(all_docs[:3], 1):
        print(f"{i}. {doc.page_content[:50]}...")


if __name__ == "__main__":
    main()
