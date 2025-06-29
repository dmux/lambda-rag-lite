import sys

sys.path.insert(0, "/Users/guru/work/dmux/lambda-rag-lite")

from lambda_rag_lite.utils import chunk_text

# Teste do primeiro caso
text = "A" * 100  # Texto de 100 caracteres
chunks = chunk_text(text, chunk_size=30, chunk_overlap=10)
print(f"Primeiro teste - Texto: {len(text)} chars")
print(f"Chunks gerados: {len(chunks)}")
for i, chunk in enumerate(chunks):
    print(
        f"Chunk {i}: len={len(chunk)}, text='{chunk[:50]}{'...' if len(chunk) > 50 else ''}'"
    )

# Teste do segundo caso
from lambda_rag_lite.utils import TextProcessor

processor = TextProcessor(chunk_size=30, chunk_overlap=5)
long_text = "Este é um texto muito longo que definitivamente será dividido em múltiplos chunks para testar a funcionalidade de processamento de texto em pedaços menores."
print(f"\nSegundo teste - Texto original: {len(long_text)} chars")
print(f"Texto: '{long_text}'")

results = processor.process_text(long_text)
print(f"Chunks processados: {len(results)}")
for i, result in enumerate(results):
    print(
        f"Chunk {i}: len={len(result['text'])}, text='{result['text'][:50]}{'...' if len(result['text']) > 50 else ''}'"
    )
