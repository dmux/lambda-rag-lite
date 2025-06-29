import sys

sys.path.insert(0, "/Users/guru/work/dmux/lambda-rag-lite")

from lambda_rag_lite.text_cleaning import clean_text
from lambda_rag_lite.utils import TextProcessor, chunk_text

long_text = "Este é um texto muito longo que definitivamente será dividido em múltiplos chunks para testar a funcionalidade de processamento de texto em pedaços menores."
print(f"Texto original: {len(long_text)} chars")
print(f"Texto: '{long_text}'")

# Verificar o que a limpeza faz
cleaned = clean_text(long_text)
print(f"\nTexto após limpeza: {len(cleaned)} chars")
print(f"Texto limpo: '{cleaned}'")

# Testar chunking direto no texto limpo
chunks = chunk_text(cleaned, chunk_size=30, chunk_overlap=5)
print(f"\nChunks do texto limpo: {len(chunks)}")
for i, chunk in enumerate(chunks):
    print(f"Chunk {i}: len={len(chunk)}, text='{chunk}'")

# Testar sem limpeza
processor = TextProcessor(chunk_size=30, chunk_overlap=5, clean_text=False)
results = processor.process_text(long_text)
print(f"\nResultados sem limpeza: {len(results)}")
for i, result in enumerate(results):
    print(f"Chunk {i}: len={len(result['text'])}, text='{result['text']}'")
