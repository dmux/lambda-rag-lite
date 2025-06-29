import sys

sys.path.insert(0, "/Users/guru/work/dmux/lambda-rag-lite")

from lambda_rag_lite.strategies.chunking import ChunkingConfig, TextChunker

# Teste 1: Problema com chunks pequenos
print("=== TESTE 1: CHUNKS PEQUENOS ===")
text = "A" * 100  # Texto de 100 caracteres
print(f"Texto original: {len(text)} chars")

chunker = TextChunker()
config = ChunkingConfig(chunk_size=30, chunk_overlap=10, min_chunk_size=1)
chunks = chunker.chunk(text, config)
print(f"Chunks gerados: {len(chunks)}")
for i, chunk in enumerate(chunks):
    print(f"Chunk {i}: len={len(chunk)}")

# Teste 2: Problema com texto longo
print("\n=== TESTE 2: TEXTO LONGO ===")
long_text = "Este é um texto muito longo que definitivamente será dividido em múltiplos chunks para testar a funcionalidade de processamento de texto em pedaços menores."
print(f"Texto original: {len(long_text)} chars")
print(f"Texto: '{long_text}'")

config2 = ChunkingConfig(chunk_size=30, chunk_overlap=5, min_chunk_size=1)
chunks2 = chunker.chunk(long_text, config2)
print(f"Chunks gerados: {len(chunks2)}")
for i, chunk in enumerate(chunks2):
    print(f"Chunk {i}: len={len(chunk)}, text='{chunk}'")

# Teste 3: Verificar configuração padrão
print("\n=== TESTE 3: CONFIGURAÇÃO PADRÃO ===")
try:
    default_config = ChunkingConfig()
    print(
        f"Config padrão - chunk_size: {default_config.chunk_size}, min_chunk_size: {default_config.min_chunk_size}"
    )
except Exception as e:
    print(f"Erro criando config padrão: {e}")
