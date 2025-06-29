import sys

sys.path.insert(0, "/Users/guru/work/dmux/lambda-rag-lite")

from lambda_rag_lite.strategies.chunking import (
    CharacterChunkingStrategy,
    ChunkingConfig,
)

text = "Este é um texto muito longo que definitivamente será dividido em múltiplos chunks para testar a funcionalidade de processamento de texto em pedaços menores."
print(f"Texto: '{text}'")
print(f"Tamanho: {len(text)}")

strategy = CharacterChunkingStrategy()
config = ChunkingConfig(chunk_size=30, chunk_overlap=5, min_chunk_size=1)

# Teste da função de quebra de sentença
for i in range(0, len(text), 30):
    end_pos = min(i + 30, len(text))
    print(f"\nPosição {i}-{end_pos}: '{text[i:end_pos]}'")

    if end_pos < len(text):
        break_point = strategy._find_sentence_break(text, end_pos, config)
        print(f"Break point encontrado: {break_point}")
        print(f"Texto até break: '{text[i:break_point]}'")

        # Verifica caracteres ao redor do break point
        start_check = max(0, break_point - 5)
        end_check = min(len(text), break_point + 5)
        print(f"Contexto ao redor: '{text[start_check:end_check]}' (pos {break_point})")
