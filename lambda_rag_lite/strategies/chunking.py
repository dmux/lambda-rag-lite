"""
Estratégias de chunking de texto implementando o padrão Strategy.

"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..config import ChunkingConfig


class ChunkingStrategy(ABC):
    """Interface base para estratégias de chunking."""

    @abstractmethod
    def chunk(self, text: str, config: ChunkingConfig) -> list[str]:
        """
        Divide texto em chunks usando a estratégia específica.

        Args:
            text: Texto para dividir
            config: Configuração de chunking

        Returns:
            Lista de chunks de texto
        """
        pass


class SeparatorChunkingStrategy(ChunkingStrategy):
    """Estratégia de chunking baseada em separadores."""

    def chunk(self, text: str, config: ChunkingConfig) -> list[str]:
        """
        Divide texto usando separadores preferenciais.

        Args:
            text: Texto para dividir
            config: Configuração de chunking

        Returns:
            Lista de chunks
        """
        # Validações iniciais
        if not self._is_valid_input(text, config):
            return []

        if len(text) <= config.chunk_size:
            return [text]

        # Escolhe estratégia baseada na presença do separador
        if config.separator in text:
            return self._chunk_by_separator(text, config)
        else:
            return self._chunk_by_characters(text, config)

    def _is_valid_input(self, text: str, config: ChunkingConfig) -> bool:
        """Valida entrada para chunking."""
        # Se chunk_size é 0 ou negativo, ou text é vazio, retorna False
        return bool(text and config.chunk_size > 0)

    def _chunk_by_separator(self, text: str, config: ChunkingConfig) -> list[str]:
        """Divide texto usando separadores."""
        sections = text.split(config.separator)
        chunks = []
        current_chunk = ""

        for section in sections:
            current_chunk = self._process_section(
                section, current_chunk, chunks, config
            )

        # Adiciona último chunk se não vazio
        if current_chunk:
            chunks.append(current_chunk.strip())

        return self._filter_valid_chunks(chunks, config)

    def _chunk_by_characters(self, text: str, config: ChunkingConfig) -> list[str]:
        """Divide texto usando estratégia de caracteres."""
        character_strategy = CharacterChunkingStrategy()
        return character_strategy.chunk(text, config)

    def _process_section(
        self,
        section: str,
        current_chunk: str,
        chunks: list[str],
        config: ChunkingConfig,
    ) -> str:
        """Processa uma seção do texto dividido por separador."""
        # Se seção é muito grande, divide ela separadamente
        if len(section) > config.chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""

            # Divide seção grande usando estratégia de caracteres
            large_section_chunks = self._split_large_section(section, config)
            chunks.extend(large_section_chunks)
            return ""

        # Verifica se pode adicionar seção ao chunk atual
        elif self._can_add_section(current_chunk, section, config):
            return (
                current_chunk + config.separator + section if current_chunk else section
            )

        # Se não cabe, salva chunk atual e inicia novo
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            return section

    def _can_add_section(
        self, current_chunk: str, section: str, config: ChunkingConfig
    ) -> bool:
        """Verifica se uma seção pode ser adicionada ao chunk atual."""
        if not current_chunk:
            return True

        total_length = len(current_chunk) + len(section) + len(config.separator)
        return total_length <= config.chunk_size

    def _split_large_section(self, section: str, config: ChunkingConfig) -> list[str]:
        """Divide uma seção muito grande em chunks menores."""
        chunks = []
        start = 0

        while start < len(section):
            end = start + config.chunk_size

            if end >= len(section):
                chunks.append(section[start:].strip())
                break

            # Procura quebra natural
            break_point = self._find_natural_break(section, end, config)

            chunks.append(section[start:break_point].strip())
            start = max(start + 1, break_point - config.chunk_overlap)

        return chunks

    def _find_natural_break(
        self, text: str, position: int, config: ChunkingConfig
    ) -> int:
        """Encontra uma quebra natural próxima à posição especificada."""
        search_range = config.natural_break_search_range
        search_start = max(0, position - search_range)
        search_end = min(len(text), position + search_range)

        # Procura por caracteres de quebra natural
        for i in range(search_start, search_end):
            if i < len(text) and text[i] in config.natural_break_chars:
                return i

        return position

    def _filter_valid_chunks(
        self, chunks: list[str], config: ChunkingConfig
    ) -> list[str]:
        """Remove chunks vazios ou muito pequenos."""
        return [
            chunk
            for chunk in chunks
            if chunk.strip() and len(chunk.strip()) >= config.min_chunk_size
        ]


class CharacterChunkingStrategy(ChunkingStrategy):
    """Estratégia de chunking baseada em contagem de caracteres."""

    def chunk(self, text: str, config: ChunkingConfig) -> list[str]:
        """
        Divide texto por contagem de caracteres com quebras naturais.

        Args:
            text: Texto para dividir
            config: Configuração de chunking

        Returns:
            Lista de chunks
        """
        # Validações iniciais
        if not self._is_valid_input(text, config):
            return []

        if len(text) <= config.chunk_size:
            return [text]

        return self._split_text_into_chunks(text, config)

    def _is_valid_input(self, text: str, config: ChunkingConfig) -> bool:
        """Valida entrada para chunking."""
        return bool(text and config.chunk_size > 0)

    def _split_text_into_chunks(self, text: str, config: ChunkingConfig) -> list[str]:
        """Divide texto em chunks com quebras naturais."""
        chunks = []
        start = 0

        while start < len(text):
            chunk, actual_end = self._extract_next_chunk_with_end(text, start, config)
            if chunk:
                chunks.append(chunk)

            # Calcula próximo início baseado na sobreposição
            if config.chunk_overlap > 0 and actual_end < len(text):
                next_start = max(start + 1, actual_end - config.chunk_overlap)
            else:
                next_start = actual_end

            start = next_start

        return self._filter_valid_chunks(chunks, config)

    def _extract_next_chunk_with_end(
        self, text: str, start: int, config: ChunkingConfig
    ) -> tuple[str, int]:
        """Extrai o próximo chunk e retorna também a posição final."""
        end = start + config.chunk_size

        if end >= len(text):
            return text[start:].strip(), len(text)

        # Procura quebra natural próxima ao fim
        break_point = self._find_sentence_break(text, end, config)
        return text[start:break_point].strip(), break_point

    def _extract_next_chunk(self, text: str, start: int, config: ChunkingConfig) -> str:
        """Extrai o próximo chunk do texto."""
        end = start + config.chunk_size

        if end >= len(text):
            return text[start:].strip()

        # Procura quebra natural próxima ao fim
        break_point = self._find_sentence_break(text, end, config)
        return text[start:break_point].strip()

    def _find_sentence_break(
        self, text: str, position: int, config: ChunkingConfig
    ) -> int:
        """Encontra uma quebra de sentença próxima à posição."""
        search_range = config.natural_break_search_range
        search_start = max(0, position - search_range)
        search_end = min(len(text), position + search_range)

        # Primeiro procura por caracteres de fim de sentença, de trás para frente
        for i in range(position, search_start - 1, -1):
            if i < len(text) and text[i] in config.sentence_break_chars:
                return i + 1

        # Se não encontrou, procura por quebras naturais, de trás para frente
        for i in range(position, search_start - 1, -1):
            if i < len(text) and text[i] in config.natural_break_chars:
                return i

        # Se ainda não encontrou, procura para frente
        for i in range(position, search_end):
            if i < len(text) and text[i] in config.sentence_break_chars:
                return i + 1

        for i in range(position, search_end):
            if i < len(text) and text[i] in config.natural_break_chars:
                return i

        return position

    def _calculate_next_start(
        self, current_start: int, break_point: int, overlap: int
    ) -> int:
        """Calcula a posição inicial do próximo chunk."""
        if overlap <= 0:
            return break_point

        # Evita loop infinito garantindo progresso mínimo
        next_start = break_point - overlap
        return max(current_start + 1, next_start)

    def _filter_valid_chunks(
        self, chunks: list[str], config: ChunkingConfig
    ) -> list[str]:
        """Remove chunks vazios ou muito pequenos."""
        return [
            chunk
            for chunk in chunks
            if chunk.strip() and len(chunk.strip()) >= config.min_chunk_size
        ]


class TextChunker:
    """
    Classe principal para chunking de texto usando diferentes estratégias.

    Substitui a função chunk_text() original com uma arquitetura mais modular.
    """

    def __init__(self, strategy: ChunkingStrategy | None = None):
        """
        Inicializa o chunker com uma estratégia específica.

        Args:
            strategy: Estratégia de chunking a usar (padrão: SeparatorChunkingStrategy)
        """
        self.strategy = strategy or SeparatorChunkingStrategy()

    def chunk(
        self,
        text: str | None,
        config: ChunkingConfig | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        separator: str | None = None,
    ) -> list[str]:
        """
        Divide texto em chunks usando a estratégia configurada.

        Args:
            text: Texto para dividir
            config: Configuração de chunking (prioridade sobre parâmetros individuais)
            chunk_size: Tamanho do chunk (compatibilidade com API antiga)
            chunk_overlap: Overlap entre chunks (compatibilidade com API antiga)
            separator: Separador preferencial (compatibilidade com API antiga)

        Returns:
            Lista de chunks de texto
        """
        if not text:
            return []

        # Se não foi fornecida config, cria uma com os parâmetros
        if config is None:
            # Ajusta chunk_overlap se for maior que chunk_size
            effective_chunk_size = chunk_size or 1000
            effective_chunk_overlap = chunk_overlap or 200
            if effective_chunk_overlap >= effective_chunk_size:
                effective_chunk_overlap = max(0, effective_chunk_size - 1)

            config = ChunkingConfig(
                chunk_size=effective_chunk_size,
                chunk_overlap=effective_chunk_overlap,
                separator=separator or "\n\n",
            )

        return self.strategy.chunk(text, config)

    def set_strategy(self, strategy: ChunkingStrategy):
        """
        Altera a estratégia de chunking.

        Args:
            strategy: Nova estratégia a usar
        """
        self.strategy = strategy


# Função de conveniência para manter compatibilidade com API antiga
def chunk_text(
    text: str | None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    separator: str = "\n\n",
) -> list[str]:
    """
    Função de conveniência para chunking de texto (compatibilidade).

    Esta função mantém a API original mas usa a nova arquitetura internamente.

    Args:
        text: Texto para dividir
        chunk_size: Tamanho máximo de cada chunk
        chunk_overlap: Sobreposição entre chunks
        separator: Separador preferencial

    Returns:
        Lista de chunks de texto
    """
    # Handle edge cases before config creation
    if not text or chunk_size <= 0:
        return []

    chunker = TextChunker()
    return chunker.chunk(
        text=text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator=separator,
    )
