"""
Detector de encoding de arquivos.

"""

from __future__ import annotations

from pathlib import Path

from ..config import LoaderConfig


class EncodingDetector:
    """
    Detector inteligente de encoding de arquivos.

    Usa uma combinação de heurísticas e bibliotecas para detectar
    o encoding mais provável de um arquivo.
    """

    def __init__(self, config: LoaderConfig | None = None):
        """
        Inicializa o detector com configurações.

        Args:
            config: Configuração para detecção (usa padrão se None)
        """
        self.config = config or LoaderConfig()

    def detect_and_read(self, file_path: Path) -> tuple[str, str]:
        """
        Detecta encoding e lê o conteúdo do arquivo.

        Args:
            file_path: Caminho para o arquivo

        Returns:
            Tupla (conteúdo, encoding_usado)

        Raises:
            FileNotFoundError: Se arquivo não existe
            UnicodeDecodeError: Se não conseguiu decodificar com nenhum encoding
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

        # Tenta primeiro com o encoding padrão
        try:
            content = file_path.read_text(encoding=self.config.encoding)
            return content, self.config.encoding
        except UnicodeDecodeError:
            pass

        if self.config.auto_detect_encoding:
            # Tenta com encodings da lista de fallback
            for encoding in self.config.fallback_encodings:
                try:
                    content = file_path.read_text(encoding=encoding)
                    return content, encoding
                except UnicodeDecodeError:
                    continue

            # Se ainda falhou, tenta detecção automática com chardet
            detected_encoding = self._detect_with_chardet(file_path)
            if detected_encoding:
                try:
                    content = file_path.read_text(encoding=detected_encoding)
                    return content, detected_encoding
                except UnicodeDecodeError:
                    # Failed to decode with detected encoding, continue to fallback
                    pass

            # Último recurso: lê ignorando erros
            try:
                content = file_path.read_text(errors="ignore")
                return content, "utf-8 (com erros ignorados)"
            except Exception as e:
                # Log the error but continue to raise ValueError below
                import logging

                logging.debug(
                    f"Failed to read file {file_path} with error handling: {e}"
                )

        raise ValueError(
            f"Não foi possível decodificar o arquivo {file_path} "
            f"com nenhum dos encodings tentados: {self.config.fallback_encodings}"
        )

    def detect_encoding(self, file_path: Path) -> str | None:
        """
        Detecta apenas o encoding sem ler o arquivo completo.

        Args:
            file_path: Caminho para o arquivo

        Returns:
            Encoding detectado ou None se não conseguiu detectar
        """
        if not file_path.exists():
            return None

        # Tenta detecção com chardet
        return self._detect_with_chardet(file_path)

    def _detect_with_chardet(self, file_path: Path) -> str | None:
        """
        Usa heurísticas simples para detectar encoding lendo uma amostra do arquivo.

        Args:
            file_path: Caminho para o arquivo

        Returns:
            Encoding detectado ou None
        """
        try:
            # Lê primeiros 1KB para análise
            with open(file_path, "rb") as f:
                raw_data = f.read(1024)

            if not raw_data:
                return None

            # Heurísticas simples para detecção de encoding
            # Verifica se parece ser UTF-8
            try:
                raw_data.decode("utf-8")
                return "utf-8"
            except UnicodeDecodeError:
                pass

            # Verifica se parece ser Latin-1
            try:
                raw_data.decode("latin-1")
                return "latin-1"
            except UnicodeDecodeError:
                # Failed to decode with latin-1, will return None
                pass

        except Exception as e:
            # Log the error for debugging
            import logging

            logging.debug(f"Error during encoding detection for {file_path}: {e}")

        return None

    def get_supported_encodings(self) -> list[str]:
        """
        Retorna lista de encodings suportados pelo detector.

        Returns:
            Lista de strings com nomes dos encodings
        """
        return [self.config.encoding] + self.config.fallback_encodings
