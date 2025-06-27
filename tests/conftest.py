"""
Testes para configuração do pytest.
"""

import sys
from pathlib import Path

import pytest

# Adiciona o diretório raiz do projeto ao Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configurações globais do pytest
pytest_plugins = []
