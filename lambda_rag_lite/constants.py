"""
Constantes centralizadas para tipos de arquivo e extensões.

Centraliza as definições de tipos de arquivo e extensões para eliminar
duplicação de código entre os módulos.
"""

# Extensões de arquivo por categoria - texto e documentos
_TEXT_EXTS = [
    ".txt",
    ".md",
    ".py",
    ".js",
    ".ts",
    ".json",
    ".yaml",
    ".yml",
    ".html",
    ".htm",
    ".css",
    ".sql",
    ".sh",
    ".bash",
    ".zsh",
    ".csv",
    ".log",
    ".conf",
    ".cfg",
    ".ini",
    ".xml",
    ".rst",
    ".tex",
]
TEXT_EXTENSIONS: set[str] = set(_TEXT_EXTS)

_MARKDOWN_EXTS = [".md", ".markdown", ".mdown", ".mkd", ".mkdn"]
MARKDOWN_EXTENSIONS: set[str] = set(_MARKDOWN_EXTS)

# Tipos de arquivo por categoria - linguagens de programação
_PROGRAMMING_LANGS = [
    "python",
    "javascript",
    "typescript",
    "java",
    "c",
    "cpp",
    "csharp",
    "php",
    "ruby",
    "golang",
    "rust",
    "swift",
    "kotlin",
    "scala",
    "r",
    "matlab",
    "perl",
    "shell",
    "bash",
    "zsh",
    "fish",
    "powershell",
    "batch",
]
CODE_TYPES: set[str] = set(_PROGRAMMING_LANGS)

# Tipos de documentos e configurações
_DOC_TYPES = [
    "markdown",
    "text",
    "restructured_text",
    "latex",
    "html",
    "readme",
    "changelog",
    "license",
]
DOCUMENT_TYPES: set[str] = set(_DOC_TYPES)

_DATA_FORMATS = ["json", "yaml", "toml", "csv", "tsv", "sql", "xml"]
DATA_TYPES: set[str] = set(_DATA_FORMATS)

_CONFIG_FILES = [
    "ini",
    "config",
    "gitignore",
    "dockerignore",
    "editorconfig",
    "requirements",
    "package_config",
    "composer_config",
    "cargo_config",
    "python_project",
]
CONFIG_TYPES: set[str] = set(_CONFIG_FILES)

# Mapeamento de extensões para tipos de arquivo
EXTENSION_TYPE_MAPPING = {
    # Documentos de texto
    ".md": "markdown",
    ".markdown": "markdown",
    ".mdown": "markdown",
    ".mkd": "markdown",
    ".mkdn": "markdown",
    ".txt": "text",
    ".rst": "restructured_text",
    ".tex": "latex",
    # Código
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".jsx": "javascript_react",
    ".tsx": "typescript_react",
    ".java": "java",
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c_header",
    ".hpp": "cpp_header",
    ".cs": "csharp",
    ".php": "php",
    ".rb": "ruby",
    ".go": "golang",
    ".rs": "rust",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
    ".r": "r",
    ".m": "matlab",
    ".pl": "perl",
    ".sh": "shell",
    ".bash": "bash",
    ".zsh": "zsh",
    ".fish": "fish",
    ".ps1": "powershell",
    ".bat": "batch",
    ".cmd": "batch",
    # Web
    ".html": "html",
    ".htm": "html",
    ".xml": "xml",
    ".css": "css",
    ".scss": "scss",
    ".sass": "sass",
    ".less": "less",
    # Dados
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".ini": "ini",
    ".cfg": "config",
    ".conf": "config",
    ".csv": "csv",
    ".tsv": "tsv",
    ".sql": "sql",
    # Logs e documentação
    ".log": "log",
    ".out": "output",
    ".err": "error_log",
    ".diff": "diff",
    ".patch": "patch",
    # Outros
    ".gitignore": "gitignore",
    ".dockerignore": "dockerignore",
    ".editorconfig": "editorconfig",
}

# Arquivos especiais por nome
SPECIAL_FILES = {
    "readme": "readme",
    "changelog": "changelog",
    "license": "license",
    "dockerfile": "dockerfile",
    "makefile": "makefile",
    "rakefile": "rakefile",
    "gemfile": "gemfile",
    "requirements.txt": "requirements",
    "setup.py": "setup_script",
    "package.json": "package_config",
    "composer.json": "composer_config",
    "cargo.toml": "cargo_config",
    "pyproject.toml": "python_project",
}

# Mapeamento de MIME types para tipos internos
MIME_TYPE_MAPPING = {
    "text/plain": "text",
    "text/markdown": "markdown",
    "text/html": "html",
    "text/css": "css",
    "text/javascript": "javascript",
    "text/xml": "xml",
    "application/json": "json",
    "application/xml": "xml",
    "application/javascript": "javascript",
    "application/sql": "sql",
}
