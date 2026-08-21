"""Structural-first source chunking with token-bounded fallback."""
from __future__ import annotations

import re
import threading
import ast

from pathlib import Path

try:
    from .chunking import token_chunks_from_text
except ImportError:
    from chunking import token_chunks_from_text

LANGUAGE_BY_EXT = {
    ".py": "python", ".js": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "tsx", ".go": "go", ".rs": "rust",
    ".java": "java",
}

NODE_TYPES = {
    "function_definition": "function", "function_declaration": "function",
    "function_item": "function", "method_definition": "method",
    "method_declaration": "method", "method_declaration": "method",
    "class_definition": "class", "class_declaration": "class",
    "class_specifier": "class", "struct_item": "struct",
    "struct_specifier": "struct", "type_declaration": "struct",
    "interface_declaration": "interface", "interface_type": "interface",
    "function_declaration": "function", "generator_function_declaration": "function",
}

CONFIG_EXTS = {".toml", ".ini", ".cfg", ".conf", ".env", ".yaml", ".yml", ".json"}
_PARSER_LOCK = threading.Lock()  # serialize native tree traversal
_PARSER_LOCAL = threading.local()  # native Parser objects are thread-affine
KEYWORDS = {
    "def", "func", "function", "class", "type", "struct", "interface",
    "return", "const", "let", "var", "true", "false", "none", "null",
}


def extract_symbols(text: str) -> list[str]:
    definitions = re.findall(
        r"\b(?:def|class|func|function|type|struct|interface)\s+([A-Za-z_][A-Za-z0-9_]*)",
        text,
    )
    identifiers = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", text)
    useful = definitions + [
        value for value in identifiers
        if value not in KEYWORDS and ("_" in value or re.search(r"[a-z][A-Z]", value))
    ]
    out = []
    for value in useful:
        if value not in out:
            out.append(value)
        if len(out) >= 20:
            break
    return out


def _make_parser(language: str):
    """Build a parser from official per-language bindings (no language-pack)."""
    from tree_sitter import Language, Parser
    if language == "python":
        import tree_sitter_python as binding
        capsule = binding.language()
    elif language == "javascript":
        import tree_sitter_javascript as binding
        capsule = binding.language()
    elif language in {"typescript", "tsx"}:
        import tree_sitter_typescript as binding
        capsule = binding.language_tsx() if language == "tsx" else binding.language_typescript()
    elif language == "go":
        import tree_sitter_go as binding
        capsule = binding.language()
    elif language == "rust":
        import tree_sitter_rust as binding
        capsule = binding.language()
    elif language == "java":
        import tree_sitter_java as binding
        capsule = binding.language()
    else:
        raise LookupError(language)
    return Parser(Language(capsule))


def _config_chunks(filepath: str, text: str, chunk_tokens: int,
                   overlap_tokens: int) -> list[dict]:
    ext = Path(filepath).suffix.lower()
    lines = text.splitlines()
    starts = []
    for index, line in enumerate(lines):
        match = None
        if ext in {".toml", ".ini", ".cfg", ".conf"}:
            match = re.match(r"^\s*\[+([^\]]+)\]+\s*$", line)
        elif ext in {".yaml", ".yml"}:
            match = re.match(r"^([A-Za-z_][A-Za-z0-9_.-]*):(?:\s|$)", line)
        elif ext == ".env":
            match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)=", line)
        if match:
            starts.append((index, match.group(1)))
    if not starts:
        return []
    chunks = []
    for pos, (start, symbol) in enumerate(starts):
        end = starts[pos + 1][0] if pos + 1 < len(starts) else len(lines)
        section = "\n".join(lines[start:end]).rstrip()
        chunks.extend(token_chunks_from_text(
            section, filepath=filepath, chunk_tokens=chunk_tokens,
            overlap_tokens=overlap_tokens, base_line=start + 1,
            chunk_type="config_section", symbols=[symbol],
        ))
    return chunks


def _python_ast_chunks(filepath: str, text: str, chunk_tokens: int,
                       overlap_tokens: int) -> list[dict]:
    """Stable structural Python chunks without native parser bindings."""
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return []
    lines = text.splitlines()
    chunks = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            chunk_type = "function"
        elif isinstance(node, ast.ClassDef):
            chunk_type = "class"
        else:
            continue
        start = max(1, int(node.lineno))
        end = max(start, int(getattr(node, "end_lineno", start)))
        code = "\n".join(lines[start - 1:end])
        chunks.extend(token_chunks_from_text(
            code, filepath=filepath, chunk_tokens=chunk_tokens,
            overlap_tokens=overlap_tokens, base_line=start,
            chunk_type=chunk_type, symbols=extract_symbols(code),
        ))
    for index, chunk in enumerate(chunks):
        chunk["chunk_index"] = index
    return chunks


def structural_chunks(filepath: str, chunk_tokens: int = 400,
                      overlap_tokens: int = 60) -> list[dict]:
    # py-tree-sitter native grammars can crash when invoked from worker
    # threads on macOS. Returning [] selects the safe token fallback. The
    # plugin's asyncio index pipeline calls this on the main/event-loop thread,
    # so normal indexing still receives structural chunks.
    if threading.current_thread() is not threading.main_thread():
        return []
    try:
        source = Path(filepath).read_bytes()
    except OSError:
        return []
    text = source.decode("utf-8", errors="ignore")
    ext = Path(filepath).suffix.lower()
    if ext == ".py":
        return _python_ast_chunks(filepath, text, chunk_tokens, overlap_tokens)
    if ext in CONFIG_EXTS:
        config = _config_chunks(filepath, text, chunk_tokens, overlap_tokens)
        if config:
            for index, chunk in enumerate(config):
                chunk["chunk_index"] = index
            return config

    language = LANGUAGE_BY_EXT.get(ext)
    if not language:
        return []
    try:
        # Cache one official parser per language and copy primitive node
        # offsets while under the lock so no Tree/Node crosses workers.
        with _PARSER_LOCK:
            parsers = getattr(_PARSER_LOCAL, "parsers", None)
            if parsers is None:
                parsers = {}
                _PARSER_LOCAL.parsers = parsers
            parser = parsers.get(language)
            if parser is None:
                parser = _make_parser(language)
                parsers[language] = parser
            tree = parser.parse(source)
            node_specs = []

            def capture(node):
                chunk_type = NODE_TYPES.get(node.type)
                if not chunk_type:
                    return False
                node_specs.append((
                    node.start_byte, node.end_byte,
                    node.start_point.row, chunk_type,
                ))
                return True

            # Top-level semantic units plus bounded wrapper traversal for
            # patterns such as `export default { register(ctx) { ... } }`.
            # Depth 3 is enough for export/object wrappers without materializing
            # the entire native tree.
            stack = [(node, 0) for node in reversed(tree.root_node.children)]
            while stack:
                node, depth = stack.pop()
                if capture(node):
                    continue
                if depth < 3:
                    stack.extend((child, depth + 1) for child in reversed(node.children))

    except Exception:
        return []
    chunks = []
    for start_byte, end_byte, start_row, chunk_type in node_specs:
        code = source[start_byte:end_byte].decode("utf-8", errors="ignore")
        symbols = extract_symbols(code)
        chunks.extend(token_chunks_from_text(
            code, filepath=filepath, chunk_tokens=chunk_tokens,
            overlap_tokens=overlap_tokens, base_line=start_row + 1,
            chunk_type=chunk_type, symbols=symbols,
        ))
    for index, chunk in enumerate(chunks):
        chunk["chunk_index"] = index
    return chunks
