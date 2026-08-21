from pathlib import Path

import core


def test_token_chunks_respect_budget_and_overlap(tmp_path):
    source = tmp_path / "large.py"
    source.write_text("\n".join(f"value_{i} = '{'x' * 35}'" for i in range(300)))

    chunks = core.chunk_file(str(source), chunk_size=400, chunk_overlap=60)

    assert len(chunks) > 1
    assert all(core.estimate_tokens(chunk["chunk"]) <= 450 for chunk in chunks)
    assert chunks[0]["line_end"] >= chunks[1]["line_start"]


def test_python_functions_are_structural_chunks_with_symbols(tmp_path):
    source = tmp_path / "scheduler.py"
    source.write_text(
        "def ScheduleCampaign(ctx):\n"
        "    return calculate_send_at(ctx)\n\n"
        "def other():\n"
        "    return 1\n"
    )

    chunks = core.chunk_file(str(source), chunk_size=400, chunk_overlap=60)

    schedule = next(c for c in chunks if "def ScheduleCampaign" in c["chunk"])
    assert schedule["chunk_type"] == "function"
    assert "ScheduleCampaign" in schedule["symbols"]
    assert "calculate_send_at" in schedule["symbols"]
    assert schedule["line_start"] == 1
    assert schedule["line_end"] == 2


def test_javascript_exported_object_method_is_structural_without_native_parser(tmp_path, monkeypatch):
    source = tmp_path / "plugin.js"
    source.write_text(
        "export default {\n"
        "  register(ctx) {\n"
        "    function QdrantPill() { return 'statusbar pill' }\n"
        "    return QdrantPill\n"
        "  }\n"
        "}\n"
    )

    import structural
    monkeypatch.setattr(structural, "_make_parser", lambda _lang: (_ for _ in ()).throw(
        AssertionError("JS/TS must not enter the native parser")
    ))
    chunks = core.chunk_file(str(source), chunk_size=350, chunk_overlap=60)

    register = next(chunk for chunk in chunks if "register(ctx)" in chunk["chunk"])
    assert register["chunk_type"] == "method"
    assert "QdrantPill" in register["symbols"]


def test_config_files_chunk_by_sections(tmp_path):
    source = tmp_path / "config.toml"
    source.write_text("[server]\nhost='localhost'\n\n[embedding]\nmodel='x'\n")

    chunks = core.chunk_file(str(source), chunk_size=400, chunk_overlap=60)

    assert len(chunks) == 2
    assert chunks[0]["chunk_type"] == "config_section"
    assert chunks[0]["symbols"] == ["server"]
    assert chunks[1]["symbols"] == ["embedding"]


def test_embedding_text_contains_navigation_metadata():
    text = core.build_embedding_text(
        project="emailchaser",
        rel_path="internal/campaign/scheduler.go",
        language="go",
        symbols=["ScheduleCampaign", "calculateSendAt"],
        code="func ScheduleCampaign() {}",
    )

    assert "project: emailchaser" in text
    assert "path: internal/campaign/scheduler.go" in text
    assert "language: go" in text
    assert "ScheduleCampaign" in text
    assert "code:\nfunc ScheduleCampaign" in text


def test_final_embedding_inputs_are_hard_bounded_without_truncation(tmp_path):
    source = tmp_path / "generated.py"
    marker = "END_MARKER"
    source.write_text("payload = '" + ("x" * 4000) + marker + "'\n")
    chunks = core.chunk_file(str(source), chunk_size=350, chunk_overlap=60)

    prepared = core.prepare_embedding_chunks(
        chunks,
        filepath=str(source),
        project="project",
        rel_path="generated.py",
        language="python",
    )
    texts = [core.build_embedding_text(
        project="project", rel_path="generated.py", language="python",
        symbols=chunk.get("symbols", []), code=chunk["chunk"],
    ) for chunk in prepared]

    assert len(prepared) > 1
    assert all(len(text) <= core.EMBEDDING_MAX_CHARS for text in texts)
    assert marker in "".join(chunk["chunk"] for chunk in prepared)


def test_embedding_ceiling_stays_below_observed_512_token_model_limit(tmp_path):
    """High-token-density source must stay under the live-tested safe ceiling."""
    source = tmp_path / "symbols.txt"
    source.write_text(("!@#$%^&*()_+-=[]{}|;:,.<>?/" * 80) + "END")
    chunks = core.chunk_file(str(source), chunk_size=350, chunk_overlap=60)
    prepared = core.prepare_embedding_chunks(
        chunks, filepath=str(source), project="project",
        rel_path="symbols.txt", language="text",
    )
    texts = [core.build_embedding_text(
        project="project", rel_path="symbols.txt", language="text",
        symbols=chunk.get("symbols", []), code=chunk["chunk"],
    ) for chunk in prepared]

    assert core.EMBEDDING_MAX_CHARS <= 480
    assert all(len(text) <= 480 for text in texts)
    assert "END" in "".join(chunk["chunk"] for chunk in prepared)
