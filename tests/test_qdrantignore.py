"""Tests for per-project .qdrantignore support in discover_files."""
from pathlib import Path

import core


def test_discover_files_respects_qdrantignore(tmp_path):
    (tmp_path / "keep.txt").write_text("keep")
    skip = tmp_path / "skip"
    skip.mkdir()
    (skip / "junk.txt").write_text("junk")
    (tmp_path / ".qdrantignore").write_text("skip\n")

    files = core.discover_files(str(tmp_path))

    assert [Path(f).name for f in files] == ["keep.txt"]


def test_discover_files_nested_qdrantignore(tmp_path):
    sub = tmp_path / "src" / "gen"
    sub.mkdir(parents=True)
    (sub / "gen.txt").write_text("gen")
    (tmp_path / "src" / "real.py").write_text("x")
    (tmp_path / ".qdrantignore").write_text("# generated code\nsrc/gen\n")

    files = core.discover_files(str(tmp_path))

    assert len(files) == 1
    assert files[0].endswith("real.py")


def test_discover_files_no_ignore_file_unchanged(tmp_path):
    (tmp_path / "a.txt").write_text("a")
    sub = tmp_path / "b"
    sub.mkdir()
    (sub / "c.txt").write_text("c")

    files = core.discover_files(str(tmp_path))

    assert len(files) == 2
