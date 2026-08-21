import subprocess
import sys
from pathlib import Path


def test_structural_chunking_is_safe_under_threaded_index_pipeline():
    root = Path(__file__).resolve().parent.parent
    script = r'''
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import core
root = Path.cwd()
files = [str(p) for p in root.glob("*.py")] + [str(p) for p in (root / "tests").glob("*.py")]
with ThreadPoolExecutor(max_workers=8) as pool:
    futures = [pool.submit(core.chunk_file, path) for _ in range(12) for path in files]
    for future in futures:
        assert isinstance(future.result(), list)
print("THREAD_PARSE_OK")
'''
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        text=True,
        capture_output=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert "THREAD_PARSE_OK" in result.stdout
