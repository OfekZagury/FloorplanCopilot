"""Assemble the Hugging Face Space folder for the FloorplanCopilot demo.

Usage (from the repository root, after building the databases):
    python -m ResPlan_RAG.pipeline_code.data_ingestion.data_ingestion
    python deploy/huggingface/build_space.py <output-folder>

The output folder is a ready-to-push Space repo: app.py, requirements.txt, the
README card, the pipeline package and the prebuilt SQLite + Chroma databases
(tracked with Git LFS).
"""
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PIPELINE = REPO / "ResPlan_RAG" / "pipeline_code"

# Not needed at runtime in the Space
SKIP = shutil.ignore_patterns(
    "__pycache__", "*.pyc", "tests", "static_analysis", "snapshots_results",
    "dxf_output", "requirements*.txt", "*.dxf",
)

GITATTRIBUTES = """\
ResPlan_RAG/pipeline_code/databases/** filter=lfs diff=lfs merge=lfs -text
*.db filter=lfs diff=lfs merge=lfs -text
*.sqlite3 filter=lfs diff=lfs merge=lfs -text
*.bin filter=lfs diff=lfs merge=lfs -text
*.pickle filter=lfs diff=lfs merge=lfs -text
"""


GITIGNORE = """\
__pycache__/
*.pyc
dxf_output/
snapshots_results/
"""


def main(out_dir: str) -> None:
    out = Path(out_dir).resolve()
    databases = PIPELINE / "databases"
    if not (databases / "apartments_raw.db").exists():
        sys.exit("databases/ not found - run the data ingestion first (see README).")

    out.mkdir(parents=True, exist_ok=True)
    for name in ("app.py", "requirements.txt", "README.md"):
        shutil.copy2(HERE / name, out / name)
    (out / ".gitattributes").write_text(GITATTRIBUTES, encoding="utf-8")
    (out / ".gitignore").write_text(GITIGNORE, encoding="utf-8")

    target_pkg = out / "ResPlan_RAG" / "pipeline_code"
    if target_pkg.exists():
        shutil.rmtree(target_pkg)
    shutil.copytree(PIPELINE, target_pkg, ignore=SKIP)
    print(f"Space folder ready: {out}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
