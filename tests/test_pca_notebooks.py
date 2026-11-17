import json
from pathlib import Path


def test_pca_notebooks_are_valid_and_code_cells_compile():
    repository_root = Path(__file__).resolve().parents[1]
    notebook_paths = [
        repository_root / "notebooks" / "02_pca_experiments.ipynb",
        repository_root / "notebooks" / "02b_patch_pca_experiments.ipynb",
    ]

    for notebook_path in notebook_paths:
        notebook = json.loads(notebook_path.read_text())
        assert notebook["nbformat"] == 4

        code_sources = []
        for cell_number, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue

            source = "".join(cell["source"])
            compile(source, f"{notebook_path.name}:cell-{cell_number}", "exec")
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
            code_sources.append(source)

        combined_source = "\n".join(code_sources)
        assert "np.linalg.eigh" in combined_source
        assert "sklearn.decomposition" not in combined_source



def test_every_notebook_has_valid_unique_cell_ids():
    import re

    repository_root = Path(__file__).resolve().parents[1]
    notebook_paths = sorted((repository_root / "notebooks").glob("*.ipynb"))

    assert notebook_paths

    for notebook_path in notebook_paths:
        notebook = json.loads(notebook_path.read_text())
        cell_ids = [cell.get("id") for cell in notebook["cells"]]

        assert all(
            isinstance(cell_id, str) and re.fullmatch(r"[A-Za-z0-9_-]+", cell_id)
            for cell_id in cell_ids
        ), f"Invalid cell ID in {notebook_path.name}"

        assert len(cell_ids) == len(set(cell_ids)), (
            f"Duplicate cell IDs in {notebook_path.name}"
        )
