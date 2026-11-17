import csv
from pathlib import Path


def read_csv(relative_path):
    repository_root = Path(__file__).resolve().parents[1]
    file_path = repository_root / relative_path

    with file_path.open(newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        headers = reader.fieldnames

    assert headers is not None
    assert all(
        None not in row and all(value is not None for value in row.values())
        for row in rows
    )
    return headers, rows


def macro_values():
    _, rows = read_csv("results/tables/memae_macro.csv")
    return {row["metric"]: float(row["macro_mean"]) for row in rows}


def test_patchcore_category_and_macro_tables_are_consistent():
    category_headers, categories = read_csv("results/tables/patchcore_anomalib_categories.csv")
    macro_headers, macro_rows = read_csv("results/tables/patchcore_anomalib_macro.csv")

    assert len(categories) == 15
    assert {"category", "runtime_minutes", "image_AUROC", "image_F1Score", "pixel_AUROC", "pixel_F1Score"} <= set(category_headers)
    assert len(macro_rows) == 4

    recorded_macro = {row["metric"]: float(row["macro_mean"]) for row in macro_rows}

    for metric in ["image_AUROC", "image_F1Score", "pixel_AUROC", "pixel_F1Score"]:
        category_mean = sum(float(row[metric]) for row in categories) / len(categories)
        assert abs(category_mean - recorded_macro[metric]) < 1e-5

    recorded_runtime = sum(float(row["runtime_minutes"]) for row in categories)
    assert abs(recorded_runtime - 38.99) < 0.01


def test_memae_category_table_is_rectangular_and_f1_is_not_mislabelled():
    category_headers, categories = read_csv("results/tables/memae_categories.csv")
    assert len(categories) == 15
    assert "image_AP" not in category_headers
    assert "image_F1Score" not in category_headers
    assert "pixel_F1Score" not in category_headers
    assert "f1_score_field_unverified" in category_headers

    recorded_macro = macro_values()
    for metric in ["image_AUROC", "pixel_AUROC", "pixel_AP"]:
        category_mean = sum(float(row[metric]) for row in categories) / len(categories)
        assert abs(category_mean - recorded_macro[metric]) < 1e-5

    recorded_runtime = sum(float(row["runtime_minutes"]) for row in categories)
    assert abs(recorded_runtime - 54.97) < 0.01
