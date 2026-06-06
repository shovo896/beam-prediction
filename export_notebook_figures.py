import base64
import json
from pathlib import Path


NOTEBOOK = Path("beam_predict_final_paper.ipynb")
OUTPUT_DIR = Path("figures/notebook_diagnostics")

CELL_NAMES = {
    45: "initial_image_model_test_accuracy",
    67: "position_training_loss",
    69: "position_validation_accuracy",
    71: "position_test_accuracy",
    73: "position_confusion_matrix",
    85: "position_height_training_validation_test",
    86: "position_height_confusion_matrix",
    94: "position_height_distance_training_validation_test",
    95: "position_height_distance_confusion_matrix",
    97: "position_modality_test_accuracy_comparison",
}


def main():
    notebook = json.loads(NOTEBOOK.read_text())
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    exported = 0
    for cell_index, name in CELL_NAMES.items():
        outputs = notebook["cells"][cell_index].get("outputs", [])
        for output in outputs:
            image_data = output.get("data", {}).get("image/png")
            if image_data is None:
                continue
            if isinstance(image_data, list):
                image_data = "".join(image_data)
            path = OUTPUT_DIR / f"{name}.png"
            path.write_bytes(base64.b64decode(image_data))
            exported += 1
            print(f"Saved {path}")
            break

    if exported != len(CELL_NAMES):
        raise RuntimeError(
            f"Expected {len(CELL_NAMES)} notebook figures, exported {exported}."
        )


if __name__ == "__main__":
    main()
