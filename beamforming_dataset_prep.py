from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


IMAGE_COLUMN = "unit1_rgb"
POWER_COLUMN = "unit1_pwr_60ghz"
POSITION_COLUMN = "unit2_loc"


@dataclass(frozen=True)
class ScenarioPaths:
    scenario_root: Path
    scenario_csv: Path


def resolve_scenario_paths(data_root: Path, scenario_num: int) -> ScenarioPaths:
    scenario_csv_name = f"scenario{scenario_num}.csv"
    candidate_dirs = [
        data_root / f"Scenario{scenario_num}",
        data_root / f"scenario{scenario_num}",
        data_root / f"scenario{scenario_num}_dev",
    ]

    for scenario_root in candidate_dirs:
        scenario_csv = scenario_root / scenario_csv_name
        if scenario_csv.is_file():
            return ScenarioPaths(scenario_root=scenario_root, scenario_csv=scenario_csv)

    matches = sorted(data_root.rglob(scenario_csv_name))
    if not matches:
        raise FileNotFoundError(
            f"Could not find {scenario_csv_name} under {data_root}. "
            "Check where your extracted dataset lives."
        )

    scenario_csv = matches[0]
    return ScenarioPaths(scenario_root=scenario_csv.parent, scenario_csv=scenario_csv)


def resolve_data_file(scenario_root: Path, entry: str) -> Path:
    relative_path = str(entry).replace("\\", "/").strip()
    if relative_path.startswith("./"):
        relative_path = relative_path[2:]
    return scenario_root / relative_path


def ensure_columns(frame: pd.DataFrame, required_columns: Iterable[str], csv_path: Path) -> None:
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{csv_path} is missing required columns: {missing}. "
            f"Available columns: {list(frame.columns)}"
        )


def load_power_values(power_file: Path) -> np.ndarray:
    if not power_file.is_file():
        raise FileNotFoundError(f"Power file not found: {power_file}")

    power_values = np.loadtxt(power_file, ndmin=1)
    power_values = np.asarray(power_values).reshape(-1)
    if power_values.size == 0:
        raise ValueError(f"Power file is empty: {power_file}")
    return power_values


def load_position_values(position_file: Path) -> np.ndarray:
    if not position_file.is_file():
        raise FileNotFoundError(f"Position file not found: {position_file}")

    position_values = np.loadtxt(position_file, ndmin=1)
    position_values = np.asarray(position_values).reshape(-1)
    if position_values.size < 2:
        raise ValueError(
            f"Position file must contain at least 2 numeric values: {position_file}"
        )
    return position_values[:2]


def normalize(values: list[float]) -> list[float]:
    if not values:
        return []

    min_value = min(values)
    max_value = max(values)
    if min_value == max_value:
        return [0.0 for _ in values]

    return [(value - min_value) / (max_value - min_value) for value in values]


def create_subsampled_beams(
    frame: pd.DataFrame,
    scenario_root: Path,
    power_column: str = POWER_COLUMN,
    subsample_step: int = 2,
) -> tuple[list[int], list[int]]:
    updated_beams: list[int] = []
    original_beams: list[int] = []

    for entry in frame[power_column].tolist():
        power_file = resolve_data_file(scenario_root, entry)
        power_values = load_power_values(power_file)

        original_beams.append(int(np.argmax(power_values)) + 1)
        subsampled_values = power_values[::subsample_step]
        updated_beams.append(int(np.argmax(subsampled_values)) + 1)

    return updated_beams, original_beams


def split_dataframe(frame: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    shuffled = frame.sample(frac=1, random_state=seed).reset_index(drop=True)
    train_end = int(0.6 * len(shuffled))
    val_end = int(0.9 * len(shuffled))
    train = shuffled.iloc[:train_end].reset_index(drop=True)
    val = shuffled.iloc[train_end:val_end].reset_index(drop=True)
    test = shuffled.iloc[val_end:].reset_index(drop=True)
    return train, val, test


def save_split_files(frame: pd.DataFrame, output_dir: Path, prefix: str, seed: int) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    full_path = output_dir / f"{prefix}.csv"
    frame.to_csv(full_path, index=False)

    train, val, test = split_dataframe(frame, seed=seed)
    split_paths = {
        "full": full_path,
        "train": output_dir / f"{prefix}_train.csv",
        "val": output_dir / f"{prefix}_val.csv",
        "test": output_dir / f"{prefix}_test.csv",
    }
    train.to_csv(split_paths["train"], index=False)
    val.to_csv(split_paths["val"], index=False)
    test.to_csv(split_paths["test"], index=False)
    return split_paths


def build_image_beam_dataset(
    data_root: Path,
    processed_root: Path,
    scenario_num: int,
    seed: int,
) -> dict[str, Path]:
    paths = resolve_scenario_paths(data_root, scenario_num)
    frame = pd.read_csv(paths.scenario_csv)
    ensure_columns(frame, [IMAGE_COLUMN, POWER_COLUMN], paths.scenario_csv)

    updated_beams, original_beams = create_subsampled_beams(frame, paths.scenario_root)
    image_paths = [
        str(resolve_data_file(paths.scenario_root, entry))
        for entry in frame[IMAGE_COLUMN].tolist()
    ]

    output_frame = pd.DataFrame(
        {
            "index": np.arange(1, len(frame) + 1),
            IMAGE_COLUMN: image_paths,
            "unit1_beam_32": updated_beams,
            "unit1_beam_64": original_beams,
        }
    )

    output_dir = processed_root / f"Scenario{scenario_num}" / "image_beam"
    prefix = f"scenario{scenario_num}_image_beam"
    return save_split_files(output_frame, output_dir, prefix, seed)


def build_position_beam_dataset(
    data_root: Path,
    processed_root: Path,
    scenario_num: int,
    seed: int,
) -> dict[str, Path]:
    paths = resolve_scenario_paths(data_root, scenario_num)
    frame = pd.read_csv(paths.scenario_csv)
    ensure_columns(frame, [POSITION_COLUMN, POWER_COLUMN], paths.scenario_csv)

    updated_beams, original_beams = create_subsampled_beams(frame, paths.scenario_root)

    latitudes: list[float] = []
    longitudes: list[float] = []
    for entry in frame[POSITION_COLUMN].tolist():
        position_file = resolve_data_file(paths.scenario_root, entry)
        lat_value, lon_value = load_position_values(position_file)
        latitudes.append(float(lat_value))
        longitudes.append(float(lon_value))

    normalized_positions = [
        [lat_norm, lon_norm]
        for lat_norm, lon_norm in zip(normalize(latitudes), normalize(longitudes))
    ]

    output_frame = pd.DataFrame(
        {
            "index": np.arange(1, len(frame) + 1),
            "unit2_pos": normalized_positions,
            "unit1_beam_32": updated_beams,
            "unit1_beam_64": original_beams,
        }
    )

    output_dir = processed_root / f"Scenario{scenario_num}" / "pos_beam"
    prefix = f"scenario{scenario_num}_pos_beam"
    return save_split_files(output_frame, output_dir, prefix, seed)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Create cleaned beamforming datasets from a scenario CSV."
    )
    parser.add_argument("--scenario", type=int, required=True, help="Scenario number, for example 23")
    parser.add_argument(
        "--mode",
        choices=["image", "position", "all"],
        default="all",
        help="Which processed dataset to create",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val/test split",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=project_root / "dataset",
        help="Root folder that contains your extracted scenario data",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=project_root / "task_specific_data",
        help="Where the generated CSV files should be saved",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    processed_root = args.processed_root.resolve()

    if not data_root.exists():
        raise FileNotFoundError(
            f"Dataset root does not exist: {data_root}. "
            "Pass the correct path with --data-root."
        )

    print(f"Dataset root: {data_root}")
    print(f"Processed output root: {processed_root}")

    created_files: dict[str, dict[str, Path]] = {}

    if args.mode in {"image", "all"}:
        created_files["image"] = build_image_beam_dataset(
            data_root=data_root,
            processed_root=processed_root,
            scenario_num=args.scenario,
            seed=args.seed,
        )

    if args.mode in {"position", "all"}:
        created_files["position"] = build_position_beam_dataset(
            data_root=data_root,
            processed_root=processed_root,
            scenario_num=args.scenario,
            seed=args.seed,
        )

    print("")
    print("Created files:")
    for mode_name, paths in created_files.items():
        print(f"[{mode_name}]")
        for split_name, path in paths.items():
            print(f"  {split_name}: {path}")


if __name__ == "__main__":
    main()
