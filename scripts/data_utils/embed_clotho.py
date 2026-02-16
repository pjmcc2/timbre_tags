# AI Code
from __future__ import annotations

from pathlib import Path
from typing import Tuple
import pandas as pd


def find_wavs_df(root_dir: str | Path) -> pd.DataFrame:
    """
    Recursively find all .wav files under root_dir and return a DataFrame.

    """
    root = Path(root_dir).expanduser().resolve()
    wav_paths = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".wav"]

    records = []
    for p in wav_paths:
        try:
            records.append(
                {
                    "path": str(p),
                    "name": p.name,
                    
                }
            )
        except OSError:
            # Skip unreadable files
            continue

    df = pd.DataFrame.from_records(records)
    # In case directory is empty
    if df.empty:
        return pd.DataFrame(
            columns=["path", "name"]
        )
    return df


def _normalize_filename_series(s: pd.Series) -> pd.Series:
    """
    Normalize a series of filenames for robust matching:
    - convert to string
    - lowercase
    - strip whitespace
    - remove a trailing .wav (case-insensitive) if present
    """
    s = s.astype(str).str.strip().str.lower()
    s = s.str.replace(r"\.wav$", "", regex=True)  # remove only trailing .wav
    return s


def prepare_matches(
    found_df: pd.DataFrame,
    existing_df: pd.DataFrame,
    existing_name_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Merge discovered wav files with an existing DataFrame that has a column of file names.

    Args:
        found_df: DataFrame returned by find_wavs_df(...)
        existing_df: DataFrame with a column of filenames (with or without extension)
        existing_name_col: name of the column in existing_df that contains the filenames

    Returns:
        matched_df: INNER JOIN of existing_df with found_df on normalized name/stem.
                    Includes all original columns from existing_df + file path metadata.
        missing_df: rows from existing_df for which no matching wav was found.
        extras_df: rows from found_df for wav files not present in existing_df.

    Behavior:
        - Matching is case-insensitive and ignores the .wav extension on the existing list.
        - If there are duplicates on either side, result will be a many-to-many join.
    """
    if existing_name_col not in existing_df.columns:
        raise ValueError(
            f"existing_name_col='{existing_name_col}' not in existing_df columns: {existing_df.columns.tolist()}"
        )

    # Normalize keys
    found = found_df.copy()
    found["__key"] = _normalize_filename_series(found["name"])  # from name (has .wav)
    found["__key"] = _normalize_filename_series(found["__key"])  # idempotent

    existing = existing_df.copy()
    existing["__key"] = _normalize_filename_series(existing[existing_name_col])

    # Matched (inner join)
    matched_df = existing.merge(
        found.drop(columns=["stem"]),  # keep "name" with .wav to preserve true name
        on="__key",
        how="inner",
        suffixes=("", "_found"),
    ).drop(columns=["__key"])

    # Missing from found (left anti-join)
    missing_mask = ~existing["__key"].isin(found["__key"])
    missing_df = existing.loc[missing_mask].drop(columns=["__key"])

    # Extras present in filesystem but not in existing list (right anti-join)
    extras_mask = ~found["__key"].isin(existing["__key"])
    extras_df = found.loc[extras_mask].drop(columns=["__key"])

    return matched_df, missing_df, extras_df


def process_results(matched_df: pd.DataFrame, missing_df: pd.DataFrame, extras_df: pd.DataFrame) -> None:
    """
    Black box: replace with your downstream logic.
    For now, we’ll just print some quick stats.
    """
    print(f"Matched rows: {len(matched_df)}")
    print(f"Missing in filesystem: {len(missing_df)}")
    print(f"Extra .wav files (not listed): {len(extras_df)}")
    # TODO: replace with real logic, e.g.:
    # my_downstream_fn(matched_df, missing_df, extras_df)


def main():
    # === Example usage ===
    # 1) Find all wavs
    root_dir = "path/to/your/audio/root"  # e.g., "../data/audio"
    found_df = find_wavs_df(root_dir)

    # 2) Load or use your existing DataFrame that has the file names.
    #    Suppose it has a column "file_name" like "clip01" or "clip01.wav".
    #    If you're loading from CSV:
    # existing_df = pd.read_csv("path/to/filelist.csv")
    #
    # For demonstration, let's mock one up:
    existing_df = pd.DataFrame(
        {
            "file_name": ["clip01", "clip02.wav", "clip03", "does_not_exist"],
            "label": ["A", "B", "C", "D"],  # some extra columns you might have
        }
    )

    # 3) Prepare matched/filtered/merged sets
    matched_df, missing_df, extras_df = prepare_matches(
        found_df=found_df,
        existing_df=existing_df,
        existing_name_col="file_name",
    )

    # 4) Hand off to your black box
    process_results(matched_df, missing_df, extras_df)


if __name__ == "__main__":
    main()




def embed_clotho(audio_paths,captions,out_path=None):




def test_embedding_correctness():
    # sample files
    # do they line up after embedding?
    return sample_dataset
