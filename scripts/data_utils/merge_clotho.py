# AI-Code
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import pandas as pd


# ----------------- CONFIG (matches your schema) -----------------
CAPTIONS_GLOB = "*captions*.csv"
METADATA_GLOB = "*metadata*.csv"

MERGE_KEYS: Sequence[str] = ["file_name"]  # explicit merge key
CAPTION_COLS: Sequence[str] = ["caption_1", "caption_2", "caption_3", "caption_4", "caption_5"]

# Outputs
FULL_CAPTIONS_OUT = "clotho_captions_full.csv"
FULL_METADATA_OUT = "clotho_metadata_full.csv"
MERGED_FULL_OUT   = "clotho_full_merged.csv"
ID_MAP_OUT        = "clotho_ids.csv"
ONE_COL_CAPTIONS  = "captions_one_column.csv"
ONE_COL_FINAL_OUT = "captions_one_column_deduped_vs_extant.csv"
DUPLICATE_IDS_OUT = "duplicates_vs_extant_ids.csv"


# ----------------- HELPERS -----------------

def _find_csvs(root: Path, glob_pattern: str) -> List[Path]:
    print(root)
    print(glob_pattern)
    return [p for p in root.glob(glob_pattern) if p.is_file() and p.suffix.lower() == ".csv"]


def combine_csvs(root_dir: str | Path, glob_pattern: str, out_name: str, force: bool = False) -> pd.DataFrame:
    """
    Combine all CSVs under root_dir that match glob_pattern into a single CSV saved as out_name.
    Reads cached output if present (and force=False).
    """
    root = Path(root_dir)
    out_path = root / out_name
    if out_path.exists() and not force:
        return pd.read_csv(out_path, low_memory=False)

    files = _find_csvs(root, glob_pattern)
    if not files:
        raise FileNotFoundError(f"No CSVs matched {glob_pattern} under {root}")
    
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False,encoding='ISO-8859-1')
            dfs.append(df)
        except Exception as e:
            print(f"ERROR ENCOUNTERED: {e}")
            print(f"Offending file: {f}")
    full = pd.concat(dfs, ignore_index=True, sort=False)
    full.to_csv(out_path, index=False)
    return full


def merge_fulls(
    captions_full: pd.DataFrame,
    metadata_full: pd.DataFrame,
    keys: Sequence[str],
    how: str = "outer",
    drop_duplicate_keys: bool = True,
) -> pd.DataFrame:
    # Align dtypes on the join keys
    for k in keys:
        captions_full[k] = captions_full[k].astype("string")
        metadata_full[k] = metadata_full[k].astype("string")

    merged = pd.merge(captions_full, metadata_full, on=list(keys), how=how, suffixes=("_cap", "_meta"))

    if drop_duplicate_keys:
        old_length = len(merged)
        print(f"Dropping duplicate keys. Previous length: {old_length}")

        merged = merged.drop_duplicates(subset=list(keys)).reset_index(drop=True)
        new_length = len(merged)
        print(f"Duplicated Dropped. New length: {new_length}")
        print(f"Total duplicates dropped: {old_length - new_length}")
    return merged


def make_id_map(merged: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    """
    Save id-mapping with columns: file_name, sound_id (if both exist).
    """
    cols_needed = ["file_name", "sound_id"]
    missing = [c for c in cols_needed if c not in merged.columns]
    if missing:
        raise KeyError(f"ID map requires columns {cols_needed}. Missing: {missing}")
    id_map = merged.loc[:, cols_needed].drop_duplicates().reset_index(drop=True) # IDK if I want this? didn't we already drop columns
    id_map.to_csv(out_path, index=False)
    return id_map


def captions_wide_to_long_with_ids(
    merged: pd.DataFrame,
    caption_cols: Sequence[str],
    id_cols: Sequence[str] = ("file_name", "sound_id"),
    out_caption_col: str = "caption",
) -> pd.DataFrame:
    """
    Unpivot N caption columns to a single column and retain IDs for filtering.
    Result columns: [file_name, sound_id, caption]
    """
    # Ensure caption columns exist
    caps = [c for c in caption_cols if c in merged.columns]
    if not caps:
        raise KeyError(f"None of the specified caption columns found: {caption_cols}")

    # Ensure id columns exist (we need sound_id for extant filtering)
    ids = [c for c in id_cols if c in merged.columns]
    if "sound_id" not in ids:
        raise KeyError("The merged data must contain 'sound_id' for external duplicate filtering.")

    long = merged.loc[:, list(ids) + caps].copy()

    # Melt: keep IDs as id_vars, captions go into one value column
    long = long.melt(id_vars=list(ids), value_vars=caps, value_name=out_caption_col)

    # Clean: drop blanks/NaN captions, trim
    long[out_caption_col] = long[out_caption_col].astype("string").str.strip()
    long = long[long[out_caption_col].notna() & (long[out_caption_col] != "")]
    # Optional: drop exact duplicate captions regardless of which file_id they came from
    long = long.drop_duplicates(subset=[out_caption_col]).reset_index(drop=True)

    # Keep only IDs + caption
    return long.loc[:, list(ids) + [out_caption_col]].reset_index(drop=True)


def filter_by_extant_sound_ids(
    long_with_ids: pd.DataFrame,
    extant_sound_id_csv: str | Path,
    id_col: str = "sound_id",
    caption_col: str = "caption",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove rows whose sound_id appears in the external one-column CSV.
    Returns:
        (overlap_ids_df, filtered_captions_one_col_df)
    """
    extant = pd.read_csv(extant_sound_id_csv, low_memory=False)
    if id_col not in extant.columns:
        raise KeyError(f"External CSV must contain '{id_col}' column.")

    # Normalize IDs
    long_with_ids[id_col] = long_with_ids[id_col].astype("string").str.strip()
    extant[id_col] = extant[id_col].astype("string").str.strip()

    extant_ids = set(extant[id_col].dropna())
    # Overlap report (which IDs are in both)
    overlap_ids = long_with_ids[long_with_ids[id_col].isin(extant_ids)].loc[:, [id_col]].drop_duplicates()

    # Keep only captions where sound_id not in extant
    filtered = long_with_ids[~long_with_ids[id_col].isin(extant_ids)].copy()
    # Output: one-column of captions (unique)
    filtered_one_col = filtered.loc[:, [caption_col]].drop_duplicates().reset_index(drop=True)

    return overlap_ids.reset_index(drop=True), filtered_one_col


# ----------------- ORCHESTRATION -----------------

def run_pipeline(
    root_dir: str | Path,
    *,
    captions_glob: str = CAPTIONS_GLOB,
    metadata_glob: str = METADATA_GLOB,
    merge_keys: Sequence[str] = MERGE_KEYS,
    caption_cols: Sequence[str] = CAPTION_COLS,
    extant_sound_id_csv: Optional[str | Path] = None,
    force_rebuild_fulls: bool = False,
) -> dict:
    """
    1) Combine captions -> clotho_captions_full.csv
    2) Combine metadata -> clotho_metadata_full.csv
    3) Merge on file_name -> clotho_full_merged.csv
    4) Save ID map -> clotho_ids.csv
    5) Unpivot captions to [file_name, sound_id, caption]
    6) Save pre-filter one-column captions -> captions_one_column.csv
    7) If extant sound_id CSV provided, remove those IDs and save:
         - duplicates_vs_extant_ids.csv  (overlap report)
         - captions_one_column_deduped_vs_extant.csv (final one-column captions)
    """
    root = Path(root_dir)

    # 1–2: combine
    captions_full = combine_csvs(root, captions_glob, FULL_CAPTIONS_OUT, force=force_rebuild_fulls)
    metadata_full = combine_csvs(root, metadata_glob, FULL_METADATA_OUT, force=force_rebuild_fulls)

    # 3: merge
    merged = merge_fulls(captions_full, metadata_full, keys=merge_keys, how="outer", drop_duplicate_keys=True)
    merged.to_csv(root / MERGED_FULL_OUT, index=False)

    # 4: ID map
    id_map = make_id_map(merged, root / ID_MAP_OUT)

    # 5: unpivot to long (keep IDs to allow filtering by external sound_id list)
    long_with_ids = captions_wide_to_long_with_ids(merged, caption_cols=caption_cols, id_cols=("file_name", "sound_id"))

    # 6: write one-column (pre-filter) for reference
    one_col = long_with_ids.loc[:, ["caption"]].drop_duplicates().reset_index(drop=True)
    one_col.to_csv(root / ONE_COL_CAPTIONS, index=False)

    result = {
        "captions_full": str(root / FULL_CAPTIONS_OUT),
        "metadata_full": str(root / FULL_METADATA_OUT),
        "merged_full": str(root / MERGED_FULL_OUT),
        "id_map": str(root / ID_MAP_OUT),
        "one_column_captions": str(root / ONE_COL_CAPTIONS),
        "duplicates_vs_extant_ids": None,
        "one_column_captions_minus_extant": None,
    }

    # 7: filter by external sound_id file, if provided
    if extant_sound_id_csv:
        overlap_ids, filtered_one_col = filter_by_extant_sound_ids(long_with_ids, extant_sound_id_csv)
        overlap_ids.to_csv(root / DUPLICATE_IDS_OUT, index=False)
        filtered_one_col.to_csv(root / ONE_COL_FINAL_OUT, index=False)

        result["duplicates_vs_extant_ids"] = str(root / DUPLICATE_IDS_OUT)
        result["one_column_captions_minus_extant"] = str(root / ONE_COL_FINAL_OUT)

    return result


if __name__ == "__main__":
    # Example usage — edit paths as needed:
    outputs = run_pipeline(
        root_dir="/nfs/hpc/share/mccabepe/clotho",
        captions_glob="*captions*.csv",
        metadata_glob="*metadata*.csv",
        merge_keys=["file_name"],
        caption_cols=["caption_1", "caption_2", "caption_3", "caption_4", "caption_5"],
        extant_sound_id_csv="fsd_ids_to_use.txt",  # set to None to skip
        force_rebuild_fulls=False,
    )
    print(outputs)




