# AI Code
from __future__ import annotations

from pathlib import Path
from typing import Tuple
import pandas as pd
from src.load_dataset import _batch_encode_text_data, _load_clap
from src.load_dataset import _batch_encode_audio_paths
import torch
import pickle

import soundfile as sf




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
        found,  # keep "name" with .wav to preserve true name
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



 
def is_valid_wav(path: str | Path) -> bool:
    try:
        with sf.SoundFile(str(path)) as f:
            # Optionally read a small chunk to ensure data section is sane
            _ = f.read(frames=min(len(f), 1024))
        return True
    except Exception:
        return False

def filter_valid_audio(df: pd.DataFrame, path_col: str = "path") -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    mask = df[path_col].apply(is_valid_wav)
    return df[mask].reset_index(drop=True), df[~mask].reset_index(drop=True)



def embed_clotho(matched_df,path_col,caption_col):
    """
    embeds data with CLAP.
    """
    clap = _load_clap(device="cuda" if torch.cuda.is_available() else "cpu")
    audio_file_paths = matched_df[path_col].to_list()
    audio_file_paths = [str(p) for p in audio_file_paths]
    captions = matched_df[caption_col].to_list()
    text_embeddings = _batch_encode_text_data(captions,clap)
    audio_embeddings = _batch_encode_audio_paths(audio_file_paths,clap)

    return (text_embeddings,audio_embeddings)


def main(root_dir,existing_df_path,out_path,save=True):
    # === Example usage ===
    # 1) Find all wavs
    found_df = find_wavs_df(root_dir)

    # 2) Load or use your existing DataFrame that has the file names.
    #    Suppose it has a column "file_name" like "clip01" or "clip01.wav".
    #    If you're loading from CSV:
    # existing_df = pd.read_csv("path/to/filelist.csv")
    #
    # For demonstration, let's mock one up:
    existing_df = pd.read_csv(existing_df_path)
    
    # 3) Prepare matched/filtered/merged sets
    matched_df, missing_df, extras_df = prepare_matches(
        found_df=found_df,
        existing_df=existing_df,
        existing_name_col="file_name",
    )
    print("Lengths of matched, missing, extra: ", len(matched_df),len(missing_df), len(extras_df))
    matched_df_valid, matched_df_invalid = filter_valid_audio(matched_df, path_col="path")
    print(f"Valid WAVs: {len(matched_df_valid)}, Invalid WAVs dropped: {len(matched_df_invalid)}")

    
    embeddings = embed_clotho(matched_df_valid, "path", "caption") # tuple of text,audio embeddings

    if save:
        with open(out_path,"wb") as f:
            pickle.dump(embeddings,f)
    else:
        print(embeddings)


if __name__ == "__main__":
    main("/nfs/hpc/share/mccabepe/clotho","/nfs/hpc/share/mccabepe/clotho/captions_one_column.csv", "data/processed/clotho/clotho_clap_embeddings.pickle")

