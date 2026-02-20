from src.calc_metrics import calc_dist_metrics, calc_rep_metrics
from src.augment_data import _project, _add_noise, _mean_shift, c2, _normalize

import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from data.utils.get_dataset import get_dataset_generic
from functools import partial
from itertools import product

# TODO check
# all methods
# noise
# all datasets
# all metrics
# 

METHODS = [
    "nothing",
    "mean_shift",
    "c2",
    "linear",
    "nonlinear",
    "linear_noisy",
    "nonlinear_noisy",
]

AC_VERSIONS = [
    "booming",
    "bright",
    "deep",
    "hard",
    "reverb",
    "rough",
    "sharp",
    "warm",
]


def method_map(method):
    if method == "nothing":
        return lambda x: x
    if method == "mean_shift":
        return _mean_shift
    if method == "c2":
        return c2
    if method in ["linear", "nonlinear", "linear_noisy", "nonlinear_noisy"]:
        return partial(_project, method=method)
    raise ValueError(f"Not supported method: {method}")


# ---- dataset loaders ----

def load_clotho_t():
    with open("data/processed/clotho/clotho_clap_embeddings.pickle", "rb") as f:
        c_t_embs, _ = pickle.load(f)
    with open("data/processed/sb/sb_clap_embeddings.pickle", "rb") as f:
        sb_t_embs, _ = pickle.load(f)
    return np.vstack([c_t_embs, sb_t_embs])


def load_clotho_a():
    with open("data/processed/clotho/clotho_clap_embeddings.pickle", "rb") as f:
        _, c_a_embs = pickle.load(f)
    with open("data/processed/sb/sb_clap_embeddings.pickle", "rb") as f: 
        _, sb_a_embs = pickle.load(f)
    return np.vstack([c_a_embs, sb_a_embs])


def load_wavcaps_clap():
    with open("data/processed/wavcaps/wv_cap_precomputed_timbre_CLAP_ac.pickle", "rb") as f:
        wavcaps, _ = pickle.load(f)
    return wavcaps


def load_wavcaps_sbert():
    with open("data/processed/wavcaps/wv_cap_precomputed_timbre_SBERT_ac.pickle", "rb") as f:
        wavcaps, _ = pickle.load(f)
    return wavcaps


def load_llm():
    with open("/nfs/guille/eecs_research/soundbendor/mccabepe/timbre_tags/data/llama/captions/synth_dataset_ONLY_AC_embs.pickle", "rb") as f:
        llm, _ = pickle.load(f)
    return llm


def load_ac(version):
    ac_data, _ = get_dataset_generic(version)
    return ac_data


def load_noise_text():
    with open("/nfs/guille/eecs_research/soundbendor/mccabepe/evo2026/text_only_timbre_classification/data/processed/noise/noise_ac.pickle", "rb") as f:  # TODO
        noise_text, _ = pickle.load(f)
    return noise_text


# ---- noise helpers ----

def add_gaussian_noise(x, rng, sigma=0.023023764):
    noise = rng.normal(loc=x.mean(axis=0), scale=sigma, size=x.shape)
    return x + noise


# ---- metrics into row dicts ----

def prefix_keys(d, prefix):
    """Return a new dict with keys like f'{prefix}{k}'."""
    return {f"{prefix}{k}": v for k, v in d.items()}



def compute_rows_for_combination(
    text_name,
    audio_name,
    method_name,
    text_embs,
    audio_embs,
    rng,
    k,
    gauss_sigma,
    
):
    aug_fn = method_map(method_name)
    rows = []

    # ----- base (no extra noise) -----
    t_aug = aug_fn(text_embs)

    base_dist = calc_dist_metrics(t_aug, audio_embs)      # dict
    base_rep = calc_rep_metrics(t_aug, audio_embs)        # dict
    base_extra = calc_dist_metrics(
        _normalize(t_aug), _normalize(audio_embs) #TODO
    )                                                

    row_base = {
        "text_dataset": text_name,
        "audio_dataset": audio_name,
        "method": method_name,
        "run_type": "base",
        "noisy_iter": -1,
    }
    row_base.update(prefix_keys(base_dist, "dist_"))
    row_base.update(prefix_keys(base_rep, "rep_"))
    row_base.update(prefix_keys(base_extra, "extra_")) # TODO
    rows.append(row_base)

    # ----- noisy runs -----
    for i in range(k):
        t_noisy = aug_fn(text_embs)
        

        t_noisy = add_gaussian_noise(t_noisy, rng, sigma=gauss_sigma)

        dist_n = calc_dist_metrics(t_noisy, audio_embs)       # dict
        rep_n = calc_rep_metrics(t_noisy, audio_embs)         # dict
        extra_n = calc_dist_metrics(
            _normalize(t_noisy), _normalize(audio_embs)
        )                                                  #TODO

        row_noisy = {
            "text_dataset": text_name,
            "audio_dataset": audio_name,
            "method": method_name,
            "run_type": "noisy",
            "noisy_iter": i,
        }
        row_noisy.update(prefix_keys(dist_n, "dist_"))
        row_noisy.update(prefix_keys(rep_n, "rep_"))
        row_noisy.update(prefix_keys(extra_n, "extra_")) # TODO
        rows.append(row_noisy)

    return rows



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1066)
    parser.add_argument("--gauss_sigma", type=float, default=0.1)#  TODO
    parser.add_argument("--out_csv", type=str, default="/results/metrics/metrics.csv")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    if args.out_csv is not None:
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_csv = None

    text_loaders = {
        "clotho+sb_text": load_clotho_t,
        "wavcaps_clap": load_wavcaps_clap,
        "wavcaps_sbert": load_wavcaps_sbert,
        "llm": load_llm,
        "noise": load_noise_text
    }

    audio_loaders = {
        "clotho+sb_audio": load_clotho_a,
        **{f"ac_{v}": partial(load_ac, v) for v in AC_VERSIONS},
    }

    all_rows = []

    for (t_name, t_loader), (a_name, a_loader), method in product(
        text_loaders.items(), audio_loaders.items(), METHODS
    ):
        print(f"Running text={t_name}, audio={a_name}, method={method}")

        text_embs = t_loader()
        audio_embs = a_loader()

        rows = compute_rows_for_combination(
            text_name=t_name,
            audio_name=a_name,
            method_name=method,
            text_embs=text_embs,
            audio_embs=audio_embs,
            rng=rng,
            k=args.k,
            gauss_sigma=args.gauss_sigma,
            mix_scale=args.mix_scale,
        )
        all_rows.extend(rows)

        del text_embs, audio_embs

    df = pd.DataFrame(all_rows)
    if out_csv is not None:
        df.to_csv(out_csv, index=False)

    print(f"Saved dataframe with {len(df)} rows to {out_csv}")



if __name__ == "__main__":
    main()
