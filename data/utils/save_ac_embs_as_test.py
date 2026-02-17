import numpy as np
import pickle



if __name__ == "__main__":

    with open("/nfs/guille/eecs_research/soundbendor/mccabepe/timbre_tags/data/llama/captions/synth_dataset_ONLY_AC_embs.pickle","rb") as f:
        embs,labs = pickle.load(f)
    embs = np.vstack(embs)
    print(embs.shape)
    labs = np.array(labs)
    print(labs.shape)
    with open("tests/data/test_precomputed.pickle", "wb") as f:
        pickle.dump((embs[:100],labs[:100]),f)
        print(f"Saving embeddings to {f}")

