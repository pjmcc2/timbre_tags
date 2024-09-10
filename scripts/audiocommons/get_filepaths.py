import pandas as pd
import os


if __name__ == "__main__":
    paths = []
    file_set = set()
    for root, dirs, files in os.walk("/nfs/guille/eecs_research/soundbendor/datasets/sounds_and_noise/AudioCommonsTimbre/TimbralModels_v0.2_Development/"):
            for file in files:
                if file.endswith(".wav"):
                    if file not in file_set:
                        filepath = os.path.join(root, file)
                        paths.append(filepath)
                        file_set.add(file)

    pd.to_pickle(pd.DataFrame(paths,columns=["path"]),"/nfs/guille/eecs_research/soundbendor/mccabepe/timbre_tags/data/ac_paths.pickle")