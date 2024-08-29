import timbral_models
import pandas as pd
import os


def consilidate_wavs(directory):

    results = []
    file_set = set()
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".wav"):
                if file not in file_set:
                    filepath = os.path.join(root, file)
                    try:
                        result = timbral_models.timbral_extractor(filepath,verbose=False)
                        result["sound_file"] = filepath # changed to filepath
                        results.append(result)
                    except Exception as e:
                        print(f"File {file} caused error {e}")
                    
                    file_set.add(file)

    # Convert the results list into a DataFrame
    df = pd.DataFrame(results)
    df = df.set_index("sound_file")
    df["reverb"] = df.reverb.astype("bool")

    return df



if __name__ == "__main__":

    AUDIO_PATH = "/nfs/guille/eecs_research/soundbendor/datasets/sounds_and_noise/AudioCommonsTimbre/TimbralModels_v0.2_Development/"
    OUT_PATH = "/nfs/guille/eecs_research/soundbendor/mccabepe/timbre_tags/data/audiocommons/AC_dataset.pickle"

    tm_approx_df = consilidate_wavs(AUDIO_PATH)
    num_df = tm_approx_df.select_dtypes(include='number')
    cat_df = (num_df >= 50)
    cat_df["reverb"] = tm_approx_df.reverb
    pd.to_pickle(cat_df, OUT_PATH)




