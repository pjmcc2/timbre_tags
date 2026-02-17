
import pandas as pd
import soundfile
import os
import re          
from itertools import combinations
import pickle
import json

AC_LABELS = ["booming","bright","deep","hard","reverb","rough","sharp","warm"]

def _remove_overlaps(df, json_path='/nfs/guille/eecs_research/soundbendor/datasets/sounds_and_noise/wavcaps/json_files/fsd_final.json', id_column='freesound_id'):
    try:
        with open(json_path, 'r') as f:
            json_data = json.load(f)

        json_ids = {entry['id'] for entry in json_data.get('data', [])}
        cleaned_df = df[~df[id_column].isin(json_ids)].reset_index(drop=True)
        return cleaned_df

    except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
        print(f"Error processing JSON file: {e}")
        return df.copy()




def get_dataset():
    raw = pd.read_pickle("/nfs/stak/users/mccabepe/research_folder/timbre_tags/data/audiocommons/single sets/ACTD_Brightness.pickle")
    numeric = raw.drop(["sound_file"],axis=1)
    mean_value = numeric.mean(axis=1)
    raw["mean"] = mean_value
    raw["label"] = [1 if mean_value[i] > 50 else 0 for i in range(len(mean_value))]


    path_loc = "/nfs/guille/eecs_research/soundbendor/datasets/sounds_and_noise/AudioCommonsTimbre/TimbralModels_v0.2_Development/Brightness/Select Stimuli/Listening tests/Stimuli"
    paths = [os.path.join(path_loc,file) for file in raw.sound_file]
    raw['path'] = paths
    raw = raw[raw["path"] != "/nfs/guille/eecs_research/soundbendor/datasets/sounds_and_noise/AudioCommonsTimbre/TimbralModels_v0.2_Development/Brightness/Select Stimuli/Listening tests/Stimuli/soundscape-default-3-218564"]

    return raw


def get_dataset_generic(key):
    
    DF_PATHS = {
        "booming":"/nfs/guille/eecs_research/soundbendor/mccabepe/timbre_tags/data/audiocommons/single sets/ACTD_Boominess.pickle",
        "bright":"/nfs/guille/eecs_research/soundbendor/mccabepe/timbre_tags/data/audiocommons/single sets/ACTD_Brightness.pickle",
        "deep":"/nfs/guille/eecs_research/soundbendor/mccabepe/timbre_tags/data/audiocommons/single sets/ACTD_Depth.pickle",
        "hard":"/nfs/guille/eecs_research/soundbendor/mccabepe/timbre_tags/data/audiocommons/single sets/ACTD_Hardness.pickle",
        "reverb":"/nfs/guille/eecs_research/soundbendor/mccabepe/timbre_tags/data/audiocommons/single sets/reverb_standardized.pickle",
        "rough":"/nfs/guille/eecs_research/soundbendor/mccabepe/timbre_tags/data/audiocommons/single sets/ACTD_Roughness.pickle",
        "sharp":"/nfs/guille/eecs_research/soundbendor/mccabepe/timbre_tags/data/audiocommons/single sets/ACTD_Sharpness.pickle",
        "warm":"/nfs/guille/eecs_research/soundbendor/mccabepe/timbre_tags/data/audiocommons/single sets/ACTD_Warmth.pickle"
    }
    OLD_OLD_FILE_PATHS = {
        "booming":"/nfs/guille/eecs_research/soundbendor/datasets/sounds_and_noise/AudioCommonsTimbre/TimbralModels_v0.2_Development/Boominess/Select Stimuli/Listening tests/Stimuli/",
        "bright": "/nfs/guille/eecs_research/soundbendor/datasets/sounds_and_noise/AudioCommonsTimbre/TimbralModels_v0.2_Development/Brightness/Select Stimuli/Listening tests/Stimuli/",
        "deep":"/nfs/guille/eecs_research/soundbendor/datasets/sounds_and_noise/AudioCommonsTimbre/TimbralModels_v0.2_Development/Depth/Select Stimuli/Listening tests/Stimuli/",
        "hard":"/nfs/guille/eecs_research/soundbendor/datasets/sounds_and_noise/AudioCommonsTimbre/TimbralModels_v0.2_Development/Hardness/Select Stimuli/Listening tests/Stimuli",
        "reverb":"/nfs/guille/eecs_research/soundbendor/datasets/sounds_and_noise/AudioCommonsTimbre/TimbralModels_v0.2_Development/Reverb/stimuli/",
        "rough":"/nfs/guille/eecs_research/soundbendor/datasets/sounds_and_noise/AudioCommonsTimbre/TimbralModels_v0.2_Development/Roughness/Select Stimuli/Listening tests/Stimuli/",
        "sharp":"/nfs/guille/eecs_research/soundbendor/datasets/sounds_and_noise/AudioCommonsTimbre/TimbralModels_v0.2_Development/Sharpness/Select Stimuli/Listening tests/Stimuli/",
        "warm": "/nfs/guille/eecs_research/soundbendor/datasets/sounds_and_noise/AudioCommonsTimbre/TimbralModels_v0.2_Development/Warmth/Select Stimuli/Listening tests/Stimuli/"

                }
    OLD_FILE_PATHS = {
        "booming":"/nfs/guille/eecs_research/soundbendor/datasets/sounds_and_noise/AudioCommonsTimbre/TimbralModels_v0.2_Development/Boominess/Boominess ratings/Listening tests/Stimuli/",
        "bright": "/nfs/guille/eecs_research/soundbendor/datasets/sounds_and_noise/AudioCommonsTimbre/TimbralModels_v0.2_Development/Brightness/Brightness ratings/Listening tests/Stimuli/",
        "deep":"/nfs/guille/eecs_research/soundbendor/datasets/sounds_and_noise/AudioCommonsTimbre/TimbralModels_v0.2_Development/Depth/Depth ratings/Listening tests/Stimuli",
        "hard":"/nfs/guille/eecs_research/soundbendor/datasets/sounds_and_noise/AudioCommonsTimbre/TimbralModels_v0.2_Development/Hardness/Hardness ratings/Listening tests/LoudNormStimuli/",
        "reverb":"/nfs/guille/eecs_research/soundbendor/datasets/sounds_and_noise/AudioCommonsTimbre/TimbralModels_v0.2_Development/Reverb/stimuli/",
        "rough":"/nfs/guille/eecs_research/soundbendor/datasets/sounds_and_noise/AudioCommonsTimbre/TimbralModels_v0.2_Development/Roughness/Roughness ratings/Listening tests/Stimuli/",
        "sharp":"/nfs/guille/eecs_research/soundbendor/datasets/sounds_and_noise/AudioCommonsTimbre/TimbralModels_v0.2_Development/Sharpness/Sharpness ratings/Listening tests/Stimuli/",
        "warm": "/nfs/guille/eecs_research/soundbendor/datasets/sounds_and_noise/AudioCommonsTimbre/TimbralModels_v0.2_Development/Warmth/Warmth ratings/Listening tests/Stimuli/",

                }
    FILE_PATHS = {
        "booming":"/nfs/stak/users/mccabepe/hpc_share/audiocommons/TimbralModels_v0.2_Development/Boominess/Boominess ratings/Listening tests/Stimuli/",
        "bright": "/nfs/stak/users/mccabepe/hpc_share/audiocommons/TimbralModels_v0.2_Development/Brightness/Brightness ratings/Listening tests/Stimuli/",
        "deep":"/nfs/stak/users/mccabepe/hpc_share/audiocommons/TimbralModels_v0.2_Development/Depth/Depth ratings/Listening tests/Stimuli",
        "hard":"/nfs/stak/users/mccabepe/hpc_share/audiocommons/TimbralModels_v0.2_Development/Hardness/Hardness ratings/Listening tests/LoudNormStimuli/",
        "reverb":"/nfs/stak/users/mccabepe/hpc_share/audiocommons/TimbralModels_v0.2_Development/Reverb/stimuli/",
        "rough":"/nfs/stak/users/mccabepe/hpc_share/audiocommons/TimbralModels_v0.2_Development/Roughness/Roughness ratings/Listening tests/Stimuli/",
        "sharp":"/nfs/stak/users/mccabepe/hpc_share/audiocommons/TimbralModels_v0.2_Development/Sharpness/Sharpness ratings/Listening tests/Stimuli/",
        "warm": "/nfs/stak/users/mccabepe/hpc_share/audiocommons/TimbralModels_v0.2_Development/Warmth/Warmth ratings/Listening tests/Stimuli/",

                }


    
    
    if key == "reverb":
        df = pd.read_pickle(DF_PATHS["reverb"])
        pattern = re.compile(r'-(\d+)\.wav$')
        df['freesound_id'] = [pattern.search(f).group(1) if pattern.search(f) else None 
                                for f in df['sound_file']]
        df = _remove_overlaps(df)
        return df, key
    print(f"Retrieving audio data from: {DF_PATHS[key]}")
    df = pd.read_pickle(DF_PATHS[key])
    numeric = df.drop(["sound_file"],axis=1)

    mean_value = numeric.mean(axis=1)
    df["mean"] = mean_value
    df["label"] = [1 if mean_value[i] >= 50 else 0 for i in range(len(mean_value))]
    curr_file_path=FILE_PATHS[key]
    paths = [os.path.join(curr_file_path,file) for file in df.sound_file]
    df['path'] = paths

    bad_file_ids = []
    for i,file in enumerate(df.path):
        try:
            soundfile.read(file)
        except Exception as e:
            bad_file_ids.append(i)
            print(e)
    if len(bad_file_ids) > 0:
        df = df.drop(df.index[bad_file_ids],axis=0)


    #store freesound ids.
    pattern = re.compile(r'-(\d+)\.wav$')
    df['freesound_id'] = [pattern.search(f).group(1) if pattern.search(f) else None 
                             for f in df['sound_file']]
    df = _remove_overlaps(df)
    return df, key




if __name__ == "__main__":
   
    label_set = ["booming","bright","deep","hard","reverb","rough","sharp","warm"]
    dfs = [get_dataset_generic(l) for l in label_set]

    for df in dfs:
        print(df.label.value_counts())
        
    

    exit()
    #df = get_dataset_generic("hard")
    #t = pd.read_pickle("/nfs/guille/eecs_research/soundbendor/datasets/sounds_and_noise/AudioCommonsTimbre/TimbralModels_v0.2_Development/Brightness/Brightness ratings/Ancillary files/brightness_df_mean.pickle")
    #print(t.head())
    #print(df.head())
    #test = soundfile.read(df.path.iloc[0])
    #for file in df.path:
    #    try:
    #        t = soundfile.read(file)
    #    except Exception as e:
    #        print(e, file)

    

    # Loop over all unique pairs of DataFrames
    for i, j in combinations(range(len(dfs)), 2):
        df_i_files = set(dfs[i]["sound_file"])
        df_j_files = set(dfs[j]["sound_file"])
        intersection = df_i_files & df_j_files
        print(f"Overlap between df{i+1} and df{j+1}: {len(intersection)} common sound files")
