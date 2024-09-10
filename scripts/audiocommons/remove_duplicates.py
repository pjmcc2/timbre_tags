import pandas as pd
import json
import os
from tqdm import tqdm
import chardet 
  
def detect_encoding(file_path): 
    with open(file_path, 'rb') as file: 
        detector = chardet.universaldetector.UniversalDetector() 
        for line in file: 
            detector.feed(line) 
            if detector.done: 
                break
        detector.close() 
    return detector.result['encoding'] 

if __name__ == "__main__":

    wav_fs_path = "/nfs/guille/eecs_research/soundbendor/datasets/sounds_and_noise/wavcaps/json_files/fsd_final.json"
    ls = [
        "/nfs/guille/eecs_research/soundbendor/datasets/sounds_and_noise/AudioCommonsTimbre/TimbralModels_v0.2_Development/Boominess/Select Stimuli/Licences/BoominessStimuliSelectionTestLicences.csv",
        "/nfs/guille/eecs_research/soundbendor/datasets/sounds_and_noise/AudioCommonsTimbre/TimbralModels_v0.2_Development/Brightness/Select Stimuli/Licences/BrightnessStimuliSelectionTestLicences.csv",
        "/nfs/guille/eecs_research/soundbendor/datasets/sounds_and_noise/AudioCommonsTimbre/TimbralModels_v0.2_Development/Depth/Select Stimuli/Licences/DepthStimuliSelectionTestLicences.csv",
        "/nfs/guille/eecs_research/soundbendor/datasets/sounds_and_noise/AudioCommonsTimbre/TimbralModels_v0.2_Development/Hardness/Select Stimuli/Licences/HardnessStimuliSelectionTestLicences.csv",
        "/nfs/guille/eecs_research/soundbendor/datasets/sounds_and_noise/AudioCommonsTimbre/TimbralModels_v0.2_Development/Reverb/Licence information.csv",
        "/nfs/guille/eecs_research/soundbendor/datasets/sounds_and_noise/AudioCommonsTimbre/TimbralModels_v0.2_Development/Roughness/Select Stimuli/Licences/RoughnessStimuliSelectionTestLicences.csv",
        "/nfs/guille/eecs_research/soundbendor/datasets/sounds_and_noise/AudioCommonsTimbre/TimbralModels_v0.2_Development/Sharpness/Select Stimuli/Licences/SharpnessStimuliSelectionTestLicences.csv",
        "/nfs/guille/eecs_research/soundbendor/datasets/sounds_and_noise/AudioCommonsTimbre/TimbralModels_v0.2_Development/Warmth/Select Stimuli/Licences/WarmthStimuliSelectionTestLicences.csv"
    ]

    encodings = [detect_encoding(path) for path in ls]
    
    fs_ids = []
    for i,f in enumerate(ls):
        print(f"reading: {f}")
        l_df = pd.read_csv(f,header=0,encoding=encodings[i])
        fs_ids.extend(l_df["Freesound_ID"].to_list())

    with open(wav_fs_path) as f:
        dict_v = json.load(f)
    wav_fs = pd.DataFrame(dict_v["data"])

    dup_ids = []
    for i in tqdm(range(len(fs_ids))):
        if fs_ids[i] in wav_fs['id']:
            dup_ids.append(i)

    wav_fs = wav_fs.drop(dup_ids,axis=0)
    pd.to_pickle(wav_fs[["id","description"]],"/nfs/guille/eecs_research/soundbendor/datasets/sounds_and_noise/wavcaps/misc/audiocommons_duplicates_dropped_descriptions.pickle")