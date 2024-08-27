import pandas as pd
import re
import os
from functools import reduce

def merge_files_to_dataframe(directory):
    """Combines a AudioCommon Results directory into a pandas dataframe.
       The ratings of duplicate sound files in a listening session file (same reviewer) are averaged, and duplicates across listening sessions are averaged again.
        E.g.: SubXResults_p1.txt
            0 A.wav 8
            ...
            5 A.wav 7
            --->
            A.wav 7.5


            SubXResults_pt.txt
            0 A.wav 8

            --->
            concatenated dataframe:
            0 A.wav 7.75 (Real mean: 7.67)
    """
    data_dict = {}

    # Regular expression pattern to match files of the form "SubXResults_pY.txt"
    pattern = re.compile(r"Sub(\d{1,2})Results_p(\d).txt")

    # Iterate through all files in the specified directory
    for filename in os.listdir(directory):
        # Check if the file matches the pattern
        match = pattern.match(filename)
        if match:
            # Extract X and Y values from the filename
            x_value = match.group(1)
            y_value = match.group(2)

            # Construct the full path of the file
            file_path = os.path.join(directory, filename)

            # Read the file into a DataFrame
            df = pd.read_csv(file_path,delimiter=None,names=["page_number","sound_file",f"rating_{x_value}"],skipinitialspace=True)  # Adjust delimiter if necessary
            df = df.drop("page_number",axis=1)
            df['sound_file'] = df['sound_file'].str.strip().str.lower()
            df = df.groupby('sound_file', as_index=False).mean()


            # Append or create a new entry in the dictionary for this X value
            if x_value not in data_dict:
                data_dict[x_value] = df
            else:
                data_dict[x_value] = pd.concat([data_dict[x_value], df], ignore_index=True).groupby('sound_file', as_index=False).mean()
                print(data_dict[x_value].head())

    # Combine all DataFrames for each X value into a single DataFrame
    combined_df = None
    
    for key, df in data_dict.items():
        if combined_df is None:
            combined_df = df
        else:
            print(f"Merging DataFrame for X={key}, size before merge: {combined_df.shape}")
            combined_df = pd.merge(combined_df, df, on='sound_file', how='outer')
            print(f"Size after merge: {combined_df.shape}")
    

    return combined_df
       



#  a directory and its name
results_directories = ["/nfs/guille/eecs_research/soundbendor/datasets/sounds_and_noise/AudioCommonsTimbre/TimbralModels_v0.2_Development/Boominess/Boominess ratings/Listening tests/Results",
                       "/nfs/guille/eecs_research/soundbendor/datasets/sounds_and_noise/AudioCommonsTimbre/TimbralModels_v0.2_Development/Brightness/Brightness ratings/Listening tests/Results",
                       "/nfs/guille/eecs_research/soundbendor/datasets/sounds_and_noise/AudioCommonsTimbre/TimbralModels_v0.2_Development/Depth/Depth ratings/Listening tests/Results/",
                       "/nfs/guille/eecs_research/soundbendor/datasets/sounds_and_noise/AudioCommonsTimbre/TimbralModels_v0.2_Development/Hardness/Hardness ratings/Listening tests/Results",
                       "/nfs/guille/eecs_research/soundbendor/datasets/sounds_and_noise/AudioCommonsTimbre/TimbralModels_v0.2_Development/Roughness/Roughness ratings/Listening tests/Results",
                       "/nfs/guille/eecs_research/soundbendor/datasets/sounds_and_noise/AudioCommonsTimbre/TimbralModels_v0.2_Development/Sharpness/Sharpness ratings/Listening tests/Results",
                       "/nfs/guille/eecs_research/soundbendor/datasets/sounds_and_noise/AudioCommonsTimbre/TimbralModels_v0.2_Development/Warmth/Warmth ratings/Listening tests/Results",]

dir_labels = ["Boominess","Brightness","Depth","Hardness","Roughness","Sharpness","Warmth"]
#test_dirs = ["/nfs/guille/eecs_research/soundbendor/datasets/sounds_and_noise/AudioCommonsTimbre/TimbralModels_v0.2_Development/Boominess/Boominess ratings/Listening tests/Results",
#                      "/nfs/guille/eecs_research/soundbendor/datasets/sounds_and_noise/AudioCommonsTimbre/TimbralModels_v0.2_Development/Brightness/Brightness ratings/Listening tests/Results"]

#test_labels = ["Boom_test","Bright_test"]
DATA_PATH = "/nfs/guille/eecs_research/soundbendor/mccabepe/timbre_tags/data"
for name,dir in zip(dir_labels,results_directories):
    combined_df = merge_files_to_dataframe(dir)
    print(combined_df.head())
    write_file = f"ACTD_{name}.pickle"
    combined_df.to_pickle(os.path.join(DATA_PATH,write_file))
    

    