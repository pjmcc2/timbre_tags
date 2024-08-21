import pandas as pd

import pandas as pd
import re
import os

def merge_files_to_dataframe(directory):
    # Dictionary to hold dataframes for each X value
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
            df = pd.read_csv(file_path, delimiter='\t',names=["page_number","sound_file",f"rating_{x_value}"])  # Adjust delimiter if necessary
            df = df.drop("page_number",axis=1).sort_values("sound_file").set_index("sound_file")

            
            # Append or create a new entry in the dictionary for this X value
            if x_value not in data_dict:
                data_dict[x_value] = df
            else:
                data_dict[x_value] = pd.concat([data_dict[x_value], df], ignore_index=True)
    
    # Combine all DataFrames for each X value into a single DataFrame
    combined_df = pd.concat(data_dict.values(),axis=1,sort=False).reset_index()
    combined_df.rename(columns = {'index':'sound_file'})
    
    return combined_df


def aggregate_data(df,kind='mean'):
    kind_vals = ['vote','mean','greater_median',None]
    if kind not in kind_vals:
        raise ValueError(f"{kind} not in {kind_vals}")

    if kind =='mean':
        return round(df.mean(axis=1)/100)
    else:
        encoded = mean_encode(df)
        if kind == 'vote':
            return [round(sum(row)/len(row)) for _,row in encoded.iterrows()] # TODO Fix
        


def mean_encode(df):
    means = df.mean(axis=0)
    vals = {}
    for i,tup in enumerate(df.items()):
        name,series = tup
        vals[name] = series > means[i]
    return pd.DataFrame(vals)


#  a directory and its name
results_directories = ["/nfs/guille/eecs_research/soundbendor/datasets/sounds_and_noise/AudioCommonsTimbre/TimbralModels_v0.2_Development/Boominess/Boominess ratings/Listening tests/Results",
                       "/nfs/guille/eecs_research/soundbendor/datasets/sounds_and_noise/AudioCommonsTimbre/TimbralModels_v0.2_Development/Brightness/Brightness ratings/Listening tests/Results",
                       "/nfs/guille/eecs_research/soundbendor/datasets/sounds_and_noise/AudioCommonsTimbre/TimbralModels_v0.2_Development/Depth/Depth ratings/Listening tests/Results/",
                       "/nfs/guille/eecs_research/soundbendor/datasets/sounds_and_noise/AudioCommonsTimbre/TimbralModels_v0.2_Development/Hardness/Hardness ratings/Listening tests/Results",
                       "/nfs/guille/eecs_research/soundbendor/datasets/sounds_and_noise/AudioCommonsTimbre/TimbralModels_v0.2_Development/Reverb/Reverb ratings/Listening tests/Results",
                       "/nfs/guille/eecs_research/soundbendor/datasets/sounds_and_noise/AudioCommonsTimbre/TimbralModels_v0.2_Development/Sharpness/Sharpness ratings/Listening tests/Results",
                       "/nfs/guille/eecs_research/soundbendor/datasets/sounds_and_noise/AudioCommonsTimbre/TimbralModels_v0.2_Development/Warmth/Warmth ratings/Listening tests/Results",]

dir_labels = ["Boominess","Brightness","Depth","Hardness","Reverb","Roughness","Sharpness","Warmth"]
# for each results directory, combine parts into one file, then average across columns
mean_dict = {} 
for name,dir in zip(dir_labels,results_directories):
    combined_df = merge_files_to_dataframe(dir)
    #calculate median of columns
    #determine aggregation strategies: OR, mean, Vote, keep all, etc.
    

    