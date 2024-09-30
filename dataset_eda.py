import numpy as np
import pandas as pd
import os
import json
import nltk
from collections import Counter

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')

#THINGS I WANT:
## number of uses of timbre tags
## relative usage compared with all words
## relative usage compared with ajectives/
## proportion of adjectives to all words.
## comparisons between audio and music datasets.




def count_parts_of_speech(corpus):
    
    words = nltk.word_tokenize(corpus)
    
   
    pos_tags = nltk.pos_tag(words)
    
    
    pos_count = Counter(tag for word, tag in pos_tags)
    
    return pos_count


def count_tags(word_list,sentence):

    words = nltk.word_tokenize(sentence.lower()) 
    
    word_count = Counter(words)
    
    specific_word_count = {word: word_count[word] for word in word_list}
    
    return specific_word_count

def combine_json_into_dataframe(dir):
    li = []
    for filename in os.listdir(dir):
        if filename.endswith(".json"):
            filepath = os.path.join(dir, filename)
            with open(filepath) as f:
                df = pd.DataFrame(json.load(f)['data'])
                li.append(df)

    return pd.concat(li,ignore_index=True)

if __name__ == "__main__":
    
    LABEL_PATH =  "/nfs/guille/eecs_research/soundbendor/mccabepe/timbre_tags/data/tags/carron_lavengood.csv"
    data = pd.read_csv(LABEL_PATH)
    
    WAVCAPS_PATH = "/nfs/guille/eecs_research/soundbendor/datasets/sounds_and_noise/wavcaps/json_files/"
    wv = combine_json_into_dataframe(WAVCAPS_PATH)

    
    cres = count_tags(data["Carron"].dropna(),"".join(wv.caption.tolist())) # does not account for two-word phrases: slow attack, fast attack
    print(cres)
