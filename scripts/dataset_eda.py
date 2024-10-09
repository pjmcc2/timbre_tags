import numpy as np
import pandas as pd
import os
import json
import nltk
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

#THINGS I WANT:
## number of uses of timbre tags
## relative usage compared with all words
## relative usage compared with ajectives/
## proportion of adjectives to all words.
## comparisons between audio and music datasets.

def contains_two_words(s):
    words = s.split()
    return len(words) == 2

import nltk
from collections import Counter
from nltk.util import ngrams


def count_bigrams(bigram_list, corpus):
    
    words = nltk.word_tokenize(corpus.lower()) 
    
    
    bigrams_in_sentence = list(ngrams(words, 2))
    
    
    bigram_count = Counter(bigrams_in_sentence)
    
    # Convert the list of bigrams to tuples (for comparison)
    bigram_list_tuples = [tuple(bigram.split()) for bigram in bigram_list]
    
    # Count only the specific bigrams from the provided list
    specific_bigram_count = {bigram: bigram_count[bigram] for bigram in bigram_list_tuples}
    
    return specific_bigram_count

def count_parts_of_speech(corpus):
    
    words = nltk.word_tokenize(corpus)
    
   
    pos_tags = nltk.tag.pos_tag_sents(words)
    
    
    pos_count = Counter(tag for word, tag in pos_tags)
    
    return pos_count , len(words)


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


def count_adjectives(sentence):
    # Tokenize the sentence into words
    words = nltk.word_tokenize(sentence)
    # Perform part-of-speech tagging
    tagged = nltk.tag.pos_tag(words)
    # Count adjectives (tagged as 'JJ' for adjectives)
    adjective_count = sum(1 for word, tag in tagged if tag.startswith('JJ'))
    return adjective_count


def get_adjective_counts_from_text(text):
    # Tokenize the text into sentences
    sentences = nltk.sent_tokenize(text)
    # Create a dictionary to store adjectives and their counts
    adjective_counts = defaultdict(int)
    
    # Process each sentence
    for sentence in sentences:
        # Tokenize the sentence into words
        words = nltk.word_tokenize(sentence)
        # Perform part-of-speech tagging
        tagged = nltk.tag.pos_tag(words)
        
        # Count adjectives (tagged as 'JJ' for adjectives)
        for word, tag in tagged:
            if tag.startswith('JJ'):  # Check if the word is an adjective
                adjective_counts[word.lower()] += 1  # Convert word to lowercase and update count
    
    return dict(adjective_counts)


def count_sentences_with_words(sentences, word_set):
    # Convert word_set to lowercase for case-insensitive matching
    word_set = {word.lower() for word in word_set}
    
    # Initialize counter for sentences that contain at least one word from the set
    count = 0
    
    # Iterate through each sentence
    for sentence in sentences:
        # Tokenize the sentence into words
        words = nltk.word_tokenize(sentence)
        # Convert words to lowercase for case-insensitive matching
        words_lower = {word.lower() for word in words}
        
        # Check if any word in the sentence is in the word_set
        if word_set & words_lower:  # Using set intersection to check for common words
            count += 1
    
    return count


import nltk
from nltk.util import bigrams

def count_sentences_with_bigrams(sentences, bigram_set):
    # Ensure bigram_set consists of tuples of words, not characters
    bigram_set = {tuple(bigram.lower().split()) for bigram in bigram_set}
    
    # Initialize counter for sentences that contain at least one bigram from the set
    count = 0
    
    # Iterate through each sentence
    for sentence in sentences:
        # Tokenize the sentence into words
        words = nltk.word_tokenize(sentence)
        # Convert words to lowercase for case-insensitive matching
        words_lower = [word.lower() for word in words]
        
        # Generate bigrams from the tokenized words
        sentence_bigrams = set(bigrams(words_lower))
        
        # Check if any bigram in the sentence is in the bigram_set
        if bigram_set & sentence_bigrams:  # Using set intersection to check for common bigrams
            count += 1
            print(sentence)
    
    return count

if __name__ == "__main__":
    
    LABEL_PATH =  "/nfs/guille/eecs_research/soundbendor/mccabepe/timbre_tags/data/tags/carron_lavengood.csv"
    data = pd.read_csv(LABEL_PATH)
    
    WAVCAPS_PATH = "/nfs/guille/eecs_research/soundbendor/datasets/sounds_and_noise/wavcaps/json_files/"
    wv = combine_json_into_dataframe(WAVCAPS_PATH)

    my_bigrams = data["Merged"].dropna()[data["Merged"].dropna().apply(contains_two_words)]
    #unigrams = data["Merged"].dropna()[~data["Merged"].dropna().apply(contains_two_words)]
    #uni = count_tags(unigrams,"".join(wv.caption.tolist())) # does not account for two-word phrases: slow attack, fast attack
    #bi = count_bigrams(bigrams,"".join(wv.caption.tolist()))
    #print(uni, bi)
    #total_tag_count = sum(uni.values())
    #print(total_tag_count)
    #pos_count, total = count_parts_of_speech("".join(wv.caption.tolist()))
    #print(pos_count, total)
    #wv["adj_count"] = wv["caption"].apply(count_adjectives)
    #print(wv.adj_count.describe())
    #print(wv.adj_count.value_counts())
    #print(sorted([len(c) for c in wv.caption.tolist()],reverse=True)[:15])
    #adj_counts = pd.Series(get_adjective_counts_from_text("".join(wv.caption.tolist())))
    #pd.to_pickle(adj_counts, "/nfs/guille/eecs_research/soundbendor/mccabepe/timbre_tags/data/tags/wv_adj_counts.pickle")
    #print(count_sentences_with_words(wv.caption,unigrams))
    
    # print( {tuple(map(str.lower, bigram)) for bigram in my_bigrams})
    print(count_sentences_with_bigrams(wv.caption,my_bigrams))
