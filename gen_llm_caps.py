#from openai import OpenAI


#client=OpenAI(
#    organization="org-GFUWg03CGq42TdkUhHG1FSpR",
#)

import os
import json
import numpy as np


def sample_tags(tags,probs):
    idx = np.random.choice([i for i in range(1,len(probs))],size=(1,len(probs)),p=probs,replace=False)
    return [tags[i] for i in idx]


def construct_prompt(tags,nouns):
    adj = []
    n = []
    for t in tags:
        if t in nouns:
            n.append(t)
        else:
            adj.append(t)
    
    if len(adj) > 1:
        s = ", ".join(adj[:-1])
        s = s + f", and {adj[-1]}"
    elif len(adj) == 1:
        s = f"{adj[0]}"
    else:
        s=""
    
    if len(n) > 1:
        noun_chunk = ", ".join(n[:-1])
        noun_chunk = noun_chunk + f", and {n[-1]}"
    elif len(n) == 1:
        noun_chunk = f"{n[0]}"
    else:
        noun_chunk = ""
    
    s_part = f"might sound {s}" if len(s) > 0 else ""
    n_part = f"has a(n) {noun_chunk}"
    prompt = f"Briefly describe in one sentence a scene that might sound {s}, and has a(n) {noun_chunk}"
    
        



# tags: full set of AC U CHIT U Carron U lavengood?
TAG_SET = 

p= 1/3
probs = [p**i for i in range(len(TAG_SET))]
nums = [i+1 for i in range(len(TAG_SET))]

