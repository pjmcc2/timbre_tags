import numpy as np
from sklearn.preprocessing import normalize
from data.utils.embed_wavcaps import encode_data
import argparse
import os
import pickle


def gen_label_prompts(class_types,model_name):
    AC_TARGETS = ["booming","bright","deep","hard","reverb","rough","sharp","warm"]
    ESC50_TARGETS = None # TODO Implement

    timbre_prompts = {l:[f"A {l} sound.",f"A sound that could be described as: {l}",f"An audio clip with a {l} quality.", \
                    f"An audio clip that sounds {l}", f"A sound/audio clip that has or contains {l}."]
            for l in AC_TARGETS}
    esc50_prompts = None # TODO IMplement

    if class_types == "ac":
        prompt_embeddings = {l : encode_data(timbre_prompts[l],model_name=model_name) for l in AC_TARGETS}
        for l in AC_TARGETS:
            raw_mean_embedding = np.mean(prompt_embeddings[l],axis=0,keepdims=True)
            mean_embedding = normalize(raw_mean_embedding, norm='l2',axis=1,return_norm=False)
            prompt_embeddings[l] = np.vstack((prompt_embeddings[l],mean_embedding))

    elif class_types == "esc50":
        pass # TODO Implment
    else:
        raise ValueError(f"Wrong class names: {class_types}")


    return prompt_embeddings



def gen_dataset(dataset_name, n_samples, seed=456):
    EMPIRICAL_SIGMA = 0.023023764
    
    data = []
    labels = []
    rng = np.random.default_rng(seed)
    
    prompts_embeddings_dict = gen_label_prompts(dataset_name, "clap")
    num_classes = len(prompts_embeddings_dict.keys())
    num_prompts = 6
    per_label_per_prompt = (n_samples // (num_classes * num_prompts)) + 1

    for i, prompt_embs in enumerate(prompts_embeddings_dict.values()):
        # Generate samples around each class embedding
        for j in range(prompt_embs.shape[0]):
            samples = rng.multivariate_normal(
                mean=prompt_embs[j].squeeze(), 
                cov=EMPIRICAL_SIGMA * np.eye(512), 
                size=per_label_per_prompt
            )
            data.append(samples)

            # One-hot labels
            curr_labels = np.zeros((per_label_per_prompt, num_classes))
            curr_labels[:, i] = 1
            labels.append(curr_labels)

    return np.vstack(data), np.vstack(labels)


def save_dataset(data,dataset_name):
    output_dir = "data/processed/noise"
    output_file = f"noise_{dataset_name}.pickle"
    with open(os.path.join(output_dir,output_file),"wb") as f:
        pickle.dump(data,f)



if __name__ == "__main__":
     
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument("--n_samples",type=int, required=True)
    parser.add_argument("--seed",type=int)
    args = parser.parse_args()
    X,y = gen_dataset(args.dataset_name,args.n_samples,seed=args.seed)
    save_dataset((X,y),args.dataset_name)