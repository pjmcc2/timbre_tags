from scripts.dataset_eda import count_bigrams
from scripts.dataset_eda import count_tags
from scripts.dataset_eda import contains_two_words
from datasets import load_dataset
from datasets import load_dataset_builder


if __name__ == "__main__":
    ds_names = ["seungheondoh/LP-MusicCaps-MSD","seungheondoh/LP-MusicCaps-MTT","seungheondoh/LP-MusicCaps-MC"]
    ds_name = ""
    ds_builder = load_dataset_builder(ds_name)

    print(f"{ds_name} description: {ds_builder.info.description}")

    print(f"{ds_name} features: {ds_builder.info.features}")