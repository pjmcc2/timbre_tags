


def embed_clotho_captions(dir_path,out_path="clap_embedding_no_dup.pickle",drop_dupes=True):
    if os.path.exists(os.path.join(dir_path,out_path)):
        return pd.read_csv(os.path.join(dir_path,out_path))
    
    caption_df = combine_caption_csvs(dir_path)
    caption_df = pd.unmel
    captions = caption_df.
    if drop_dupes:
        meta_data = combine_id_lists(dir_path)
        # TODO

    else:
        # TODO
