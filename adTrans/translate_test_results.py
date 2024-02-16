import os

import pandas as pd
import plac
from easynmt import EasyNMT as nmt
from tqdm import tqdm


def processing_one_file(filepath, source_lang, target_lang):
    translator = nmt("opus-mt")
    save_dir = os.path.dirname(filepath)
    print(save_dir)
    df = pd.read_csv(filepath)
    pred_translated = []
    for sentence in tqdm(df["pred"].tolist()):
        pred_translated_sentence = translator.translate(sentence, source_lang=source_lang, target_lang=target_lang)
        # this is necessary. lower-case.
        pred_translated.append(pred_translated_sentence.lower())

    df["pred_translated"] = pred_translated

    save_filepath = os.path.join(save_dir, "decoded_sequences_translated.csv")
    df.to_csv(save_filepath, index=False)


def processing_one_model(model_dir):
    subdirs = ["mtg_de", "mtg_es", "mtg_fr", "nq_en", "mtg_en"]
    # subdirs =["nq"]
    for subdir in os.listdir(model_dir):
        target_lang = subdir.split("_")[-1]
        # target_lang = "en"
        if "mtg" in model_dir:
            source_lang = model_dir.replace("saves/yiyic__t5_me5_base_mtg_", "").split("_")[0]
        if "nq" in model_dir:
            source_lang = "en"

        print(f"processing {subdir}, source lang : {source_lang}, target_lang: {target_lang}")
        if subdir in subdirs:
            dirpath = os.path.join(model_dir, subdir)
            outfile = os.path.join(dirpath, "decoded_sequences_translated.csv")
            print(f"outfile {outfile}")
            if not os.path.exists(outfile):

                filepath = os.path.join(dirpath, "decoded_sequences.csv")
                print("processing....")
                processing_one_file(filepath, source_lang, target_lang)
            else:
                print(f"file {outfile} already exists")


if __name__ == '__main__':
    plac.call(processing_one_model)
