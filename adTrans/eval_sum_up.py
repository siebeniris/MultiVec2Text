import pandas as pd

import os


def eval_summary(directory):
    indices_dict = {
        "correction1": "1 Step", "correction20": "20 Steps",
        "correction50": "50 Steps", "correction50_sbeam4": "50 Steps + 4 sbeam",
        "correction50_sbeam8": "50 Steps + 8 sbeam",
        "correction100": "100 Steps", "correction100_sbeam4": "100 Steps + 4 sbeam",
        "correction100_sbeam8": "100 Steps + 8 sbeam"
    }

    subdirs = ["mtg_de", "mtg_es", "mtg_fr", "nq_en", "mtg_en", "nq"]

    bleus_gains = []
    bleu_preds = []
    bleu_translates = []

    tokenf1s_gains = []
    tokenf1s_preds = []
    tokenf1s_translates = []

    steps = []
    for dirname in os.listdir(directory):
        if dirname in subdirs:
            print(dirname)
            if dirname.startswith("correction"):
                indice = indices_dict.get(dirname, dirname)
                eval_file = os.path.join(directory, dirname, "eval_results.csv")
                df_eval = pd.read_csv(eval_file, index_col=0)
                bleu_pred = df_eval.at["bleu_score", "pred_eval"]
                bleu_translated = df_eval.at["bleu_score", "translation_eval"]
                bleu_percentages = round((bleu_translated - bleu_pred) / bleu_pred * 100, 2)

                bleu_preds.append(bleu_pred)
                bleu_translates.append(bleu_translated)

                tokenf1_pred = df_eval.at["token_set_f1", "pred_eval"]
                tokenf1_translated = df_eval.at["token_set_f1", "translation_eval"]
                tokenf1_percentages = round((tokenf1_translated - tokenf1_pred) / tokenf1_pred * 100, 2)

                tokenf1s_preds.append(tokenf1_pred)
                tokenf1s_translates.append(tokenf1_translated)

                bleus_gains.append(bleu_percentages)
                tokenf1s_gains.append(tokenf1_percentages)

                steps.append(indice)
            else:
                eval_file = os.path.join(directory, dirname, "eval_results.csv")
                df_eval = pd.read_csv(eval_file, index_col=0)
                bleu_pred = df_eval.at["bleu_score", "pred_eval"]
                bleu_translated = df_eval.at["bleu_score", "translation_eval"]
                bleu_percentages = round((bleu_translated - bleu_pred) / bleu_pred * 100, 2)

                bleu_preds.append(round(bleu_pred, 2))
                bleu_translates.append(round(bleu_translated, 2))

                tokenf1_pred = df_eval.at["token_set_f1", "pred_eval"]
                tokenf1_translated = df_eval.at["token_set_f1", "translation_eval"]
                tokenf1_percentages = round((tokenf1_translated - tokenf1_pred) / tokenf1_pred * 100, 2)

                tokenf1s_preds.append(round(tokenf1_pred, 2))
                tokenf1s_translates.append(round(tokenf1_translated, 2))

                bleus_gains.append(bleu_percentages)
                tokenf1s_gains.append(tokenf1_percentages)
                steps.append(dirname)

    df = pd.DataFrame.from_dict({"BLEU_pred": bleu_preds,
        "BLEU_translated": bleu_translates,
        "BLEU_gain": bleus_gains,
        "Tokenf1s_pred": tokenf1s_preds,
        "Tokenf1s_translated": tokenf1s_translates,
        "ToeknsF1_gain": tokenf1s_gains})
    df.index = steps

    print(df.sort_index())
    df.to_csv(os.path.join(directory, "eval_summary.csv"))


if __name__ == '__main__':
    import plac

    plac.call(eval_summary)
