import os.path

import pandas as pd
import plac
import scipy.stats
import tqdm
import collections
import evaluate
import nltk
import numpy as np

from typing import Callable, Dict, List, Tuple, Union


def sem(L: List[float]) -> float:
    result = scipy.stats.sem(np.array(L))
    if isinstance(result, np.ndarray):
        return result.mean().item()
    return result


def mean(L: Union[List[int], List[float]]) -> float:
    return sum(L) / len(L)


metric_bleu = evaluate.load("sacrebleu")
metric_rouge = evaluate.load("rouge")
metric_accuracy = evaluate.load("accuracy")


def count_overlapping_ngrams(s1: str, s2: str, n: int) -> int:
    ngrams_1 = nltk.ngrams(s1, n)
    ngrams_2 = nltk.ngrams(s2, n)
    ngram_counts_1 = collections.Counter(ngrams_1)
    ngram_counts_2 = collections.Counter(ngrams_2)
    total = 0
    for ngram, count in ngram_counts_1.items():
        total += min(count, ngram_counts_2[ngram])
    return total


def text_comparison_metics(predictions, references):
    num_preds = len(predictions)

    precision_sum = 0.0
    recall_sum = 0.0
    num_overlapping_words = []
    num_overlapping_bigrams = []
    num_overlapping_trigrams = []
    num_true_words = []
    num_pred_words = []
    f1s = []
    for i in range(num_preds):
        true_words = nltk.tokenize.word_tokenize(references[i])
        pred_words = nltk.tokenize.word_tokenize(predictions[i])
        num_true_words.append(len(true_words))
        num_pred_words.append(len(pred_words))

        true_words_set = set(true_words)
        pred_words_set = set(pred_words)
        TP = len(true_words_set & pred_words_set)
        FP = len(true_words_set) - len(true_words_set & pred_words_set)
        FN = len(pred_words_set) - len(true_words_set & pred_words_set)
        precision = (TP) / (TP + FP + 1e-20)
        recall = (TP) / (TP + FN + 1e-20)

        try:
            f1 = (2 * precision * recall) / (precision + recall + 1e-20)
        except ZeroDivisionError:
            f1 = 0.0
        f1s.append(f1)

        precision_sum += precision
        recall_sum += recall

        ############################################################
        num_overlapping_words.append(
            count_overlapping_ngrams(true_words, pred_words, 1)
        )
        num_overlapping_bigrams.append(
            count_overlapping_ngrams(true_words, pred_words, 2)
        )
        num_overlapping_trigrams.append(
            count_overlapping_ngrams(true_words, pred_words, 3)
        )

    set_token_metrics = {
        "token_set_precision": (precision_sum / num_preds),
        "token_set_recall": (recall_sum / num_preds),
        "token_set_f1": mean(f1s),
        "token_set_f1_sem": sem(f1s),
        "n_ngrams_match_1": mean(num_overlapping_words),
        "n_ngrams_match_2": mean(num_overlapping_bigrams),
        "n_ngrams_match_3": mean(num_overlapping_trigrams),
        "num_true_words": mean(num_true_words),
        "num_pred_words": mean(num_pred_words),
    }

    bleu_results = np.array(
        [
            metric_bleu.compute(predictions=[p], references=[r])["score"]
            for p, r in zip(predictions, references)
        ]
    )

    rouge_result = metric_rouge.compute(
        predictions=predictions, references=references
    )
    bleu_results_list = bleu_results.tolist()

    bleu_results = (bleu_results.tolist())

    exact_matches = np.array(predictions) == np.array(references)
    gen_metrics = {
        "bleu_score": np.mean(bleu_results),
        "bleu_score_sem": sem(bleu_results),
        "rouge_score": rouge_result[
            "rouge1"
        ],  # ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
        # "bert_score": statistics.fmean(bertscore_result["f1"]),
        "exact_match": mean(exact_matches),
        "exact_match_sem": sem(exact_matches),
    }

    all_metrics = {**set_token_metrics, **gen_metrics}
    # return bleu_results_list, exact_matches, f1s
    return all_metrics


def eval_metrics(filepath):
    df = pd.read_csv(filepath)
    print(df.head())

    filedir = os.path.dirname(filepath)

    preds = df["pred"].str.lower().tolist()
    translations = df["pred_translated"].str.lower().tolist()
    references = df["labels"].str.lower().tolist()

    metrics_preds = text_comparison_metics(predictions=preds, references=references)


    print("metrics preds/ reference")
    print(metrics_preds)
    metrics = text_comparison_metics(predictions=translations, references=references)
    print("metrics preds translations / references")
    print(metrics)

    df_eval = pd.DataFrame.from_dict({"pred_eval": metrics_preds,
        "translation_eval": metrics})
    df_eval.to_csv(os.path.join(filedir, "eval_results.csv"))


def generating_for_sequences(filepath):
    df = pd.read_csv(filepath)
    print(df.head())

    filedir = os.path.dirname(filepath)

    preds = df["pred"].tolist()
    translations = df["pred_translated"].str.lower().tolist()
    references = df["labels"].tolist()

    preds_bleu_results, preds_exact_matches, preds_f1s = text_comparison_metics(predictions=preds, references=references)
    df["pred_bleu"] = preds_bleu_results
    df["pred_exact_match"] = preds_exact_matches
    df["pred_tokens_f1"] = preds_f1s

    trans_bleu_results, trans_exact_matches, trans_f1s = text_comparison_metics(predictions=translations,
        references=references)
    df["trans_bleu"] = trans_bleu_results
    df["trans_exact_match"] = trans_exact_matches
    df["trans_tokens_f1"] = trans_f1s


    df.to_csv(os.path.join(filedir, "eval_sequences.csv"))

# def processing_one_model(model_dir):
#     subdirs = ["mtg_de", "mtg_es", "mtg_fr", "nq_en", "mtg_en"]
#     # subdirs= ["nq"]
#     for subdir in os.listdir(model_dir):
#
#         if subdir in subdirs:
#             dirpath = os.path.join(model_dir, subdir)
#             print(f"evaluating model {dirpath}")
#             outfile = os.path.join(dirpath, "eval_results.csv")
#             # outfile = os.path.join(dirpath, "eval_sequences.csv")
#
#             print(f"outfile {outfile}")
#             #if not os.path.exists(outfile):
#
#             filepath = os.path.join(dirpath, "decoded_sequences_translated.csv")
#             print("evaluating....")
#             # generating_for_sequences(filepath)
#             eval_metrics(filepath)
#             #else:
#                 # print(f"file {outfile} already exists")


if __name__ == '__main__':
    plac.call(processing_one_model)
