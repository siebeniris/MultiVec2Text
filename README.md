# Text Embedding Inversion Attacks on Multilingual Language Models

* Schematic Overview of a Text Embedding Inversion Attack.
<img src="images/attack.png" alt="attack" width="400">

Multilingual Vec2Text supports research in Text Embedding Inversion Security in Language Models, 
extending Jack Morris' [Vec2Text](https://github.com/jxmorris12/vec2text) with __Ad-hoc Translation__ and __Masking Defense Mechanism__. 
We investigate thoroughly multilingual and cross-lingual text inversion attacks, 
and defense mechanisms. This repository contains code for the __ACL 2024__ long paper [Text Embedding Inversion Attacks on Multilingual Language Models
](https://arxiv.org/abs/2401.12192).
The poster is [online](multilingual_text2vec_poster.pdf).


All the trained inversion models are on [Huggingface](https://huggingface.co/yiyic).
All the models are trained with [T5-base](https://huggingface.co/google-t5/t5-base) as the external encoder-decoder.


Black-box Encoder | Training Data        | Base Model | Corrector Model
--- |----------------------| --- | ---
[GTR-base](https://huggingface.co/sentence-transformers/gtr-t5-base) | 5M Natural Questions | [yiyic/t5_gtr_base_nq_32_inverter](https://huggingface.co/yiyic/t5_gtr_base_nq_32_inverter) | [yiyic/t5_gtr_base_nq_32_corrector](https://huggingface.co/yiyic/t5_gtr_base_nq_32_corrector) 
[ME5-base](https://huggingface.co/intfloat/multilingual-e5-base) | 5M Natural Questions | [yiyic/t5_me5_base_nq_32_inverter](https://huggingface.co/yiyic/t5_me5_base_nq_32_inverter) | [yiyic/t5_me5_base_nq_32_corrector](https://huggingface.co/yiyic/t5_me5_base_nq_32_corrector)
[ME5-base](https://huggingface.co/intfloat/multilingual-e5-base) | 5M MTG Spanish       | [yiyic/t5_me5_base_mtg_es_5m_32_inverter](https://huggingface.co/yiyic/t5_me5_base_mtg_es_5m_32_inverter) | [yiyic/t5_me5_base_mtg_es_5m_32_corrector](https://huggingface.co/yiyic/t5_me5_base_mtg_es_5m_32_corrector)
[ME5-base](https://huggingface.co/intfloat/multilingual-e5-base) | 5M MTG French        | [yiyic/t5_me5_base_mtg_fr_5m_32_inverter](https://huggingface.co/yiyic/t5_me5_base_mtg_fr_5m_32_inverter) | [yiyic/t5_me5_base_mtg_fr_5m_32_corrector](https://huggingface.co/yiyic/t5_me5_base_mtg_fr_5m_32_corrector)
[ME5-base](https://huggingface.co/intfloat/multilingual-e5-base) | 5M MTG German        | [yiyic/t5_me5_base_mtg_de_5m_32_inverter](https://huggingface.co/yiyic/t5_me5_base_mtg_de_5m_32_inverter) | [yiyic/t5_me5_base_mtg_de_5m_32_corrector](https://huggingface.co/yiyic/t5_me5_base_mtg_de_5m_32_corrector)
[ME5-base](https://huggingface.co/intfloat/multilingual-e5-base) | 5M MTG English       | [yiyic/t5_me5_base_mtg_en_5m_32_inverter](https://huggingface.co/yiyic/t5_me5_base_mtg_en_5m_32_inverter) | [yiyic/t5_me5_base_mtg_en_5m_32_corrector](https://huggingface.co/yiyic/t5_me5_base_mtg_en_5m_32_corrector)
[ME5-base](https://huggingface.co/intfloat/multilingual-e5-base) | 5M MTG Multilingual  | [yiyic/t5_me5_base_mtg_en_fr_de_es_5m_32_inverter](https://huggingface.co/yiyic/t5_me5_base_mtg_en_fr_de_es_5m_32_inverter) | [yiyic/t5_me5_base_mtg_en_fr_de_es_5m_32_corrector](https://huggingface.co/yiyic/t5_me5_base_mtg_en_fr_de_es_5m_32_corrector)









[//]: # (There is documentation of the extended functions and features of this repository compared to `Vec2Text`,)

[//]: # (see [documentation]&#40;https://github.com/siebeniris/MultiVec2Text/wiki/New-Functions-and-Features,-extended-upon-Vec2Text&#41;,)

[//]: # (which is under continuous maintenace.)


* Overview of Multilingual Vec2Text.
<img src="images/overview.png" width="1000"/>


The tutorials for setting up experiments on supercomputer nodes such as __[LUMI](https://docs.lumi-supercomputer.eu/)__ will be in Wiki pages. 
All the scripts for running experiments _will be_ provided in the GitHub repository. 
GitHub is still under construction.

## Experiments (Inversion attack simulations)
<img src="images/simulation.png" width="1000"/>


## Setup
1. download the release from [releases](https://github.com/siebeniris/MultiVec2Text/tags) and unzip.
2. `pip install -r requirements.txt`
3. donwload `punkt` package from `nltk`
```
import nltk
nltk.download("punkt")
```



## Text Embedding Examples
Usage in [interactive environment in a server](https://github.com/siebeniris/MultiVec2Text/wiki/Sample-Evaluation-in-an-Interactive-Environment)



```

from eval_samples import * 

model_path="yiyic/t5_me5_base_mtg_en_fr_de_es_5m_32_corrector"

samples = ["jack morris is a phd student at cornell tech in new york city",
"it was the best of times, it was the worst of times, it was the age of wisdom",
"in einer stunde erreichen wir kopenhagen."., 
"comment puis-je vous aider?"
]

experiment, trainer = analyze_utils.load_experiment_and_trainer_from_pretrained(
        model_path, use_less_data=3000)

trainer, device = trainer_attributes(trainer, experiment)
trainer.num_gen_recursive_steps = 10
# set sbeam
# trainer.sequence_beam_width = xx

evaluate_samples(trainer, device, samples)

```

output:

```

[pred] jack morris is a phd student at cornell tech in new york city
[true] jack morris is a phd student at cornell tech in new york city



[pred] it was the best of times, it was the worst of times, it was the age of wisdom
[true] it was the best of times, it was the worst of times, it was the age of wisdom



[pred] in einer stunde erreichen wir kopenhagen.
[true] in einer stunde erreichen wir kopenhagen.



[pred] comment puis-je vous aider?
[true] comment puis-je vous aider?

```



## Ad-Hoc Translation (AdTrans)
The codes for AdTrans evaluation is in `adTrans`.

* translate the $\hat{x}$ from training language to target language and evaluate.

```
python adTrans/translate_test_results.py $results_output_directory$
python adTrans/eval.py $results_output_directory$ 
python adTrans/eval_sum_up.py $results_output_directory$
```


__Inversion Models Limitations__
- To analyze the impact of multilingual parallel data training, 
we used [MTG](https://aclanthology.org/2022.findings-naacl.192/) benchmark in English, French, German, and Spanish, 
the texts in the datasets were given as lower-cased, so our trained inversion `ME5-based` models
work mostly for lower-cased texts as well. 
Along with dataset limitation, the best performing models invert sentences within the length of 32 tokens.
We will address these limitation for future work.


## Cite our Paper 

```
@inproceedings{chen-etal-2024-text,
    title = "Text Embedding Inversion Security for Multilingual Language Models",
    author = "Chen, Yiyi  and
      Lent, Heather  and
      Bjerva, Johannes",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-long.422",
    pages = "7808--7827",
    abstract = "Textual data is often represented as real-numbered embeddings in NLP, particularly with the popularity of large language models (LLMs) and Embeddings as a Service (EaaS). However, storing sensitive information as embeddings can be susceptible to security breaches, as research shows that text can be reconstructed from embeddings, even without knowledge of the underlying model. While defence mechanisms have been explored, these are exclusively focused on English, leaving other languages potentially exposed to attacks. This work explores LLM security through multilingual embedding inversion. We define the problem of black-box multilingual and crosslingual inversion attacks, and explore their potential implications. Our findings suggest that multilingual LLMs may be more vulnerable to inversion attacks, in part because English-based defences may be ineffective. To alleviate this, we propose a simple masking defense effective for both monolingual and multilingual models. This study is the first to investigate multilingual inversion attacks, shedding light on the differences in attacks and defenses across monolingual and multilingual settings.",
}
```


