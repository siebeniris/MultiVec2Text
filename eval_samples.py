import torch
from vec2text import analyze_utils
import datasets


samples = ["jack morris is a phd student at cornell tech in new york city",
        "it was the best of times, it was the worst of times, it was the age of wisdom",
        "in einer stunde erreichen wir kopenhagen.",
        "comment puis-je vous aider?"]


def evaluate_samples(trainer, device, samples):
    samples = [x.lower() for x in samples]
    max_seq_length = 32

    # tokenize the sampels.
    output = trainer.tokenizer(samples, return_tensors="pt", max_length=32, truncation=True, padding="max_length").to(device)
    output["labels"] = [
        [
            (-100 if token_id == trainer.tokenizer.pad_token_id else token_id)
            for token_id in ids
        ]
        for ids in output["input_ids"]
    ]
    embedder_output = trainer.embedder_tokenizer(
        samples,
        padding="max_length",
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt",
    )
    embedder_output = {f"embedder_{k}": v for k, v in embedder_output.items()}
    output["length"] = [
        (torch.tensor(input_ids) != trainer.tokenizer.pad_token_id).sum().item()
        for input_ids in output["input_ids"]
    ]
    tokenized_samples = {**output, **embedder_output}
    dataset_samples = datasets.Dataset.from_dict(tokenized_samples)
    # print(f"correction steps {num_corrections} sbeam {sbeam}")
    trainer.evaluate(dataset_samples)

def get_trainer(model_path="yiyic/t5_me5_base_mtg_en_fr_de_es_5m_32_corrector"):
    # Traceback (most recent call last):
    #   File "<stdin>", line 1, in <module>
    # TypeError: cannot unpack non-iterable Corrector object
    experiment, trainer = analyze_utils.load_experiment_and_trainer_from_pretrained(
        model_path, use_less_data=3000)


    device = experiment.training_args.device
    # only for correctors
    if "corrector" in model_path:
        trainer.model.call_embedding_model = trainer.inversion_trainer.model.call_embedding_model
        trainer.model.tokenizer = trainer.inversion_trainer.model.tokenizer
        trainer.model.embedder_tokenizer = trainer.inversion_trainer.model.embedder_tokenizer
        trainer.model.embedder = trainer.inversion_trainer.embedder

    return trainer, device

def trainer_attributes(trainer, experiment):
    device = experiment.training_args.device
    # only for correctors
    trainer.model.call_embedding_model = trainer.inversion_trainer.model.call_embedding_model
    trainer.model.tokenizer = trainer.inversion_trainer.model.tokenizer
    trainer.model.embedder_tokenizer = trainer.inversion_trainer.model.embedder_tokenizer
    trainer.model.embedder = trainer.inversion_trainer.embedder
    return trainer, device


# if __name__ == '__main__':
#     eval_samples(samples)

