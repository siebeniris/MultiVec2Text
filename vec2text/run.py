import transformers

from vec2text.experiments import experiment_from_args
from vec2text.run_args import DataArguments, ModelArguments, TrainingArguments
import torch


def main():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    print("i have this many devices:", torch.cuda.device_count())
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # model_args, data_args, training_args = parser.parse_yaml_file(yaml_file)
    experiment = experiment_from_args(model_args, data_args, training_args)
    experiment.run()


if __name__ == "__main__":
    main()
