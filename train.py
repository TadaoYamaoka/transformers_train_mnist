from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import count
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from accelerate import PartialState
from accelerate.utils import broadcast_object_list
from jsonargparse import ActionConfigFile, ArgumentParser
from torch.utils.data import Dataset, random_split
from torchvision import datasets, transforms
from transformers import EvalPrediction, Trainer, TrainingArguments, set_seed

LOGGER = logging.getLogger(__name__)


@dataclass
class DataConfig:
    data_dir: str = "data"
    train_split_ratio: float = 0.9
    download: bool = True


class MNISTForTrainer(Dataset):
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int]:
        image, label = self.dataset[index]
        return {"pixel_values": image, "labels": label}


class MnistCnn(nn.Module):
    def __init__(self, dropout: float = 0.25) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 10),
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self, pixel_values: torch.Tensor, labels: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        logits = self.classifier(self.features(pixel_values))
        outputs = {"logits": logits}
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            # Keep loss gather-compatible for DataParallel multi-GPU runs.
            outputs["loss"] = loss.unsqueeze(0)
        return outputs


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Train a CNN on MNIST with Transformers Trainer."
    )
    parser.add_argument(
        "--config", action=ActionConfigFile, help="Path to the YAML config file."
    )
    parser.add_class_arguments(DataConfig, "data")
    parser.add_class_arguments(MnistCnn, "model")
    parser.add_class_arguments(TrainingArguments, "training", instantiate=False)
    parser.add_argument(
        "--experiment_root",
        default=None,
        type=str,
        help="Base directory for versioned experiments. If omitted and training.output_dir is set, that path is used directly.",
    )
    return parser


def create_next_version_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for version in count():
        candidate = root / f"version_{version}"
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError("Failed to create a version directory")


def create_version_dir(experiment_root: str) -> Path:
    root = Path(experiment_root)
    state = PartialState()
    if state.num_processes <= 1:
        return create_next_version_dir(root)

    experiment_dir_holder: list[str | None] = [None]
    if state.is_main_process:
        experiment_dir_holder[0] = str(create_next_version_dir(root))

    broadcast_object_list(experiment_dir_holder, from_process=0)
    if experiment_dir_holder[0] is None:
        raise RuntimeError("Failed to broadcast experiment directory.")

    experiment_dir = Path(experiment_dir_holder[0])
    state.wait_for_everyone()
    return experiment_dir


def uses_versioned_output_dir(config: Any) -> bool:
    training_output_dir = (
        config.training.output_dir if config.training is not None else None
    )
    return config.experiment_root is not None or training_output_dir is None


def resolve_experiment_dir(config: Any, use_versioned_output_dir: bool) -> Path:
    if not use_versioned_output_dir:
        experiment_dir = Path(config.training.output_dir)
        experiment_dir.mkdir(parents=True, exist_ok=True)
        return experiment_dir
    experiment_root = config.experiment_root or "logs"
    return create_version_dir(experiment_root)


def resolve_training_args(config: Any, experiment_dir: Path) -> TrainingArguments:
    if config.training is None:
        raise ValueError("training must be defined in config.yaml")

    training_kwargs = config.training.as_dict()
    training_kwargs["output_dir"] = str(experiment_dir)
    return TrainingArguments(**training_kwargs)


def build_datasets(
    data_config: DataConfig, seed: int
) -> tuple[Dataset, Dataset, Dataset]:
    state = PartialState()
    if not 0.0 < data_config.train_split_ratio < 1.0:
        raise ValueError(
            f"train_split_ratio must be between 0 and 1, got {data_config.train_split_ratio}"
        )

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    with state.main_process_first():
        full_train = datasets.MNIST(
            data_config.data_dir,
            train=True,
            download=data_config.download,
            transform=transform,
        )
        test = datasets.MNIST(
            data_config.data_dir,
            train=False,
            download=data_config.download,
            transform=transform,
        )

    train_size = int(len(full_train) * data_config.train_split_ratio)
    val_size = len(full_train) - train_size
    generator = torch.Generator().manual_seed(seed)
    train, val = random_split(full_train, [train_size, val_size], generator=generator)
    return MNISTForTrainer(train), MNISTForTrainer(val), MNISTForTrainer(test)


def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
    predictions = eval_pred.predictions.argmax(axis=-1)
    accuracy = (predictions == eval_pred.label_ids).mean().item()
    return {"accuracy": accuracy}


def save_resolved_config(
    parser: ArgumentParser, config: Any, experiment_dir: Path
) -> None:
    state = PartialState()
    if state.is_main_process:
        if config.training is not None:
            config.training.output_dir = str(experiment_dir)
        parser.save(
            config,
            experiment_dir / "config.yaml",
            format="yaml",
            skip_none=False,
            overwrite=True,
            multifile=False,
        )
    state.wait_for_everyone()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    parser = build_parser()
    config = parser.parse_args()

    use_versioned_output_dir = uses_versioned_output_dir(config)
    experiment_dir = resolve_experiment_dir(config, use_versioned_output_dir)
    training_args = resolve_training_args(config, experiment_dir)
    set_seed(training_args.seed)
    split_seed = (
        training_args.data_seed
        if training_args.data_seed is not None
        else training_args.seed
    )
    save_resolved_config(parser, config, experiment_dir)
    instantiated = parser.instantiate(config)

    train_dataset, eval_dataset, test_dataset = build_datasets(
        instantiated.data, split_seed
    )
    model = instantiated.model

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    LOGGER.info("Experiment directory: %s", experiment_dir)
    resume_from_checkpoint = training_args.resume_from_checkpoint
    if use_versioned_output_dir and resume_from_checkpoint is True:
        raise ValueError(
            "training.resume_from_checkpoint=True cannot be used with a newly created versioned output directory. "
            "Pass an explicit checkpoint path instead."
        )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    test_metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
    trainer.log_metrics("test", test_metrics)
    trainer.save_metrics("test", test_metrics)
    trainer.save_model()
    trainer.save_state()


if __name__ == "__main__":
    main()
