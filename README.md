# Transformers Trainer MNIST CNN

Train a CNN on MNIST with Hugging Face Transformers `Trainer`.

## Setup

```sh
python -m venv .venv
```

Activate the virtual environment with the command for your shell and platform, then install dependencies:

```sh
python -m pip install -r requirements.txt
```

## Train

```sh
python train.py --config config.yaml
```

Run single-node distributed training with `torchrun`:

```sh
torchrun --nproc_per_node 2 train.py --config config.yaml
```

If `training.output_dir` is not set, the default experiment root is `logs`. Each run creates an incremented version directory such as `logs/version_0` or `logs/version_1`. Trainer outputs, checkpoints, TensorBoard logs, the resolved config, and the final model are written inside that version directory.

Override the experiment root from the command line without putting it in `config.yaml`:

```sh
python train.py --config config.yaml --experiment_root runs
```

If `--experiment_root` is omitted and `training.output_dir` is set, that output directory is used directly without creating a `version_*` subdirectory:

```sh
python train.py --config config.yaml --training.output_dir runs/manual_experiment
```

## Resume

Set the checkpoint path with the `training.resume_from_checkpoint` argument:

```sh
python train.py --config config.yaml --training.resume_from_checkpoint logs/version_0/checkpoint-844
```

## TensorBoard

```sh
tensorboard --logdir logs
```

## Optimizer And Scheduler

The optimizer and scheduler are configured in the `training` section of `config.yaml` using Hugging Face `TrainingArguments` fields such as `optim`, `learning_rate`, `weight_decay`, `lr_scheduler_type`, and `warmup_steps`. `Trainer` creates them internally.