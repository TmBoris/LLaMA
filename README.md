<h1 align="center">LLaMA</h1>

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
   <a href="#final-results">Final results</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

## About

This repository contains the end-to-end pipeline for training text generation model with PyTorch. The model was implemented is [LLaMA](https://arxiv.org/pdf/2302.13971).

See the task assignment [here](https://github.com/ashaba1in/hse-nlp/blob/main/2024/week5_modern_llms/homework/README.md).

See [wandb report](https://api.wandb.ai/links/mate-ball/svjn9sqi) with all experiments.

## Installation

Follow these steps to install the project:

0. (Optional) Create and activate new environment
   using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html).

   ```bash
   # create env
   conda create -n llama python=3.11

   # activate env
   conda activate llama
   ```

1. Install all required packages.

   ```bash
   pip install -r requirements.txt

2. Download dataset and tokenizer.
   ```bash
   python prepare.py

### Training

The model training contains of 2 stages. To reproduce results, train model using the following commands:

1. Train 101 epochs as pretrain on 256 len sequences

   ```bash
   python train.py writer.run_name="part1" dataloader.batch_size=3 trainer.seqs_from_sample=16
   model.rope_coef=1 model.max_seq_len=256 lr_scheduler.max_lr=5e-4
   ```

2. Train 10 epochs as fine-tune on 104 len sequences

   ```bash
   python train.py writer.run_name="part2" dataloader.batch_size=3 trainer.seqs_from_sample=3
   model.rope_coef=0.25 model.max_seq_len=1024
   ```

It takes around 26 hours to train model from scratch on V100 GPU.

## Final results

This results were obtained using argmax and language model:


```angular2html
            RTI     
tinyMMLU    0.31
```


## Credits

This repository is based on a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)