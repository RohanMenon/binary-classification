# Binary Classification Project

This project focuses on binary classification using deep learning techniques. It classifies images into two categories: chihuahua and muffin. The project utilizes the pre-trained Vision transformer from `timm` library for model training and evaluation, and it is built with Python 3.12 or higher.

## Setup
Install uv package manager and install dependencies. Refer [here](https://docs.astral.sh/uv/concepts/projects/dependencies/) for help.

## Finetune pretrained model on custom dataset
`python src/train.py`

Trained checkpoints are stored under `./checkpoints` 