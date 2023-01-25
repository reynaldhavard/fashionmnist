# fashionmnist
Repository to develop process to solve the Fashion MNIST problem.
This repository allows to train 2 different neural networks on the Fashion MNIST
dataset.

## Requirements
- Install [`poetry`](https://python-poetry.org/docs/)
- Install `pre-commit` (via `pip`)

## Setup
- Execute `poetry install` to create a virtual environment
- Execute `poetry run pre-commit install`
- Execute `poetry run ipython kernel install --user --name=fashionmnist`: this
command allows you to use the created virtual environment in a jupyter notebook.

## Commands
- To start Jupyter Lab server, run: `poetry run jupyter lab`
- In order to run the training process, run:
`poetry run python src/fashionmnist/train.py` or have a look at the notebook
[`train.ipynb`](/notebooks/train.ipynb) (don't forget to select the
`fashionmnist` kernel before executing it)
- In order to run the evaluation process, run
`poetry run python src/fashionmnist/eval.py` or have a look at the notebook
[`eval.ipynb`](/notebooks/eval.ipynb)(don't forget to select the `fashionmnist`
kernel before executing it)
- To monitor the loss or the accuracy of the trained model on the training set
and the validation set, run `poetry run tensorboard --logdir=runs` or replace
`runs` by the path specified in
[`train_config.yaml`](/config_files/train_config.yaml)

## Models implemented
2 models have been implemented:
- a fully connected model with 2 linear layers (with BatchNorm and ReLU) denoted
  as `FCNetwork`
- a convolutional model with 2 convolutional layers (with BatchNorm, ReLU and
  MaxPooling) denoted as Conv2LayersNetwork
In [`train_config.yaml`](/config_files/train_config.yaml), it is possible to modify the number of hidden units for
these layers. There is currently only one value to define the different hidden
units in these models.

## Training process
- The training process starts by getting the Fashion MNIST data using the
  torchvision API.
- We split the training data into training and validation sets.
- The different splits are passed through a custom Dataset class and then
  through a DataLoader.
- Model, loss function and optimizers are initialized.
- For a certain number of epochs, we optimize the model with the training set,
  evaluate on the validation set and we save the model which had the lowest loss
  on the validation set. The metrics are saved and can be monitored with
  Tensorboard.
- The notebook [`train.ipynb`](/notebooks/train.ipynb) provides some additional information to the
  training process.

## Evaluation process
- The current evaluation process looks at the checkpoints present in the `models` folder (or different if specified in [`eval_config.yaml`](/config_files/eval_config.yaml)).
- For each model, we run the entire test set and get the accuracy of the model.
  Additional plots (confusion matrices and misclassified instances) are visible
  in the notebook [`eval.ipynb`]('/notebooks/eval.ipynb').

## Next steps
Some ideas for the next steps:
- Add unit tests.
- Add more transforms (horizontal flip for example).
- Add more flexibility to the models.
