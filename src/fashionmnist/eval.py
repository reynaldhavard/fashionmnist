import os

import torch
from torchvision import transforms

from fashionmnist.load_config import get_config
from fashionmnist.data_setup import create_dataloaders
from fashionmnist.models import get_model
from fashionmnist.utils import load_checkpoint
from fashionmnist.eval_utils import (
    get_predictions,
    plot_incorrect_predictions,
    plot_confusion_matrix,
    get_accuracy,
)


def eval_model(
    model_file, dataloader, class_names, save_model_path, device, plot=False
):
    print(f"Evaluating {model_file}")
    checkpoint = load_checkpoint(os.path.join(save_model_path, model_file))

    model = get_model(
        model_name=checkpoint["model_name"],
        output_shape=len(class_names),
        hidden_units=checkpoint["hidden_units"],
        device=device,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    y_true, y_pred = get_predictions(model, dataloader, device)
    accuracy = get_accuracy(y_true, y_pred)
    print(f"Test accuracy: {accuracy}")

    if plot:
        plot_confusion_matrix(y_true, y_pred, class_names)
        plot_incorrect_predictions(
            model, dataloader, class_names, device, n_predictions=16
        )


if __name__ == "__main__":
    config = get_config("config_files/eval_config.yaml")

    if torch.cuda.is_available() and config["allow_gpu"]:
        device = "cuda"
    else:
        device = "cpu"

    _, _, test_dataloader, class_names = create_dataloaders(
        data_path=config["data_path"],
        train_ratio=0.5,
        transforms=transforms.ToTensor(),
        batch_size=1,
    )

    model_files = [
        f
        for f in os.listdir(config["save_model_path"])
        if os.path.isfile(os.path.join(config["save_model_path"], f))
    ]
    for model_file in model_files:
        eval_model(
            model_file=model_file,
            dataloader=test_dataloader,
            class_names=class_names,
            save_model_path=config["save_model_path"],
            device=device,
            plot=False,
        )
