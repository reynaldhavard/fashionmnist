import torch
from torchvision import transforms

from fashionmnist.load_config import get_config
from fashionmnist.data_setup import create_dataloaders
from fashionmnist.models import get_model
from fashionmnist.train_engine import get_loss_fn, get_optimizer, train


if __name__ == "__main__":
    torch.manual_seed(0)
    config = get_config("config_files/train_config.yaml")

    if torch.cuda.is_available() and config["allow_gpu"]:
        device = "cuda"
    else:
        device = "cpu"

    (
        train_dataloader,
        val_dataloader,
        test_dataloader,
        class_names,
    ) = create_dataloaders(
        data_path=config["data_path"],
        train_ratio=config["train_ratio"],
        transforms=transforms.ToTensor(),
        batch_size=config["batch_size"],
    )

    model = get_model(
        model_name=config["model_name"],
        output_shape=len(class_names),
        hidden_units=config["hidden_units"],
        device=device,
    )
    loss_fn = get_loss_fn()
    optimizer = get_optimizer(model, learning_rate=config["learning_rate"])

    results = train(
        model,
        train_dataloader,
        val_dataloader,
        loss_fn,
        optimizer,
        epochs=config["epochs"],
        device=device,
        save_model_path=config["save_model_path"],
        writer_path=config["writer_path"],
        extra_info={"hidden_units": config["hidden_units"]},
    )
