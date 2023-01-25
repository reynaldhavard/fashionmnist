from torchvision import datasets
from torch.utils.data import Dataset, DataLoader, random_split


def load_original_data(path):
    """
    Get data from torchvision API.
    """
    train_data = datasets.FashionMNIST(path, train=True, download=True)
    test_data = datasets.FashionMNIST(path, train=False, download=True)

    class_names = train_data.classes

    return train_data, test_data, class_names


def create_random_split_train_val(train_data, train_ratio):
    """
    Split the train_data into train and validation data
    """
    train_size = int(train_ratio * len(train_data))
    val_size = len(train_data) - train_size
    train_data, val_data = random_split(train_data, [train_size, val_size])

    return train_data, val_data


class FashionMNISTDataset(Dataset):
    def __init__(self, dataset, transforms=None):
        self.dataset = dataset
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, label = self.dataset[index]
        if self.transforms is None:
            return img, label
        else:
            return self.transforms(img), label


def create_dataloaders(data_path, train_ratio, transforms, batch_size):
    """
    Load the original data, split the train data into train and validation
    data, create FashionMNISTDataset instances and pass them to DataLoaders
    """
    train_data, test_data, class_names = load_original_data(data_path)
    train_data, val_data = create_random_split_train_val(
        train_data, train_ratio
    )

    train_data = FashionMNISTDataset(dataset=train_data, transforms=transforms)
    val_data = FashionMNISTDataset(dataset=val_data, transforms=transforms)
    test_data = FashionMNISTDataset(dataset=test_data, transforms=transforms)

    train_dataloader = DataLoader(
        dataset=train_data, batch_size=batch_size, num_workers=0, shuffle=True
    )
    val_dataloader = DataLoader(
        dataset=val_data, batch_size=batch_size, num_workers=0, shuffle=False
    )
    test_dataloader = DataLoader(
        dataset=test_data, batch_size=1, num_workers=0, shuffle=False
    )

    return train_dataloader, val_dataloader, test_dataloader, class_names
