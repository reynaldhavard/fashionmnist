from torch import nn


class FCNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_units):
        super().__init__()
        self.hidden_units = hidden_units
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.BatchNorm1d(hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.BatchNorm1d(hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_shape),
        )

    def forward(self, x):
        out = self.model(x)
        return out

    def get_hidden_units(self):
        return self.hidden_units


class Conv2LayersNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_units):
        super().__init__()
        self.hidden_units = hidden_units
        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=input_shape,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(
                in_features=hidden_units * 5 * 5, out_features=output_shape
            ),
        )

    def forward(self, x):
        out = self.model(x)
        return out

    def get_hidden_units(self):
        return self.hidden_units


def get_model(model_name, output_shape, hidden_units, device):
    if model_name == "FCNetwork":
        model = FCNetwork(
            input_shape=28 * 28,
            output_shape=output_shape,
            hidden_units=hidden_units,
        ).to(device)
    elif model_name == "Conv2LayersNetwork":
        model = Conv2LayersNetwork(
            input_shape=1, output_shape=output_shape, hidden_units=hidden_units
        ).to(device)

    return model
