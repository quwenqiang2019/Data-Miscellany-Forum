import hydra
from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data


class SimpleNet(nn.Module):
    def __init__(self, layers, units):
        super(SimpleNet, self).__init__()
        self.fc = nn.Sequential(
            *[nn.Linear(units, units) for _ in range(layers)] + [nn.ReLU()]
        )

    def forward(self, x):
        return self.fc(x)


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    model = SimpleNet(cfg.model.layers, cfg.model.units)
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.learning_rate)

    # Dummy dataset
    data = torch.randn((100, cfg.model.units))
    target = torch.randint(0, 2, (100,))
    dataset = Data.TensorDataset(data, target)
    train_loader = Data.DataLoader(dataset, batch_size=cfg.train.batch_size)

    for epoch in range(cfg.train.epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            optimizer.step()


if __name__ == "__main__":
    main()