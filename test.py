import torch
from model import PointNetAutoencoder
from dataset import ShapeNetPartialDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from config import config

def test(model, dataloader):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for partial, full in dataloader:
            output = model(partial)
            loss = F.mse_loss(output, full)
            total_loss += loss.item()
    return total_loss / len(dataloader)

if __name__ == "__main__":
    dataset = ShapeNetPartialDataset(config["data_root"])
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    model = PointNetAutoencoder()
    model.load_state_dict(torch.load(config["save_path"]))
    loss = test(model, dataloader)
    print(f"Test Loss: {loss:.4f}")
