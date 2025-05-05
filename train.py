import torch
from torch.utils.data import DataLoader
from model import PointNetAutoencoder
from dataset import ShapeNetPartialDataset
import torch.nn.functional as F
import torch.nn as nn
from config import config

def train(model, dataloader, optimizer, num_epochs=20):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for partial, full in dataloader:
            optimizer.zero_grad()
            output = model(partial.squeeze(0))
            gt = full[:, :2048, :]
            loss = F.mse_loss(output, gt.squeeze(0))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader):.4f}")

if __name__ == "__main__":
    dataset = ShapeNetPartialDataset(config["data_root"])
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    model = PointNetAutoencoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    print("Training starts...")
    train(model, dataloader, optimizer, num_epochs=config["num_epochs"])

    torch.save(model.state_dict(), config["save_path"])
    print("Model saved at", config["save_path"])
