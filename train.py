# train.py

import torch
from torch.utils.data import DataLoader
from model import PointNetAutoencoder
from dataset import ShapeNetPartialDataset
from losses import ChamferLoss
from logger import Logger
from config import config
import torch.nn.functional as F
import argparse


def train(model, dataloader, optimizer, loss_fn, logger, num_epochs=20):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for i, (partial, full) in enumerate(dataloader):
            optimizer.zero_grad()
            output = model(partial.squeeze(0))
            gt = full[:, :2048, :]
            loss = loss_fn(output, gt.squeeze(0))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            logger.log_batch(loss.item(), epoch, i)

        avg_loss = total_loss / len(dataloader)
        logger.log_epoch(avg_loss, epoch)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 3D Scene Completion Model")
    parser.add_argument('--epochs', type=int, default=config["num_epochs"], help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=config["learning_rate"], help='Learning rate')
    args = parser.parse_args()

    dataset = ShapeNetPartialDataset(config["data_root"], augment=True)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    model = PointNetAutoencoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = ChamferLoss()
    logger = Logger(log_dir="logs")

    print("Training starts...")
    train(model, dataloader, optimizer, loss_fn, logger, num_epochs=args.epochs)

    torch.save(model.state_dict(), config["save_path"])
    print("Model saved at", config["save_path"])
