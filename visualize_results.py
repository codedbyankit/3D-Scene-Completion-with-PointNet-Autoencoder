from utils import visualize_point_cloud
import numpy as np
import torch
from model import PointNetAutoencoder
from config import config

# Load the model
model = PointNetAutoencoder()
model.load_state_dict(torch.load(config["save_path"]))
model.eval()

# Example inference visualization
partial = np.load("example_partial.npy")  # [N, 3]
partial_tensor = torch.tensor(partial, dtype=torch.float32).unsqueeze(0)

with torch.no_grad():
    completed = model(partial_tensor.squeeze(0)).cpu().numpy()[0]

visualize_point_cloud(partial, title="Partial Input")
visualize_point_cloud(completed, title="Completed Output")
