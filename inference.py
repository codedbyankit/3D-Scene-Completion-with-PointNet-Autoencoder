import torch
import open3d as o3d
import numpy as np
from model import PointNetAutoencoder
from config import config

def inference(input_ply):
    model = PointNetAutoencoder()
    model.load_state_dict(torch.load(config["save_path"]))
    model.eval()

    pcd = o3d.io.read_point_cloud(input_ply)
    points = np.asarray(pcd.points)

    # Simulate partial input
    if points.shape[0] > 2048:
        points = points[np.random.choice(points.shape[0], 2048, replace=False), :]
    
    input_tensor = torch.tensor(points, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_tensor).cpu().numpy()[0]

    return output

if __name__ == "__main__":
    result = inference("input.ply")
    print("Inference complete.")
