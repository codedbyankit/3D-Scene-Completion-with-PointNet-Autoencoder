import torch
from torch.utils.data import Dataset
import os
import numpy as np
import open3d as o3d

class ShapeNetPartialDataset(Dataset):
    def __init__(self, data_root):
        self.data_root = data_root
        self.files = [f for f in os.listdir(data_root) if f.endswith('.ply')]
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file = self.files[idx]
        full_path = os.path.join(self.data_root, file)
        point_cloud = o3d.io.read_point_cloud(full_path)
        points = np.asarray(point_cloud.points)
        
        # Simulating partial points by randomly removing points
        if points.shape[0] > 2048:
            points = points[np.random.choice(points.shape[0], 2048, replace=False), :]
        
        return torch.tensor(points, dtype=torch.float32), torch.tensor(points, dtype=torch.float32)
