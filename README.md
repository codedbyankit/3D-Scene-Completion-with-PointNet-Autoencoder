# 3D Scene Completion with PointNet Autoencoder

This project uses a PointNet-based autoencoder to reconstruct 3D point clouds. The goal is to complete partial 3D point clouds, typically for applications in robotics and computer vision.

## Training
To train the model, run:
---bash
python train.py
---

## Inference
To run inference on a partial 3D model, use:
---bash
python inference.py
---

## Requirements
---bash---
pip install -r requirements.txt
----

## License
MIT License
