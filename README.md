This project started as an exploration of how neural networks can complete missing parts of 3D scenes — like reconstructing a broken vase or restoring incomplete scans. It’s built around a simple PointNet-style autoencoder trained on partial point clouds from ShapeNet.

🧠 What It Does
Takes in partial 3D point clouds (like damaged or occluded objects).

Learns to generate complete versions using a lightweight neural model.

Evaluates performance using Chamfer Distance.

Lets you visualize inputs and reconstructions to see how well it’s doing.

✨ Why I Built It
I’ve always been fascinated by how models “fill in the blanks” — whether in text, images, or 3D. This project was a way to dive deeper into geometric deep learning and get some hands-on experience with autoencoders, data augmentation, and shape understanding. Also, it ties well with research trends around generative models and 3D perception.

📂 Project Structure
train.py: Main training loop

test.py: Evaluate the model on unseen data

inference.py: Run the model on a new .ply file

model.py: The PointNet-style autoencoder

dataset.py: Loads and occludes ShapeNet data

utils.py: Chamfer distance, visualizations, etc.

config.py: Centralized configuration

visualize_results.py: Compare before/after shapes

requirements.txt: Dependencies

README.md: You’re reading it :)

📦 Dataset
Uses partial point clouds derived from ShapeNet. You’ll need to prepare or download .ply files and place them under ./data/ShapeNet/.
🧪 Training 

---bash 
python train.py
-----
You can tweak hyperparameters from config.py. The model trains in under an hour on a decent GPU.

📊 Evaluation
---Bash
python test.py
Outputs average Chamfer Distance, a simple way to measure shape similarity.

🧍 Inference
---bash
python inference.py
Pass a .ply file and it reconstructs the complete shape. You can visualize results using Open3D.

🚀 Try It Out
Make sure you have the dependencies:
---bash
pip install -r requirements.txt
Done All set to Go....




