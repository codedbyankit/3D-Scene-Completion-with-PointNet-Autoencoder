# 3D Scene Completion with PointNet Autoencoder

This project started as an exploration of how neural networks can complete missing parts of 3D scenes — like reconstructing a broken vase or restoring incomplete scans. It’s built around a simple PointNet-style autoencoder trained on partial point clouds from ShapeNet.

## 🧠 What It Does
- Takes in partial 3D point clouds (like damaged or occluded objects).
- Learns to generate complete versions using a lightweight neural model.
- Evaluates performance using **Chamfer Distance**.
- Lets you **visualize inputs and reconstructions** to see how well it’s doing.

## ✨ Why I Built It
I’ve always found it fascinating how models can "fill in the blanks," whether it’s with text, images, or 3D data. This project gave me the perfect opportunity to explore that idea further, especially in the context of geometric deep learning. It allowed me to dive into working with autoencoders, data augmentation, and understanding 3D shapes. On top of that, it connects really well with the current research trends around generative models and 3D perception.

## 📂 Project Structure
- **train.py**: Main training loop.
- **test.py**: Evaluate the model on unseen data.
- **inference.py**: Run the model on a new `.ply` file.
- **model.py**: The PointNet-style autoencoder.
- **dataset.py**: Loads and occludes ShapeNet data.
- **utils.py**: Chamfer distance, visualizations, etc.
- **config.py**: Centralized configuration.
- **visualize_results.py**: Compare before/after shapes.
- **requirements.txt**: Dependencies.
- **README.md**: You’re reading it :).

## 📦 Dataset
Uses partial point clouds derived from **ShapeNet**. You’ll need to prepare or download `.ply` files and place them under `./data/ShapeNet/`.

## 🧪 Training
To train the model, simply run:

```bash
python train.py

## 📊 Evaluation
To evaluate the model's performance, run:

```bash
python test.py
```

## 📊 Evaluation
To evaluate the model's performance, run:
```bash
python test.py
```
This will output the average Chamfer Distance, a simple way to measure shape similarity.
## 🧍 Inference & Try It Out
For inference on a new .ply file, run:
```bash
python inference.py --input_path path/to/your/file.ply --output_path path/to/save/reconstructed.ply
```
It will reconstruct the complete shape, and you can visualize the results using Open3D. If you want to visualize the results:

```
python visualize_results.py --original path/to/your/file.ply --reconstructed path/to/save/reconstructed.ply
```
This script will display the original and reconstructed 3D shapes for comparison.

## 🚀 Try It Out
Make sure you have the dependencies:
```
pip install -r requirements.txt

```
okay guyz you are ready to goo !

## License
This project is licensed under the MIT License.







