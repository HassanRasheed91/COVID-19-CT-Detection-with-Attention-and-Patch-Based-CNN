# COVID-19 Detection from CT Scans using Patch-Based Attention Ensemble

This project implements a novel deep learning architecture that detects COVID-19 from chest CT scan images using an ensemble of three pre-trained CNNs fused via an attention mechanism. The model focuses on lung patches to enhance early-stage detection accuracy and robustness.

## 📦 Project Structure

```
├── main.py              # Training script
├── ensemble.py          # Model with 3 CNN backbones + attention fusion
├── patch_extractor.py   # Splits CT image into patches
├── eval.py              # Evaluation on test data
├── predict.py           # Real-time single image prediction
├── gradcam.py           # Grad-CAM visualization for interpretability
├── run_gradcam.py       # Runs Grad-CAM visualization for a given image
├── slurm_job.sh         # SLURM script to run on HPC
├── colab_setup.txt      # Shell commands for Google Colab execution
```

---

## 🚀 Features
- Patch-based CT analysis for high resolution focus
- Uses ResNet50, EfficientNetB0, DenseNet121 backbones
- Learns adaptive attention weights for feature fusion
- Real-time prediction support
- Grad-CAM interpretability
- Colab & HPC compatible

---

## 🛠️ Installation

```bash
pip install torch torchvision matplotlib scikit-learn
```

---

## 📁 Dataset Structure
Prepare your training and test data folders like this:

```
data/
├── train/
│   ├── COVID/
│   └── Non-COVID/
├── test/
    ├── COVID/
    └── Non-COVID/
```
Images should be in `.png` or `.jpg` format.

---

## 🏃‍♂️ Run Training
```bash
python main.py
```

## 📊 Evaluate Model
```bash
python eval.py
```

## 🔍 Visualize Patch Attention (Grad-CAM)
```bash
python run_gradcam.py
```

## 🔮 Predict Single Image
```bash
python predict.py
```

---

## ☁️ Run on Google Colab
Follow commands in `colab_setup.txt`

---

## ⚡ Run on HPC with SLURM
```bash
sbatch slurm_job.sh
```

---

## 📄 License
MIT License

## 🙏 Acknowledgement
This work builds upon the fuzzy-integral COVID detection ensemble paper by Rohit Kundu et al. and enhances it using attention-based fusion and patch-level inference.

---

## 🔗 Citation
If you use this code in your research, please cite the original paper and consider referencing this implementation.
