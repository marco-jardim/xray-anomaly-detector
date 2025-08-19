# MURA X‑Ray Anomaly Detection Project

This project demonstrates how to train and use an anomaly detection model on human X‑ray images using the **MURA (Musculoskeletal Radiographs)** dataset.  It includes a Python script for training and inference, along with a step‑by‑step guide for downloading the full dataset, preparing your environment and running the model.

## 1. Overview of the MURA dataset

MURA contains **40,561 X‑ray images** of upper‑extremity skeletal structures (elbow, finger, forearm, hand, humerus, shoulder and wrist).  Each image is labelled as **normal (negative)** or **abnormal (positive)**【812863805455203†L480-L483】.  In 2018, Rajpurkar et al. showed that a **DenseNet‑169** pre‑trained on ImageNet and fine‑tuned on MURA achieved an **AUROC of 0.929**, with a sensitivity of 0.815 and specificity of 0.887【812863805455203†L585-L591】.

## 2. Project structure

```
MURA_Project/
├── README.md            ← this English guide
├── mura_anomaly_detection.py ← training & inference script (wrapper)
└── environment.yml       ← (optional) Conda environment specification
```

The script `mura_anomaly_detection.py` loads the MURA dataset, trains a binary classifier using transfer learning (DenseNet‑169 or ResNet‑50) and generates Grad‑CAM heatmaps to visualise which regions influenced the model’s decision.  For full documentation, see the docstring at the top of the script.

## 3. Downloading the dataset

You must create a Kaggle account and accept the MURA data usage agreement before downloading the full dataset.  Follow these steps:

1. Create an account on [Kaggle](https://www.kaggle.com/).
2. Go to the dataset page: <https://www.kaggle.com/datasets/cjinny/mura-v11/data>.
3. Click **“Download”** (you will be prompted to log in).  The file `mura-v11.zip` (or similar) will download to your machine.
4. Extract the contents into a directory of your choice, for example `~/Datasets/MURA-v1.1`.  Inside this directory you should have the folders `train/` and `valid/`.

> **Note:** the script does not download the dataset automatically; you must obtain it yourself from Kaggle due to licensing requirements.

## 4. Environment setup

It is recommended to create a Conda or virtualenv environment with the necessary libraries.  Example using Conda:

```bash
# Create and activate a new Python 3.9 environment
conda create -n mura_env python=3.11 -y
conda activate mura_env

# Install PyTorch (with GPU support if available)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install matplotlib pillow tqdm
conda install pytorch cudatoolkit -c pytorch

```

Alternatively, create a `requirements.txt` or `environment.yml` and install packages with `pip install -r requirements.txt`.

## 5. Training the model

To train using all images (or a subset) from the full MURA dataset, run the script as follows:

```bash
# Example: train DenseNet‑169 for 5 epochs
python mura_anomaly_detection.py \
    --data-dir ~/Datasets/MURA-v1.1 \
    --model densenet169 \
    --epochs 5 \
    --batch-size 16 \
    --lr 1e-4 \
    --checkpoint best_model.pth
```

Important parameters:

* `--data-dir`: directory where the extracted MURA dataset resides (`train/` and `valid/` must be inside).
* `--model`: choose the architecture (`densenet169` or `resnet50`).  DenseNet has shown strong performance【812863805455203†L585-L591】.
* `--epochs`: number of training epochs.  Increase this for higher accuracy.
* `--batch-size`: batch size.  Adjust based on available memory.
* `--lr`: learning rate for the optimiser.
* `--checkpoint`: path to save the best validation model.
* `--val-split`: fraction of the training set reserved for validation (default 0.2).
* `--tasks`: optional list of body parts (ELBOW, FINGER, etc.) to limit training to specific studies.

The training loop prints the loss and accuracy for the training and validation sets after each epoch and saves the checkpoint with the highest validation accuracy.  GPU acceleration is used if detected by PyTorch (`--device cuda` is the default when available).

## 6. Inference and Grad‑CAM visualisation

After training, load a saved checkpoint and run inference on a single X‑ray image, producing a heatmap that highlights the regions that drove the prediction:

```bash
python mura_anomaly_detection.py \
    --model densenet169 \
    --checkpoint best_model.pth \
    --predict path/to/image.png
```

The script displays three images: the original X‑ray, the Grad‑CAM heatmap and the overlay of the heatmap on the X‑ray.  It also prints the predicted class (normal or abnormal) and the confidence.

## 7. Tips for improving performance

* Use **more epochs** and/or **larger batch sizes** if you have sufficient GPU resources.
* Apply **data augmentation** (rotations, flips, brightness/contrast adjustments) to improve generalisation.
* Consider an **ensemble** of models (e.g., combining DenseNet and ResNet) to boost accuracy.
* Adjust the image normalisation if necessary; the script uses ImageNet statistics by default.

## 8. References

* Rajpurkar, P. et al. *MURA: Large Dataset for Abnormality Detection in Musculoskeletal Radiographs*. 2017.  The dataset contains 40,561 images labelled as normal or abnormal【812863805455203†L480-L483】.
* A DenseNet‑169 pre‑trained on ImageNet and fine‑tuned on MURA achieved an AUROC of 0.929 with sensitivity of 0.815 and specificity of 0.887【812863805455203†L585-L591】.