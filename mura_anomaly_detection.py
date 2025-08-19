"""
Script: MURA X‑ray anomaly detection with Grad‑CAM
==================================================

This script demonstrates how to build and train a convolutional neural
network on the **MURA** musculoskeletal radiograph dataset to classify
images as normal (negative) or abnormal (positive), and how to
highlight the regions responsible for the prediction using **Grad‑CAM**.

The **MURA** dataset contains 40 561 X‑ray images labelled as normal or
abnormal across seven musculoskeletal body parts (elbow, finger,
forearm, hand, humerus, shoulder and wrist)【812863805455203†L480-L483】.  Each
study is provided as a folder of one or more images.  In the original
study by Rajpurkar et al., a DenseNet‑169 pre‑trained on ImageNet and
fine‑tuned on MURA achieved an AUROC of 0.929, with sensitivity and
specificity of 0.815 and 0.887 respectively【812863805455203†L585-L591】.

This script does not download or redistribute the MURA dataset because
the dataset is large and subject to Stanford’s data use agreement.
Instead, it expects you to download and extract the dataset yourself
from the official competition website (https://stanfordmlgroup.github.io/competitions/mura/).
After accepting the license, place the extracted folder (e.g.
``MURA-v1.1``) in a directory of your choice and pass the path to
``--data-dir`` when running this script.

Features
--------

* **Data loader**: Scans the MURA directory structure to build a
  dataset of individual images with labels 0 (normal) and 1 (abnormal).
* **Transfer learning**: Uses a pre‑trained DenseNet‑169 or
  ResNet‑50 as a feature extractor and replaces the final
  classification layer with a two‑output head.
* **Training loop**: Includes a simple training routine with
  optional validation split and checkpointing.
* **Grad‑CAM**: Generates a class activation heatmap from the last
  convolutional layer and overlays it on the original X‑ray to
  visualise the region that contributed most to the model’s decision.
* **Command‑line interface**: Allows training a model, loading a
  checkpoint, and running inference on individual images from the
  command line.

Note
----

Training a high‑accuracy MURA classifier requires considerable
computational resources.  The code provided here is intended as a
starting point and demonstration; you may need to adjust
hyper‑parameters (learning rate, batch size, number of epochs) and
apply more advanced techniques (data augmentation, class balancing,
ensemble methods) to reproduce state‑of‑the‑art performance【812863805455203†L585-L591】.
The Grad‑CAM implementation is kept simple for clarity and ease of
implementation, prioritising interpretability rather than speed.
"""

import argparse
import os
from pathlib import Path
from typing import List, Tuple, Optional
import torch
torch.backends.cudnn.benchmark = True
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from torch.amp import autocast, GradScaler

scaler = GradScaler()


class MURADataset(Dataset):
    """
    MURA structure (each subset):
      subset/                         (train or valid)
        XR_<PART>/                    (e.g., XR_SHOULDER)
          patientXXXXXXXX/            (e.g., patient12345678)
            study1_positive/          (or studyX_negative)
              image1.png
              image2.png
            study2_negative/
              image1.png
              ...
    """
    IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    def __init__(
        self,
        root_dir: str,
        subset: str = "train",
        transform: Optional[transforms.Compose] = None,
        tasks: Optional[List[str]] = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        if subset not in {"train", "valid"}:
            raise ValueError("subset must be 'train' or 'valid'")
        self.subset = subset
        self.transform = transform or transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self.tasks = [t.upper() for t in tasks] if tasks else None
        self.samples: List[Tuple[Path, int]] = []
        self._load_samples()

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No images found under {self.root_dir / self.subset}. "
                "Expected structure: XR_* / patient* / study*_(positive|negative) / *.png"
            )

    def _load_samples(self) -> None:
        subset_path = self.root_dir / self.subset
        if not subset_path.exists():
            raise FileNotFoundError(f"Subset path not found: {subset_path}")

        # XR_<PART>
        for body_dir in sorted([p for p in subset_path.iterdir() if p.is_dir()]):
            if not body_dir.name.upper().startswith("XR_"):
                continue
            # e.g. XR_SHOULDER -> SHOULDER
            body_part = body_dir.name.split("_", 1)[1].upper() if "_" in body_dir.name else body_dir.name.upper()
            if self.tasks and body_part not in self.tasks:
                continue

            # patient folders
            for patient_dir in sorted([p for p in body_dir.iterdir() if p.is_dir()]):
                # study folders: studyX_positive / studyY_negative
                for study_dir in sorted([p for p in patient_dir.iterdir() if p.is_dir()]):
                    low = study_dir.name.lower()
                    if "positive" in low:
                        label = 1
                    elif "negative" in low:
                        label = 0
                    else:
                        continue  # skip unexpected dirs

                    # images inside the study
                    for img_path in study_dir.iterdir():
                        if img_path.suffix.lower() in self.IMG_EXTS:
                            self.samples.append((img_path, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("L")
        if self.transform:
            img = self.transform(img)
        return img, label, str(img_path)


def build_model(model_name: str = "densenet169", num_classes: int = 2) -> nn.Module:
    """Create a transfer learning model.

    Supported models: ``densenet169`` (default) and ``resnet50``.

    Parameters
    ----------
    model_name : str
        Name of the model architecture.
    num_classes : int
        Number of output classes.

    Returns
    -------
    nn.Module
        A PyTorch model with the final classification layer adapted for
        ``num_classes`` outputs.
    """
    if model_name.lower() == "densenet169":
        model = models.densenet169(weights=models.DenseNet169_Weights.DEFAULT)
        num_feats = model.classifier.in_features
        model.classifier = nn.Linear(num_feats, num_classes)
    elif model_name.lower() == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        num_feats = model.fc.in_features
        model.fc = nn.Linear(num_feats, num_classes)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    return model



def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 5,
    learning_rate: float = 1e-4,
    output_path: Optional[str] = None,
) -> None:
    """Train the model (with AMP on CUDA) and optionally save the best checkpoint."""
    from torch.amp import autocast, GradScaler

    amp_enabled = (device.type == "cuda")
    scaler = GradScaler(enabled=amp_enabled)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_acc = 0.0

    for epoch in range(epochs):
        # -------------------- TRAIN --------------------
        model.train()
        correct = 0
        total = 0
        running_loss = 0.0

        for images, labels, _ in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # Mixed precision forward + loss
            with autocast(device_type="cuda", enabled=amp_enabled):
                outputs = model(images)
                loss = criterion(outputs, labels)

            # Backward + step com GradScaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total if total > 0 else 0.0
        train_loss = running_loss / total if total > 0 else 0.0

        # -------------------- EVAL ---------------------
        model.eval()
        val_correct = 0
        val_total = 0
        val_running_loss = 0.0

        # inference_mode() é mais rápido que no_grad() e seguro para validação
        with torch.inference_mode():
            for images, labels, _ in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                # Você pode usar autocast na validação para acelerar
                with autocast(device_type="cuda", enabled=amp_enabled):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_running_loss += loss.item() * images.size(0)
                _, preds = outputs.max(1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total if val_total > 0 else 0.0
        val_loss = val_running_loss / val_total if val_total > 0 else 0.0

        print(
            f"Epoch {epoch + 1}/{epochs}\t"
            f"Train Loss: {train_loss:.4f}\tTrain Acc: {train_acc:.4f}\t"
            f"Val Loss: {val_loss:.4f}\tVal Acc: {val_acc:.4f}"
        )

        # Salva o melhor checkpoint por acurácia de validação
        if output_path and val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_path)
            print(f"Saved best model to {output_path} (val acc {val_acc:.4f})")



def generate_gradcam(
    model: nn.Module,
    image_tensor: torch.Tensor,
    target_class: Optional[int] = None,
    last_conv_layer: str = "features",
) -> np.ndarray:
    """Compute a Grad‑CAM heatmap for a single image.

    Grad‑CAM works by computing the gradients of the target class score
    with respect to the feature maps of a chosen convolutional layer.
    These gradients are averaged and used to weight the feature maps,
    producing a coarse localisation map highlighting salient regions.

    Parameters
    ----------
    model : nn.Module
        A trained model in evaluation mode.
    image_tensor : torch.Tensor
        Input image tensor of shape (C, H, W).  Do not include the batch
        dimension.
    target_class : int, optional
        Index of the class to compute the CAM for.  If ``None``, the
        class with the highest predicted score is used.
    last_conv_layer : str, default "features"
        Name of the final convolutional layer to use for Grad‑CAM.  For
        DenseNet this is "features"; for ResNet use ``"layer4"``.

    Returns
    -------
    np.ndarray
        A 2‑D heatmap rescaled to the input size, normalised to [0, 1].
    """
    model.eval()
    # Ensure input has batch dimension
    input_tensor = image_tensor.unsqueeze(0)
    input_tensor = input_tensor.requires_grad_(True)
    # Hook to capture activations and gradients
    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output.detach())

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    # Register hooks on the chosen convolutional layer
    for name, module in model.named_modules():
        if name == last_conv_layer:
            module.register_forward_hook(forward_hook)
            module.register_backward_hook(backward_hook)
    # Forward pass
    outputs = model(input_tensor)
    if target_class is None:
        target_class = outputs.argmax(dim=1).item()
    # Backward pass: compute gradients of the target class score
    target = outputs[0, target_class]
    model.zero_grad()
    target.backward()
    # Retrieve captured activations and gradients
    activation = activations[0]  # shape (1, C, H, W)
    gradient = gradients[0]      # shape (1, C, H, W)
    # Average the gradients across spatial dimensions to obtain
    # importance weights
    weights = gradient.mean(dim=[2, 3], keepdim=True)  # shape (1, C, 1, 1)
    # Weighted combination of feature maps
    cam = (weights * activation).sum(dim=1, keepdim=True)  # shape (1, 1, H, W)
    cam = torch.relu(cam)
    # Normalize to [0,1]
    cam -= cam.min()
    if cam.max() > 0:
        cam /= cam.max()
    # Upsample to input size
    cam_np = cam.squeeze().cpu().numpy()
    cam_np = np.uint8(cam_np * 255)
    cam_img = Image.fromarray(cam_np).resize((image_tensor.size(2), image_tensor.size(1)), Image.BILINEAR)
    cam_np = np.array(cam_img, dtype=np.float32) / 255.0
    return cam_np


def overlay_heatmap(
    orig_image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: str = "jet",
) -> np.ndarray:
    """Overlay a heatmap onto the original image.

    The input image should be a 2‑D grayscale array normalised to [0,1].

    Returns an RGB image with the heatmap overlaid.
    """
    # Convert grayscale image to RGB
    orig_rgb = np.stack([orig_image] * 3, axis=-1)
    cmap = plt.get_cmap(colormap)
    heatmap_rgb = cmap(heatmap)[:, :, :3]  # ignore alpha channel
    overlay = orig_rgb * (1 - alpha) + heatmap_rgb * alpha
    overlay = np.clip(overlay, 0, 1)
    return overlay

def cam_to_mask(heatmap: np.ndarray, percentile: float = 85.0) -> np.ndarray:
    """Binariza o mapa de importância pelo percentil (0–100)."""
    heatmap = np.asarray(heatmap, dtype=np.float32)
    thr = np.percentile(heatmap, percentile)
    mask = (heatmap >= thr).astype(np.uint8)
    return mask


def _connected_components(mask: np.ndarray) -> List[np.ndarray]:
    """Componentes conectados 4-vizinhos (puro NumPy, sem SciPy). Retorna listas de (y,x)."""
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: List[np.ndarray] = []
    for y in range(h):
        for x in range(w):
            if mask[y, x] and not visited[y, x]:
                stack = [(y, x)]
                visited[y, x] = True
                coords = []
                while stack:
                    cy, cx = stack.pop()
                    coords.append((cy, cx))
                    for ny, nx in ((cy-1, cx), (cy+1, cx), (cy, cx-1), (cy, cx+1)):
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
                comps.append(np.array(coords, dtype=np.int32))
    return comps


def mask_to_bboxes(mask: np.ndarray, top_k: int = 3, min_area: int = 30) -> List[Tuple[int,int,int,int,int]]:
    """Converte máscara binária em até K bounding boxes (x0,y0,x1,y1,area), maiores primeiro."""
    comps = _connected_components(mask.astype(np.uint8))
    boxes = []
    for comp in comps:
        ys, xs = comp[:, 0], comp[:, 1]
        area = len(comp)
        if area < min_area:
            continue
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
        boxes.append((x0, y0, x1, y1, area))
    boxes.sort(key=lambda b: b[4], reverse=True)
    return boxes[:top_k]


def draw_bboxes_on_image(img_gray: np.ndarray, boxes: List[Tuple[int,int,int,int,int]], title: Optional[str] = None):
    """Desenha retângulos e rótulos ROI sobre a imagem."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img_gray, cmap="gray")
    for i, (x0, y0, x1, y1, area) in enumerate(boxes, 1):
        rect = plt.Rectangle((x0, y0), (x1 - x0 + 1), (y1 - y0 + 1),
                             fill=False, edgecolor="lime", linewidth=2)
        ax.add_patch(rect)
        ax.text(x0, max(y0 - 4, 0), f"ROI {i}", color="lime", fontsize=9, weight="bold")
    ax.axis("off")
    if title:
        ax.set_title(title)
    plt.tight_layout()
    plt.show()


def gradient_saliency(model: nn.Module, image_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
    """Saliency |d score / d input|, colapsando canais -> mapa [0,1] do mesmo tamanho."""
    model.eval()
    x = image_tensor.unsqueeze(0).clone().detach().requires_grad_(True)
    output = model(x)
    if target_class is None:
        target_class = int(output.argmax(dim=1).item())
    score = output[0, target_class]
    model.zero_grad(set_to_none=True)
    score.backward()
    sal = x.grad.detach().abs().squeeze(0)  # C,H,W
    sal = sal.max(dim=0)[0].cpu().numpy()   # H,W
    sal -= sal.min()
    if sal.max() > 0:
        sal /= sal.max()
    return sal


def draw_contours(img_gray: np.ndarray, mask: np.ndarray, title: Optional[str] = None):
    """Desenha contorno binário (linha) ao invés de heatmap."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img_gray, cmap="gray")
    ax.contour(mask.astype(float), levels=[0.5], colors="red", linewidths=1.5, origin="upper")
    ax.axis("off")
    if title:
        ax.set_title(title)
    plt.tight_layout()
    plt.show()


def occlusion_importance_map(
    model: nn.Module,
    image_tensor: torch.Tensor,
    target_class: Optional[int] = None,
    window: int = 32,
    stride: int = 16,
) -> np.ndarray:
    """Mapa de importância por oclusão: quanto cai o logit ao zerar blocos (normalizado [0,1])."""
    model.eval()
    x = image_tensor.unsqueeze(0)
    with torch.no_grad():
        base_out = model(x)
        if target_class is None:
            target_class = int(base_out.argmax(dim=1).item())
        base_score = float(base_out[0, target_class].item())

    _, H, W = image_tensor.shape
    imp = np.zeros((H, W), dtype=np.float32)
    cnt = np.zeros((H, W), dtype=np.float32)

    for y in range(0, H - window + 1, stride):
        for x0 in range(0, W - window + 1, stride):
            xocc = x.clone()
            xocc[:, :, y:y+window, x0:x0+window] = 0.0
            with torch.no_grad():
                o = model(xocc)
            drop = base_score - float(o[0, target_class].item())
            drop = max(drop, 0.0)
            imp[y:y+window, x0:x0+window] += drop
            cnt[y:y+window, x0:x0+window] += 1.0

    cnt[cnt == 0] = 1.0
    imp /= cnt
    imp -= imp.min()
    if imp.max() > 0:
        imp /= imp.max()
    return imp


def predict_and_visualize(
    model: nn.Module,
    image_path: str,
    device: torch.device,
    last_conv_layer: str = "features",
    viz: str = "bbox",                 # 'bbox' | 'saliency' | 'occlusion' | 'cam'
    topk: int = 3,
    percentile: float = 85.0,
    occl_window: int = 32,
    occl_stride: int = 16,
) -> None:
    """Run inference on a single image and visualize with non-heatmap hints."""
    # Prepare transforms consistent with training
    prep = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    # Load and preprocess the image
    img_pil = Image.open(image_path).convert("L")
    img_tensor = prep(img_pil).to(device)
    # Keep a copy of the unnormalised image for visualisation
    img_np = np.array(img_pil.resize((224, 224)), dtype=np.float32) / 255.0

    # Forward pass -> prediction
    model.eval()
    with torch.no_grad():
        output = model(img_tensor.unsqueeze(0))
        probs = nn.functional.softmax(output, dim=1)
        pred_class = probs.argmax(dim=1).item()
        confidence = probs[0, pred_class].item()
    label_str = "abnormal" if pred_class == 1 else "normal"
    title = f"Prediction: {label_str} (confidence: {confidence:.2f})"

    # Choose visualization
    if viz == "bbox":
        # Use Grad-CAM internally, but mostra apenas caixas/contornos, nada de heatmap visível
        heatmap = generate_gradcam(model, img_tensor, target_class=pred_class, last_conv_layer=last_conv_layer)
        mask = cam_to_mask(heatmap, percentile=percentile)
        boxes = mask_to_bboxes(mask, top_k=topk, min_area=30)
        if boxes:
            draw_bboxes_on_image(img_np, boxes, title)
        else:
            # fallback para contorno do próprio mask
            draw_contours(img_np, mask, title)

    elif viz == "saliency":
        sal = gradient_saliency(model, img_tensor, target_class=pred_class)
        mask = cam_to_mask(sal, percentile=max(50.0, percentile))  # geralmente saliency é mais esparso
        draw_contours(img_np, mask, title)

    elif viz == "occlusion":
        imp = occlusion_importance_map(model, img_tensor, target_class=pred_class,
                                       window=occl_window, stride=occl_stride)
        mask = cam_to_mask(imp, percentile=percentile)
        boxes = mask_to_bboxes(mask, top_k=topk, min_area=30)
        if boxes:
            draw_bboxes_on_image(img_np, boxes, title)
        else:
            draw_contours(img_np, mask, title)

    else:  # 'cam' -> mantém o antigo Grad-CAM/overlay, caso queira comparar
        heatmap = generate_gradcam(model, img_tensor, target_class=pred_class, last_conv_layer=last_conv_layer)
        overlay = overlay_heatmap(img_np, heatmap, alpha=0.5)
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 3, 1); plt.title("Original"); plt.imshow(img_np, cmap="gray"); plt.axis("off")
        plt.subplot(1, 3, 2); plt.title("Grad-CAM"); plt.imshow(heatmap, cmap="jet"); plt.axis("off")
        plt.subplot(1, 3, 3); plt.title("Overlay");  plt.imshow(overlay); plt.axis("off")
        plt.suptitle(title); plt.tight_layout(); plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MURA anomaly detection training and inference")
    parser.add_argument("--data-dir", type=str, help="Path to the MURA-v1.1 dataset directory")
    parser.add_argument("--model", type=str, default="densenet169", choices=["densenet169", "resnet50"], help="Model architecture to use")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--val-split", type=float, default=0.2, help="Fraction of training data to use for validation")
    parser.add_argument("--tasks", nargs="*", default=None, help="List of body parts to include (e.g. ELBOW SHOULDER). Default is all.")
    parser.add_argument("--checkpoint", type=str, default="mura_best.pth", help="Path to save or load a model checkpoint")
    parser.add_argument("--predict", type=str, default=None, help="Path to an image for inference and Grad-CAM visualisation")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use (cuda or cpu)")
    parser.add_argument("--num-workers", type=int, default=0 if os.name == "nt" else 4, help="DataLoader workers (Windows is safer with 0)")
    parser.add_argument("--viz", type=str, default="bbox", choices=["bbox", "saliency", "occlusion", "cam"], help="Tipo de pista visual: 'bbox' (caixas), 'saliency' (contornos por gradiente), 'occlusion' (caixas via oclusão) ou 'cam' (overlay clássico).")
    parser.add_argument("--topk-roi", type=int, default=3, help="Quantidade máxima de ROIs a destacar")
    parser.add_argument("--roi-percentile", type=float, default=85.0, help="Percentil p/ binarizar mapas de importância")
    parser.add_argument("--occl-window", type=int, default=32, help="Tamanho da janela de oclusão (px)")
    parser.add_argument("--occl-stride", type=int, default=16, help="Stride da janela de oclusão (px)")


    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    # Build or load model
    model = build_model(args.model, num_classes=2).to(device)
    if args.predict and os.path.exists(args.checkpoint):
        # Load checkpoint for inference
        state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
    if args.predict:
        # Perform inference and visualisation
        predict_and_visualize(
            model, args.predict, device,
            last_conv_layer="features" if args.model.startswith("densenet") else "layer4",
            viz=args.viz,
            topk=args.topk_roi,
            percentile=args.roi_percentile,
            occl_window=args.occl_window,
            occl_stride=args.occl_stride,
        )
        return
    # Otherwise train the model
    dataset = MURADataset(args.data_dir, subset="train", tasks=args.tasks)
    print(f"[MURA] Found {dataset} train images "
        f"(tasks={args.tasks if args.tasks else 'ALL'}) under {args.data_dir}")

    # Split into training and validation sets
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=args.num_workers,
                            pin_memory=True,
                            persistent_workers=(args.num_workers > 0))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            pin_memory=True,
                            persistent_workers=(args.num_workers > 0))
    train_model(
        model,
        train_loader,
        val_loader,
        device,
        epochs=args.epochs,
        learning_rate=args.lr,
        output_path=args.checkpoint,
    )


if __name__ == "__main__":
    main()