# === Basic Libraries ===
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import imageio
import os
from datetime import datetime
import time
from tqdm import tqdm
import warnings
from torchvision.models import ResNet101_Weights
from cityscapesscripts.helpers.labels import trainId2label
from sklearn.metrics import confusion_matrix
import json

warnings.filterwarnings("ignore", category=UserWarning, module="imageio")

# Ensure outputs save relative to this script's location
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# === config ===
# Set the path to the A2D2 image
input_folder = "/home/u509816/AV/datasets/a2d2_subset_test/front_center_images"
gt_folder = "/home/u509816/AV/datasets/a2d2_subset_test/labels"
output_root = "a2d2_outputs_results_4"
mapping_path = "/home/u509816/AV/scripts/a2d2_to_cityscapes_trainid.json"
os.makedirs(output_root, exist_ok=True)

with open(mapping_path, "r") as f:
    hex_mapping = json.load(f)

# Convert hex strings → RGB tuple → trainId
a2d2_to_cityscapes_trainId = {
    tuple(int(k.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)): v["trainId"] if isinstance(v, dict) else v
    for k, v in hex_mapping.items()
}



all_times = []   # List to store inference times


csv_path = os.path.join(output_root, "metrics_log.csv")
if not os.path.exists(csv_path):
    with open(csv_path, "w") as f:
        f.write("Filename,EntropyMean,EntropyMax,EntropyMin,HighEntropyRatio,InferenceTime,mIoU,PixelAccuracy\n")

# === Device Configuration ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# === Load the A2D2 Image ===
#image_path = "/Users/mahanabasiyan/Desktop/20181108141609_camera_frontcenter_000041751.png"  # Update if needed
#image = Image.open(image_path).convert("RGB")  # Ensure it's RGB

def compute_miou(preds, labels, num_classes=19):
    cm = confusion_matrix(labels.ravel(), preds.ravel(), labels=list(range(num_classes)))
    intersection = np.diag(cm)
    union = np.sum(cm, axis=0) + np.sum(cm, axis=1) - intersection
    iou = intersection / np.maximum(union, 1)
    return np.nanmean(iou), iou

def pixel_accuracy(preds, labels):
    return np.sum(preds == labels) / np.prod(labels.shape)

# Create a lookup table where colormap[trainId] = RGB color
colormap = np.zeros((256, 3), dtype=np.uint8)
for train_id, label in trainId2label.items():
    if train_id != 255:  # Ignore 'void' class
        colormap[train_id] = label.color

# Function to map a predicted mask (trainIds) to a color RGB image
def colorize(mask):
    return colormap[mask]

# === Define the Same Transforms as Used During Training ===
transform = transforms.Compose([
    transforms.Resize((512, 1024)),  # Match training resolution
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Cityscapes/ImageNet mean
                         std=[0.229, 0.224, 0.225])   # Cityscapes/ImageNet std
])

# === Define the Model Architecture ===
model = models.segmentation.deeplabv3_resnet101(
    weights=None,
    weights_backbone=ResNet101_Weights.IMAGENET1K_V1,
    aux_loss=True)

# Modify the classifier to match the number of Cityscapes classes (19)
model.classifier = nn.Sequential(
    nn.Conv2d(2048, 256, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Dropout2d(p=0.2),
    nn.Conv2d(256, 19, kernel_size=1)
)

model.load_state_dict(torch.load("/home/u509816/AV/scripts/output_train_cityscapes_4/best_model.pth", map_location=device))


# === Set Model to Evaluation Mode and Move to Device ===
model = model.to(device)
model.eval()
print("Model loaded and ready.")


all_mious = []      # List to store mIoU values
all_accuracies = []   # List to store pixel accuracies


# === Loop Over All PNG Files ===
file_list = sorted([f for f in os.listdir(input_folder) if f.endswith(".png")])
for filename in tqdm(file_list, desc="Processing images"):
    if not filename.endswith(".png"):
        continue

    # === Load and preprocess image ===
    img_path = os.path.join(input_folder, filename)
    image = Image.open(img_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    start_time = time.time()

    # === Prepare output folder for this image ===
    base_name = os.path.splitext(filename)[0]
    output_dir = os.path.join(output_root, base_name)
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        output = model(input_tensor)['out']
        probs = torch.softmax(output, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
        entropy_map = entropy.squeeze().cpu().numpy()
        high_entropy_ratio = np.mean(entropy_map > 1.5)
        pred_mask = torch.argmax(probs, dim=1).squeeze().cpu().numpy()
        
        # Save raw logits for further post-processing
        np.save(os.path.join(output_dir, "logits.npy"), output.squeeze().cpu().numpy())

        # === Evaluate against ground truth ===
        gt_path = os.path.join(gt_folder, filename)
        if os.path.exists(gt_path):
            gt_raw = Image.open(gt_path).resize((1024, 512), resample=Image.NEAREST).convert("RGB")
            gt_mask = np.array(gt_raw).astype(np.uint8)

            remapped_mask = 255 * np.ones((gt_mask.shape[0], gt_mask.shape[1]), dtype=np.uint8)
            a2d2_rgb_keys = np.array(list(a2d2_to_cityscapes_trainId.keys()))
            for r, g, b in a2d2_rgb_keys:
                cs_id = a2d2_to_cityscapes_trainId[(r, g, b)]
                mask = (gt_mask[:, :, 0] == r) & (gt_mask[:, :, 1] == g) & (gt_mask[:, :, 2] == b)
                remapped_mask[mask] = cs_id


            gt_mask = remapped_mask


            valid = (gt_mask != 255)  
            pred_eval = pred_mask[valid]
            gt_eval = gt_mask[valid]

            if pred_eval.size > 0:
                miou_val, _ = compute_miou(pred_eval, gt_eval)
                acc_val = pixel_accuracy(pred_eval, gt_eval)
            else:
                miou_val, acc_val = np.nan, np.nan
        else:
            miou_val, acc_val = np.nan, np.nan
            
        if not np.isnan(miou_val):
            all_mious.append(miou_val)
            all_accuracies.append(acc_val)

    elapsed_time = time.time() - start_time   # Measure inference time
    all_times.append(elapsed_time)       # Store inference time

    # Use official Cityscapes colors for predicted mask visualization
    seg_rgb = colorize(pred_mask)

    # Ensure correct format before saving
    if seg_rgb.dtype != np.uint8:
        seg_rgb = (seg_rgb * 255).astype(np.uint8)
    else:
        seg_rgb = seg_rgb.astype(np.uint8)

    # Save the mask
    imageio.imwrite(f"{output_dir}/pred_mask.png", seg_rgb)

    normalized_entropy = (entropy_map - entropy_map.min()) / (entropy_map.max() - entropy_map.min() + 1e-8)
    plt.imsave(f"{output_dir}/entropy_map.png", normalized_entropy, cmap='hot')


    np.save(f"{output_dir}/pred_mask.npy", pred_mask)
    np.save(f"{output_dir}/entropy_values.npy", entropy_map)
    
    summary_json = {
        "filename": filename,
        "inference_time": elapsed_time,
        "entropy": {
            "min": float(entropy_map.min()),
            "max": float(entropy_map.max()),
            "mean": float(entropy_map.mean()),
            "high_entropy_ratio": float(high_entropy_ratio)
        },
        "miou": None if np.isnan(miou_val) else float(miou_val),
        "pixel_accuracy": None if np.isnan(acc_val) else float(acc_val),
        "predicted_classes": list(map(int, np.unique(pred_mask)))
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary_json, f, indent=2)
        
    with open(csv_path, "a") as f:
        f.write(f"{filename},{entropy_map.mean():.4f},{entropy_map.max():.4f},{entropy_map.min():.4f},{high_entropy_ratio:.4f},{elapsed_time:.4f}," +
            (f"{miou_val:.4f}" if not np.isnan(miou_val) else "NaN") + "," +
            (f"{acc_val:.4f}" if not np.isnan(acc_val) else "NaN") + "\n")

# === Compute and save mean mIoU and pixel accuracy ===
if all_mious:
    mean_miou = np.mean(all_mious)
    mean_acc = np.mean(all_accuracies)
    mean_time = np.mean(all_times)
    mean_fps = 1.0 / mean_time if mean_time > 0 else float('nan')
else:
    mean_miou = float('nan')
    mean_acc = float('nan')
    mean_time = float('nan')
    mean_fps = float('nan')


# Save to summary file
overall_path = os.path.join(output_root, "overall_metrics.csv")
with open(overall_path, "w") as f:
    f.write("Metric,Value\n")
    f.write(f"Mean mIoU,{mean_miou:.4f}\n")
    f.write(f"Mean Pixel Accuracy,{mean_acc:.4f}\n")
    f.write(f"Mean Inference Time per Image,{mean_time:.4f}\n")
    f.write(f"Mean FPS,{mean_fps:.2f}\n")


print(f"\n✅ Overall Mean mIoU: {mean_miou:.4f}")
print(f"✅ Overall Mean Pixel Accuracy: {mean_acc:.4f}")
print(f"🕒 Avg. Inference Time per Image: {mean_time:.4f} sec")
print(f"📈 Avg. Inference FPS: {mean_fps:.2f} frames/sec")
# === END OF THIs SCRIPT ===
