import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from tqdm import tqdm         # To show a progress bar during training
import torch
import torch.nn as nn           # pytorch's neural network module for building models and loss functions
import torch.optim as optim             # Pytorch's optimizer module for optimizing the model   
from torch.utils.data import DataLoader             # Dataloader and dataset tools for loading and processing data
from torchvision import models, transforms, datasets        
import os           # To handle paths, create directories, and manipulate file paths
from torch.optim.lr_scheduler import StepLR         # For updating learning rate
from sklearn.metrics import confusion_matrix     # For calculating confusion matrix
import random    # To set random seed for reproducibility
import csv
from cityscapesscripts.helpers.labels import labels
from cityscapesscripts.helpers.labels import trainId2label
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score

# setting random seed for reproducibility
seed = 42

# Make results reproducible across runs
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

colormap = np.zeros((256, 3), dtype=np.uint8)
for train_id, label in trainId2label.items():
    if train_id != 255:  # skip 'ignore' class
        colormap[train_id] = label.color

def colorize(mask):
    """Convert a trainId mask to a color RGB image."""
    return colormap[mask]


# Mapping Cityscapes label IDs to train IDs
label_id_to_train_id = {
    label.id: label.trainId
    for label in labels
    if label.trainId != 255
}

# Function to convert full label IDs to train IDs
def convert_ids(mask):
    mask = np.array(mask).astype(np.int32)  # ensure safe int comparison
    remapped = 255 * np.ones_like(mask, dtype=np.int32)  # safe type
    for label_id, train_id in label_id_to_train_id.items():
        remapped[mask == label_id] = train_id
    remapped = remapped.astype(np.uint8)  # convert after remapping
    return torch.from_numpy(remapped).long()



# Setting the device to GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to Cityscapes dataset
cityscapes_train_path = "/home/u509816/AV/datasets/cityscapes"

# Saving a saample of one image and its ground truth mask and the predicted mask after a few epochs
visualize_every = 3
sample_index = 0     # Always pick the first sample in the batch

# Function to compute mean Intersection over Union (mIoU)
def compute_miou(preds, labels, num_classes=19):
    """
    Calculates mIoU for semantic segmentation predictions.
    preds: Flattened predicted class indices
    labels: Flattened ground truth class indices
    """
    # Compute confusion matrix
    cm = confusion_matrix(labels.ravel(), preds.ravel(), labels=list(range(num_classes)))

    # Calculate intersection and union for each class
    intersection = np.diag(cm)
    union = np.sum(cm, axis=0) + np.sum(cm, axis=1) - intersection

    # Avoid division by zero
    iou = intersection / np.maximum(union, 1)

    # Average over all classes
    miou = np.mean(iou)

    return miou

def compute_per_class_iou(preds, labels, num_classes=19):
    cm = confusion_matrix(labels.ravel(), preds.ravel(), labels=list(range(num_classes)))
    intersection = np.diag(cm)
    union = np.sum(cm, axis=0) + np.sum(cm, axis=1) - intersection
    iou = intersection / np.maximum(union, 1)
    return iou

def compute_per_class_accuracy(preds, labels, num_classes=19):
    accs = []
    for cls in range(num_classes):
        mask = labels == cls
        correct = (preds[mask] == cls).sum()
        total = mask.sum()
        acc = correct / total if total > 0 else 0
        accs.append(acc)
    return accs



def pixel_accuracy(preds, labels):
    correct = (preds == labels).sum()
    total = np.prod(labels.shape)
    return correct / total


# Defining a set of transformations to be applied to the images of Cityscapes dataset
transform = transforms.Compose([
    transforms.Resize((512, 1024)),  # Resize images to 512x1024
    transforms.ToTensor(),  # Convert images from to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
])

# Defining transformation for ground truth segmentation masks
transform_mask = transforms.Compose([
    transforms.Resize((512, 1024) , interpolation=transforms.InterpolationMode.NEAREST),# Resize masks to the same size as images
    transforms.Lambda(convert_ids),     # Convert full label IDs to train IDs
])

# Loading Cityscapes' training dataset
cityscapes_train_dataset = datasets.Cityscapes(
    root=cityscapes_train_path,
    split='train',
    mode='fine',           # using fine annotations instead of coarse for better accuracy
    target_type='semantic',
    transform=transform,
    target_transform=transform_mask
)

# Loading Cityscapes' validation dataset
cityscapes_val_dataset = datasets.Cityscapes(
    root=cityscapes_train_path,
    split='val',
    mode='fine',      # using fine annotations instead of coarse for better accuracy
    target_type='semantic',
    transform=transform,
    target_transform=transform_mask
)

# Creating data loaders for training and validation datasets
train_loader = DataLoader(
    cityscapes_train_dataset,
    batch_size=4,  # Number of images per batch
    shuffle=True,  # Shuffle the dataset for better training
    num_workers=8,  # Number of subprocesses to use for data loading
    pin_memory=True,  # Load data into memory for faster access
    drop_last=True,  # Drop the last incomplete batch
    persistent_workers=True,  # Keep workers alive for faster data loading
    prefetch_factor=4     # Number of batches to prefetch

)

val_loader = DataLoader(
    cityscapes_val_dataset,
    batch_size=4,          # Same batch size for validation
    shuffle=False,         # Do not shuffle validation data because we waant the validation results to be consistent
    num_workers=8,         # Same number of workers
    pin_memory=True,       # Load data into memory for faster access
    drop_last=False,       # Because we want all validation data
    persistent_workers=True,  # Keep workers alive for faster data loading
    prefetch_factor=4     # Number of batches to prefetch
)

# Defining the semantic segmentation model

weights = DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1
model = deeplabv3_resnet101(weights=weights)
model.classifier = nn.Sequential(
    nn.Conv2d(2048, 256, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Dropout2d(p=0.2),  # Added dropout
    nn.Conv2d(256, 19, kernel_size=1)
)


model = model.to(device)  # Move the model to the device (GPU or CPU)

# Log model parameter count
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"📊 Total Parameters: {total_params:,}")
print(f"🧠 Trainable Parameters: {trainable_params:,}")


# Defining the loss function    
# Label smoothing is used to prevent the model from becoming too confident in its predictions
criterion = nn.CrossEntropyLoss(ignore_index=255 , label_smoothing= 0.1)  # Cross-entropy loss for multi-class segmentation

# Defining the optimizer
optimizer = optim.Adam(
    model.parameters(),  # The parameters we want to optimize (all model weights)
    lr= 0.0001      # Learning rate (small value for stable training)
)

# Defining the learning rate scheduler
scheduler = StepLR(
    optimizer,  # The optimizer we want to schedule
    step_size=10,  # Number of epochs after which to decrease the learning rate
    gamma= 0.5        # Multiply learning rate by 0.5 (reduce it by half)
)

# Making a file to log the training progress
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # Get current time for unique log file name
log_file_name = f"training_log_{current_time}.csv"
log_file = open(log_file_name, "w", newline='')
csv_writer = csv.writer(log_file)
csv_writer.writerow([f"# Seed used: {seed}"])  # Logs the seed value
csv_writer.writerow(["Epoch", "Train Loss", "Val Loss", "mIoU", "Pixel Accuracy", "Learning Rate"])



# Set the number of epochs
num_epochs = 30

resume_training = True  # Set to True if you want to resume from checkpoint

start_epoch = 0  # Default if not resuming
best_val_loss = float('inf')  # Default best val loss
# Number of epochs to wait for improvement before stopping
patience = 10

# Early stopping counter
epochs_without_improvement = 0

cityscapes_classes = [
                "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
                "traffic sign", "vegetation", "terrain", "sky", "person", "rider",
                "car", "truck", "bus", "train", "motorcycle", "bicycle"
            ]

# Check if checkpoint exists and resume
if resume_training and os.path.exists('best_checkpoint.pth'):
    checkpoint = torch.load('best_checkpoint.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint['val_loss']
    
    print(f"🔄 Resumed training from epoch {start_epoch} with best val loss {best_val_loss:.4f}")
    csv_writer.writerow([f"# Resumed from checkpoint at epoch {start_epoch}"])
    
    
# Start the training process
for epoch in range(start_epoch , num_epochs):
    
    # --- Training Phase ---
    model.train()  # Set model to training mode (activates Dropout, BatchNorm updating)
    running_loss = 0.0  # Initialize cumulative training loss
    
    for images, masks in tqdm(train_loader , desc= f"Training Epoch {epoch+1}"):  # Loop through all batches
        images = images.to(device)  # Move images to device (GPU or CPU)
        masks = masks.squeeze(1).long().to(device)  # Move masks to device and format them
        
        optimizer.zero_grad()  # Clear old gradients
        
        outputs = model(images)['out']  # Forward pass: get predictions
        loss = criterion(outputs, masks)  # Compute loss
        
        loss.backward()  # Backward pass: compute gradients
        optimizer.step()  # Optimizer step: update model parameters
        
        running_loss += loss.item()  # Accumulate batch loss
    
    avg_train_loss = running_loss / len(train_loader)  # Calculate average training loss for this epoch

    # --- Validation Phase ---
    model.eval()  # Set model to evaluation mode (deactivates Dropout, BatchNorm updating)
    val_loss = 0.0  # Initialize cumulative validation loss
    
    # Store predictions and labels for mIoU calculation across batches
    all_preds = []
    all_labels = []

    with torch.no_grad():  # No need to compute gradients during validation
        for images, masks in tqdm(val_loader, desc= f"Validating Epoch {epoch+1}"):  # Loop through all validation batches
            images = images.to(device)
            masks = masks.squeeze(1).long().to(device)

            outputs = model(images)['out']  # Forward pass only
            
            loss = criterion(outputs, masks)  # Compute validation loss
            val_loss += loss.item()  # Accumulate batch loss
            
            preds = torch.argmax(outputs, dim=1)       # Converting raw outputs to predicted class indices
            all_preds.append(preds.cpu().numpy())
            all_labels.append(masks.cpu().numpy())
        # Concatenate all predictions and labels into flat arrays
        all_preds_np = np.concatenate(all_preds)
        all_labels_np = np.concatenate(all_labels)
        
        # Calculate mIoU
        miou = compute_miou(all_preds_np, all_labels_np)
        acc = pixel_accuracy(all_preds_np, all_labels_np)
        
        if epoch + 1 == num_epochs:
            
            per_class_iou = compute_per_class_iou(all_preds_np, all_labels_np)
            for i, val in enumerate(per_class_iou):
                print(f"Class {i} IoU: {val:.4f}")
            with open("per_class_iou.csv", "w") as f:
                f.write("Class,IoU\n")
                for i, val in enumerate(per_class_iou):
                    f.write(f"{cityscapes_classes[i]},{val:.4f}\n")
            plt.figure(figsize=(12, 5))
            plt.bar(cityscapes_classes, per_class_iou)
            plt.xticks(rotation=45, ha='right')
            plt.ylabel("IoU")
            plt.title("Per-Class IoU")
            plt.tight_layout()
            plt.savefig("iou_bar_chart.png")
            plt.close()

            
                
                # Compute class-wise accuracy
            per_class_acc = compute_per_class_accuracy(all_preds_np, all_labels_np)
            for i, acc_val in enumerate(per_class_acc):
                print(f"Class {i} Accuracy: {acc_val:.4f}")
                
            # Save class-wise accuracy to CSV
            with open("per_class_accuracy.csv", "w") as f:
                f.write("Class,Accuracy\n")
                for i, acc_val in enumerate(per_class_acc):
                    f.write(f"{cityscapes_classes[i]},{acc_val:.4f}\n")

            # Compute precision, recall, F1
            precision = precision_score(all_labels_np.ravel(), all_preds_np.ravel(), average=None, labels=list(range(19)))
            recall = recall_score(all_labels_np.ravel(), all_preds_np.ravel(), average=None, labels=list(range(19)))
            f1 = f1_score(all_labels_np.ravel(), all_preds_np.ravel(), average=None, labels=list(range(19)))

            # Save to CSV
            with open("per_class_prf.csv", "w") as f:
                f.write("Class,Precision,Recall,F1\n")
                for i in range(19):
                    f.write(f"{cityscapes_classes[i]},{precision[i]:.4f},{recall[i]:.4f},{f1[i]:.4f}\n")
                            
            
           
            cm = confusion_matrix(all_labels_np.ravel(), all_preds_np.ravel(), labels=list(range(19)))
            plt.figure(figsize=(12, 10))
            sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=cityscapes_classes, yticklabels=cityscapes_classes)
            plt.title("Confusion Matrix (Validation)")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.savefig("confusion_matrix.png")
            plt.close()
            # === Save final summary CSV ===
            with open("final_summary.csv", "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Metric", "Value"])
                writer.writerow(["Best Val Loss", f"{best_val_loss:.4f}"])
                writer.writerow(["Final mIoU", f"{miou:.4f}"])
                writer.writerow(["Final Pixel Accuracy", f"{acc:.4f}"])
            
            # Compute overall (macro-averaged) precision/recall/F1
            macro_precision = precision_score(all_labels_np.ravel(), all_preds_np.ravel(), average='macro')
            macro_recall = recall_score(all_labels_np.ravel(), all_preds_np.ravel(), average='macro')
            macro_f1 = f1_score(all_labels_np.ravel(), all_preds_np.ravel(), average='macro')

            # Append to final_summary.csv
            with open("final_summary.csv", "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Macro Precision", f"{macro_precision:.4f}"])
                writer.writerow(["Macro Recall", f"{macro_recall:.4f}"])
                writer.writerow(["Macro F1 Score", f"{macro_f1:.4f}"])

                
            with open("model_summary.csv", "w") as f:
                writer = csv.writer(f)
                writer.writerow(["Model", "Backbone", "Params", "Trainable", "Final mIoU", "Pixel Acc"])
                writer.writerow(["DeepLabV3", "ResNet101", total_params, trainable_params, f"{miou:.4f}", f"{acc:.4f}"])
        
        # --- Visualization: Save side-by-side image every few epochs ---
        if (epoch + 1) % visualize_every == 0:
            # Get the first sample from the last batch
            input_img = images[sample_index].cpu()
            gt_mask = masks[sample_index].cpu().numpy()
            pred_mask = preds[sample_index].cpu().numpy()

            # Unnormalize input image
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            input_img = input_img.permute(1, 2, 0).numpy()
            input_img = (input_img * std + mean).clip(0, 1)

            # Create side-by-side plot
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))

            axs[0].imshow(input_img)
            axs[0].set_title("Input Image")
            axs[0].axis("off")

            axs[1].imshow(colorize(gt_mask))
            axs[1].set_title("Ground Truth")
            axs[1].axis("off")

            axs[2].imshow(colorize(pred_mask))
            axs[2].set_title("Prediction")
            axs[2].axis("off")

            # Save image
            vis_dir = "visualizations"
            os.makedirs(vis_dir, exist_ok=True)
            save_path = os.path.join(vis_dir, f"epoch_{epoch+1}.png")
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()

            print(f"🖼️ Saved visualization to {save_path}")

    
    avg_val_loss = val_loss / len(val_loader)  # Calculate average validation loss for this epoch

    # Print the progress of training and validation
    print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | mIoU: {miou:.4f} | Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
    
    # Log the training and validation loss to the file
    csv_writer.writerow([epoch+1, round(avg_train_loss, 4), round(avg_val_loss, 4), round(miou, 4), round(acc, 4), scheduler.get_last_lr()[0]])


    # Save the model if validation loss improved
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        # Save full checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': best_val_loss
        }
        torch.save(checkpoint, 'best_checkpoint.pth')
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"✅ Checkpoint saved at epoch {epoch+1} with val loss {best_val_loss:.4f}")

        epochs_without_improvement = 0  # Reset the counter if we have an improvement
    else:
        epochs_without_improvement += 1   # Increment the counter if no improvement
        print(f"No improvement for {epochs_without_improvement} epochs")
    
    # Check for early stopping
    if epochs_without_improvement >= patience:
        print(f"Early stopping at epoch {epoch+1} due to no improvement")
        break
    
    # Step the learning rate scheduler
    scheduler.step()        # Update the learning rate based on the scheduler
    print(f"Learning rate updated to {scheduler.get_last_lr()[0]:.6f}")
log_file.close()  # Close the log file after training

# === Plot training curves after training ===
import pandas as pd
import matplotlib.pyplot as plt

# Read the training log CSV (ignoring comments)
df = pd.read_csv(log_file_name, comment='#')

# Plot Train vs Val Loss
plt.figure(figsize=(8, 5))
plt.plot(df['Epoch'], df['Train Loss'], label='Train Loss')
plt.plot(df['Epoch'], df['Val Loss'], label='Val Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_curve.png")
plt.close()

# Plot mIoU over epochs
plt.figure(figsize=(8, 5))
plt.plot(df['Epoch'], df['mIoU'], color='green')
plt.xlabel("Epoch")
plt.ylabel("Mean IoU")
plt.title("mIoU over Epochs")
plt.grid(True)
plt.tight_layout()
plt.savefig("miou_curve.png")
plt.close()

# Plot Pixel Accuracy over epochs
plt.figure(figsize=(8, 5))
plt.plot(df['Epoch'], df['Pixel Accuracy'], color='orange')
plt.xlabel("Epoch")
plt.ylabel("Pixel Accuracy")
plt.title("Pixel Accuracy over Epochs")
plt.grid(True)
plt.tight_layout()
plt.savefig("pixel_accuracy_curve.png")
plt.close()

print("📊 Training curves saved as: loss_curve.png, miou_curve.png, pixel_accuracy_curve.png")
# === END OF THIs SCRIPT ===
    