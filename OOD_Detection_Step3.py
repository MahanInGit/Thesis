import os
import numpy as np
from skimage import measure
from PIL import Image, ImageDraw
import csv
from tqdm import tqdm
from datetime import datetime
from scipy.ndimage import binary_opening, binary_closing
import matplotlib.pyplot as plt

# === CONFIGURATION ===
a2d2_image_folder = '/home/u509816/AV/datasets/a2d2_subset_test/front_center_images'
#a2d2_image_folder = '/Users/mahanabasiyan/Desktop/Thesis/my code/datasets/a2d2_subset_test/front_center_images'
output_root = '/home/u509816/AV/scripts/a2d2_outputs_results_4'
#output_root = '/Users/mahanabasiyan/Desktop/Thesis/my code/main codes'
entropy_threshold = 0.3  # Threshold for uncertainity
apply_class_aware_suppression = False    # Set to True to apply class-aware uncertainity
apply_class_aware_sky_veg = True      # Set to True to override the class-aware uncertainity
threshold_for_sky_veg = 0.9  # Threshold for sky and vegetation   
apply_morphological_smoothing = True  # Set to True to apply morphological smoothing
min_blob_area = 300
process_limit = None  # Set to an integer (e.g. 5) for testing, or None for full dataset
verbose = False  # Set to False to suppress print output

# Create timestamped CSV filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_csv_path = os.path.join(output_root, f"uncertain_blob_summary_{timestamp}.csv")

# Open CSV and write headers
log_file = open(log_csv_path, mode="w", newline="")
csv_writer = csv.writer(log_file)
csv_writer.writerow(["image_name", "num_blobs", "total_blob_area"])
# === Get folders ===
folders = sorted(os.listdir(output_root))
#folders = ["20181108141609_camera_frontcenter_000041751"]
if process_limit is not None:
    folders = folders[:process_limit]

for folder_name in tqdm(folders , desc="processing images"):
    folder_path = os.path.join(output_root, folder_name)
    if not os.path.isdir(folder_path):
        continue

    entropy_path = os.path.join(folder_path, "entropy_values.npy")
    image_path = os.path.join(a2d2_image_folder, f"{folder_name}.png")

    if not os.path.exists(entropy_path) or not os.path.exists(image_path):
        if verbose:
            print(f"⚠️ Skipping {folder_name}: missing entropy or image")
        continue

    # === Load data ===
    entropy_map = np.load(entropy_path)
    # Normalize the entropy map
    
    
    
    entropy_map = (entropy_map - np.min(entropy_map)) / (np.max(entropy_map) - np.min(entropy_map) + 1e-8)
    
    
    # Class_aware uncertainity for sky and vegetation
    if apply_class_aware_suppression:
        pred_mask_path = os.path.join(folder_path, "pred_mask.npy")
        if os.path.exists(pred_mask_path):
            pred_mask = np.load(pred_mask_path)
            ignore_classes = [10, 11]  # 10 = sky, 11 = vegetation
            ignore_mask = np.isin(pred_mask, ignore_classes)
            entropy_map[ignore_mask] *= 0.5  
            
    if apply_class_aware_sky_veg:
        pred_mask_path = os.path.join(folder_path, "pred_mask.npy")
        if os.path.exists(pred_mask_path):
            pred_mask = np.load(pred_mask_path)

            # Identify sky and vegetation pixels
            sky_veg_mask = np.isin(pred_mask, [10, 11])
            other_mask = ~sky_veg_mask

            # Initialize binary mask
            binary_mask = np.zeros_like(entropy_map, dtype=np.uint8)
            binary_mask[(entropy_map > entropy_threshold) & other_mask] = 1
            binary_mask[(entropy_map > threshold_for_sky_veg) & sky_veg_mask] = 1
        else:
            print(f"⚠️ pred_mask.npy not found in {folder_path}, skipping class-aware sky/veg logic.")
            binary_mask = (entropy_map > entropy_threshold).astype(np.uint8)
    else:
        binary_mask = (entropy_map > entropy_threshold).astype(np.uint8)
        
    plt.imsave(os.path.join(folder_path, "debug_binary_mask.png"), binary_mask, cmap='gray')

        
    if apply_morphological_smoothing:
        binary_mask = binary_closing(binary_mask, structure=np.ones((5, 5))).astype(np.uint8)
        binary_mask = binary_opening(binary_mask, structure=np.ones((5, 5))).astype(np.uint8)

    

    # === Find blobs ===
    labeled_mask = measure.label(binary_mask, connectivity=2)
    regions = measure.regionprops(labeled_mask)
    filtered_regions = [r for r in regions if r.area >= min_blob_area]
    
    total_area = sum(r.area for r in filtered_regions)
    csv_writer.writerow([folder_name, len(filtered_regions), total_area])

    # === Load original image and prepare for drawing/cropping ===
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    # === Create output folders ===
    save_folder = os.path.join(folder_path, "uncertain_crops")
    os.makedirs(save_folder, exist_ok=True)

    # === Save each crop ===
    for i, region in enumerate(filtered_regions):
        minr, minc, maxr, maxc = region.bbox
        cropped = image.crop((minc, minr, maxc, maxr))
        crop_path = os.path.join(save_folder, f"uncertain_{i:02d}.png")
        cropped.save(crop_path)

        draw.rectangle([minc, minr, maxc, maxr], outline="red", width=2)

    # === Save annotated preview ===
    preview_path = os.path.join(folder_path, "annotated_uncertain_regions.png")
    image.save(preview_path)

    if verbose:
        print(f"✅ {folder_name}: {len(filtered_regions)} uncertain region(s) saved.")
        
        
# Close the CSV file
log_file.close()
# === END OF THIs SCRIPT ===
