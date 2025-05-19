# Thesis
This repository presents a comprehensive pipeline for semantic segmentation and out-of-distribution (OOD) detection, specifically tailored for urban scene understanding in autonomous driving contexts. The work focuses on leveraging the strengths of the Cityscapes and Audi A2D2 datasets to evaluate model generalization and uncertainty estimation in cross-domain settings.

The project is organized into three main stages. In the first stage (Semantic_Segmentation_Cityscapes_Step 1.py), a semantic segmentation model based on DeepLabV3 with a ResNet101 backbone is trained on the Cityscapes dataset using fine annotations. The training process includes performance tracking with mIoU and pixel accuracy metrics, along with detailed visualizations and per-class analysis.

The second stage (Inference_a2d2_Step 2.py) adapts the trained Cityscapes model to the A2D2 dataset. This step performs inference on real A2D2 images, producing class predictions, pixel-wise entropy maps, and performance metrics such as mean Intersection-over-Union (mIoU) and pixel accuracy when ground truth labels are available. It also saves colorized segmentation outputs and entropy heatmaps to visualize model uncertainty.

The third stage (OOD_Detection_Step3.py) focuses on detecting regions with high uncertainty, which may indicate out-of-distribution content. Using entropy thresholds and optional class-aware suppression strategies, the script segments and highlights uncertain regions. Morphological operations help refine the detection of these regions, which are then extracted, annotated, and saved for further analysis. Summary statistics such as the number and total area of detected blobs are logged for each image.

Overall, this project demonstrates a full pipeline for training, inference, and uncertainty-based analysis in semantic segmentation models used in autonomous vehicles. It showcases cross-domain generalization from Cityscapes to A2D2 and proposes an efficient approach to identifying high-risk or unknown content in urban driving scenes.


