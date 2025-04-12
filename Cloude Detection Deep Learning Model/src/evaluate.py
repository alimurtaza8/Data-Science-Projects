import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc
import seaborn as sns
import argparse
import cv2
from tqdm import tqdm
from data_preparation import CloudDataLoader

def evaluate_model(model_path, data_dir, output_dir, img_size=(256, 256), batch_size=16):
    """
    Evaluate the trained model on the test dataset
    
    Args:
        model_path: Path to the trained model
        data_dir: Directory containing the dataset
        output_dir: Directory to save evaluation results
        img_size: Image dimensions (height, width)
        batch_size: Batch size for evaluation
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load custom metrics
    custom_objects = {
        'dice_coefficient': dice_coefficient
    }
    
    # Load the trained model
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    print(f"Loaded model from {model_path}")
    
    # Set up data loader for test set
    data_loader = CloudDataLoader(data_dir, img_size=img_size, batch_size=batch_size)
    
    # Get test data paths
    test_img_dir = os.path.join(data_dir, 'test_red_blue_green')
    test_mask_dir = os.path.join(data_dir, 'test_ground_truth')
    
    test_img_paths = sorted([os.path.join(test_img_dir, f) for f in os.listdir(test_img_dir) if f.endswith('.jpg')])
    test_mask_paths = []
    
    # Match each image with its corresponding mask
    for img_path in test_img_paths:
        img_filename = os.path.basename(img_path)
        mask_filename = img_filename.replace('.jpg', '_mask.png')
        mask_path = os.path.join(test_mask_dir, mask_filename)
        
        if os.path.exists(mask_path):
            test_mask_paths.append(mask_path)
        else:
            print(f"Warning: Mask not found for {img_filename}")
            test_img_paths.remove(img_path)
    
    print(f"Found {len(test_img_paths)} test images with matching masks")
    
    # Process test images and get predictions
    all_predictions = []
    all_ground_truth = []
    
    for i, (img_path, mask_path) in enumerate(tqdm(zip(test_img_paths, test_mask_paths), total=len(test_img_paths))):
        # Load and preprocess image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, img_size)
        img = img / 255.0  # Normalize to [0, 1]
        
        # Load ground truth mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, img_size)
        mask = (mask > 127).astype(np.float32)
        
        # Make prediction
        pred = model.predict(np.expand_dims(img, axis=0))[0, :, :, 0]
        
        all_predictions.append(pred)
        all_ground_truth.append(mask)
        
        # Save a few visualization examples
        if i < 5:  # Save first 5 examples
            visualize_prediction(img, mask, pred, os.path.join(output_dir, f'example_{i+1}.png'))
    
    # Convert lists to numpy arrays
    all_predictions = np.array(all_predictions)
    all_ground_truth = np.array(all_ground_truth)
    
    # Calculate metrics
    calculate_and_save_metrics(all_predictions, all_ground_truth, output_dir)
    
    print(f"Evaluation completed. Results saved to {output_dir}")

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """
    Dice coefficient for measuring segmentation quality
    """
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def visualize_prediction(image, ground_truth, prediction, output_path):
    """
    Visualize and save prediction results
    
    Args:
        image: Input image
        ground_truth: Ground truth mask
        prediction: Predicted mask
        output_path: Path to save the visualization
    """
    plt.figure(figsize=(15, 5))
    
    # Original image
    plt.subplot(1, 4, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    # Ground truth
    plt.subplot(1, 4, 2)
    plt.imshow(ground_truth, cmap='gray')
    plt.title('Ground Truth')
    plt.axis('off')
    
    # Prediction
    plt.subplot(1, 4, 3)
    plt.imshow(prediction, cmap='gray')
    plt.title('Prediction')
    plt.axis('off')
    
    # Overlay prediction on image
    overlay = image.copy()
    prediction_binary = (prediction > 0.5).astype(np.uint8)
    mask_rgb = np.zeros_like(overlay)
    mask_rgb[:, :, 0] = prediction_binary * 255  # Red channel
    overlay = cv2.addWeighted(overlay, 0.7, mask_rgb, 0.3, 0)
    
    plt.subplot(1, 4, 4)
    plt.imshow(overlay)
    plt.title('Overlay')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def calculate_and_save_metrics(predictions, ground_truth, output_dir):
    """
    Calculate and save evaluation metrics
    
    Args:
        predictions: Model predictions
        ground_truth: Ground truth masks
        output_dir: Directory to save results
    """
    # Flatten arrays for metric calculation
    pred_flat = predictions.flatten()
    gt_flat = ground_truth.flatten()
    
    # Calculate binary metrics using threshold of 0.5
    pred_binary = (pred_flat > 0.5).astype(np.uint8)
    
    # Confusion matrix
    cm = confusion_matrix(gt_flat, pred_binary)
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate Dice coefficient
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    
    # Calculate IoU (Intersection over Union)
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    
    # Save metrics to file
    with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1_score:.4f}\n")
        f.write(f"Dice Coefficient: {dice:.4f}\n")
        f.write(f"IoU: {iou:.4f}\n")
        f.write(f"\nConfusion Matrix:\n")
        f.write(f"TN: {tn}, FP: {fp}\n")
        f.write(f"FN: {fn}, TP: {tp}\n")
    
    # Plot and save confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Cloud', 'Cloud'],
                yticklabels=['No Cloud', 'Cloud'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    
    # Precision-Recall curve
    precision_values, recall_values, thresholds = precision_recall_curve(gt_flat, pred_flat)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall_values, precision_values, lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'))
    plt.close()
    
    # ROC curve
    fpr, tpr, _ = roc_curve(gt_flat, pred_flat)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    plt.close()
    
    # Save threshold analysis
    analyze_thresholds(gt_flat, pred_flat, output_dir)

def analyze_thresholds(gt_flat, pred_flat, output_dir):
    """
    Analyze model performance at different thresholds
    
    Args:
        gt_flat: Flattened ground truth masks
        pred_flat: Flattened prediction probabilities
        output_dir: Directory to save results
    """
    thresholds = np.arange(0.1, 1.0, 0.1)
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    dice_scores = []
    ious = []
    
    for threshold in thresholds:
        pred_binary = (pred_flat > threshold).astype(np.uint8)
        
        # Confusion matrix
        cm = confusion_matrix(gt_flat, pred_binary)
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        dice_scores.append(dice)
        ious.append(iou)
    
    # Plot metrics vs threshold
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, accuracies, '-o', label='Accuracy')
    plt.plot(thresholds, precisions, '-o', label='Precision')
    plt.plot(thresholds, recalls, '-o', label='Recall')
    plt.plot(thresholds, f1_scores, '-o', label='F1 Score')
    plt.plot(thresholds, dice_scores, '-o', label='Dice')
    plt.plot(thresholds, ious, '-o', label='IoU')
    
    plt.xlabel('Threshold')
    plt.ylabel('Metric Value')
    plt.title('Metrics vs. Threshold')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'threshold_analysis.png'))
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate cloud detection model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='./evaluation', help='Directory to save results')
    parser.add_argument('--height', type=int, default=256, help='Image height')
    parser.add_argument('--width', type=int, default=256, help='Image width')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    
    args = parser.parse_args()
    
    evaluate_model(
        model_path=args.model_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        img_size=(args.height, args.width),
        batch_size=args.batch_size
    )