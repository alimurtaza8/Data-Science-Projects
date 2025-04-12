# src/utils.py
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_training_history(history, output_path):
    """
    Plot training history (loss and metrics)
    
    Args:
        history: Training history from model.fit()
        output_path: Path to save the plot
    """
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(history.history['accuracy'], label='Training Accuracy')
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    # Plot additional metrics if available
    if 'dice_coefficient' in history.history:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot Dice coefficient
        ax1.plot(history.history['dice_coefficient'], label='Training Dice')
        ax1.plot(history.history['val_dice_coefficient'], label='Validation Dice')
        ax1.set_title('Dice Coefficient')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Dice Coefficient')
        ax1.legend()
        ax1.grid(True)
        
        # Plot precision and recall
        ax2.plot(history.history['precision'], label='Training Precision')
        ax2.plot(history.history['val_precision'], label='Validation Precision')
        ax2.plot(history.history['recall'], label='Training Recall')
        ax2.plot(history.history['val_recall'], label='Validation Recall')
        ax2.set_title('Precision and Recall')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Score')
        ax2.legend()
        ax2.grid(True)
        
        # Save figure
        metrics_path = os.path.splitext(output_path)[0] + "_metrics.png"
        plt.tight_layout()
        plt.savefig(metrics_path)
        plt.close()

def visualize_predictions(model, images, true_masks, output_path, threshold=0.5):
    """
    Visualize model predictions compared to ground truth
    
    Args:
        model: Trained model
        images: Input images
        true_masks: Ground truth masks
        output_path: Path to save the visualization
        threshold: Threshold for binary prediction
    """
    # Make predictions
    pred_masks = model.predict(images)
    
    # Convert predictions to binary
    pred_masks_binary = (pred_masks > threshold).astype(np.float32)
    
    # Get number of samples
    n_samples = len(images)
    
    # Create figure
    fig, axes = plt.subplots(n_samples, 3, figsize=(15, 5 * n_samples))
    
    # Handle single sample case
    if n_samples == 1:
        axes = np.expand_dims(axes, axis=0)
    
    # Plot each sample
    for i in range(n_samples):
        # Original image
        axes[i, 0].imshow(images[i])
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')
        
        # Ground truth mask
        axes[i, 1].imshow(true_masks[i].squeeze(), cmap='gray')
        axes[i, 1].set_title('Ground Truth Mask')
        axes[i, 1].axis('off')
        
        # Predicted mask
        axes[i, 2].imshow(pred_masks_binary[i].squeeze(), cmap='gray')
        axes[i, 2].set_title('Predicted Mask')
        axes[i, 2].axis('off')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def create_sample_images(img_paths, mask_paths, output_path, num_samples=5):
    """
    Create a visualization of sample images and their corresponding masks
    
    Args:
        img_paths: List of image paths
        mask_paths: List of mask paths
        output_path: Path to save the visualization
        num_samples: Number of samples to visualize
    """
    # Get random indices
    indices = np.random.choice(len(img_paths), min(num_samples, len(img_paths)), replace=False)
    
    # Create figure
    fig, axes = plt.subplots(len(indices), 2, figsize=(10, 5 * len(indices)))
    
    # Handle single sample case
    if len(indices) == 1:
        axes = np.expand_dims(axes, axis=0)
    
    # Plot each sample
    for i, idx in enumerate(indices):
        # Load image
        img = plt.imread(img_paths[idx])
        
        # Load mask
        mask = plt.imread(mask_paths[idx])
        
        # Plot image
        axes[i, 0].imshow(img)
        axes[i, 0].set_title('Image')
        axes[i, 0].axis('off')
        
        # Plot mask
        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 1].set_title('Cloud Mask')
        axes[i, 1].axis('off')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()