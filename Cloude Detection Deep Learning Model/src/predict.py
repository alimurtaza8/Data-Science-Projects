import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
import cv2
from glob import glob
from tqdm import tqdm
import time

def predict_clouds(model_path, image_dir, output_dir, img_size=(256, 256), batch_size=16):
    """
    Use the trained model to predict cloud masks for new satellite images
    
    Args:
        model_path: Path to the trained model
        image_dir: Directory containing input images
        output_dir: Directory to save prediction results
        img_size: Image dimensions (height, width)
        batch_size: Batch size for prediction
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    masks_dir = os.path.join(output_dir, 'predicted_masks')
    overlay_dir = os.path.join(output_dir, 'overlays')
    
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)
    
    # Load custom metrics
    custom_objects = {
        'dice_coefficient': dice_coefficient
    }
    
    # Load the trained model
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    print(f"Loaded model from {model_path}")
    
    # Get list of images
    image_paths = sorted(glob(os.path.join(image_dir, '*.jpg')))
    print(f"Found {len(image_paths)} images to process")
    
    # Process images in batches
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        
        # Load and preprocess images
        batch_images = []
        original_sizes = []
        
        for img_path in batch_paths:
            img = cv2.imread(img_path)
            orig_size = (img.shape[1], img.shape[0])  # (width, height)
            original_sizes.append(orig_size)
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img, img_size)
            img_normalized = img_resized / 255.0
            batch_images.append(img_normalized)
        
        # Convert to numpy array
        batch_images = np.array(batch_images)
        
        # Predict on batch
        start_time = time.time()
        predictions = model.predict(batch_images)
        end_time = time.time()
        
        # Save prediction results
        for j, img_path in enumerate(batch_paths):
            img_filename = os.path.basename(img_path)
            base_filename = os.path.splitext(img_filename)[0]
            
            # Original image (for overlay)
            original_img = cv2.imread(img_path)
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            
            # Get prediction mask
            pred_mask = predictions[j, :, :, 0]
            
            # Resize mask to original image size
            pred_mask_resized = cv2.resize(pred_mask, original_sizes[j])
            
            # Create binary mask (threshold at 0.5)
            binary_mask = (pred_mask_resized > 0.5).astype(np.uint8) * 255
            
            # Save mask as image
            cv2.imwrite(os.path.join(masks_dir, f"{base_filename}_mask.png"), binary_mask)
            
            # Create overlay (mask on original image)
            mask_rgb = np.zeros_like(original_img)
            mask_rgb[:, :, 0] = binary_mask  # Red channel
            overlay = cv2.addWeighted(original_img, 0.7, mask_rgb, 0.3, 0)
            
            # Save overlay
            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(overlay_dir, f"{base_filename}_overlay.jpg"), overlay_rgb)
        
        # Report progress
        batch_time = end_time - start_time
        print(f"Processed batch {i//batch_size + 1}/{(len(image_paths)+batch_size-1)//batch_size} "
              f"in {batch_time:.2f} seconds ({batch_time/len(batch_paths):.2f} seconds per image)")
    
    print(f"Prediction completed. Results saved to {output_dir}")

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """
    Dice coefficient for measuring segmentation quality
    """
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def prepare_test_images(output_dir='sample_data', num_images=5):
    """
    Create a few sample .jpg images for testing the model
    
    Args:
        output_dir: Directory to save sample images
        num_images: Number of sample images to create
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create simple sample images with cloud-like patterns
    for i in range(num_images):
        # Create blue background (sky)
        img = np.ones((512, 512, 3), dtype=np.uint8) * 255
        img[:, :, 0] = 150  # Blue component
        img[:, :, 1] = 200  # Green component
        img[:, :, 2] = 255  # Red component
        
        # Add some land features (brown-green)
        for _ in range(3):
            x1, y1 = np.random.randint(0, 512, 2)
            radius = np.random.randint(50, 150)
            color = (
                np.random.randint(100, 150),  # Blue
                np.random.randint(100, 200),  # Green
                np.random.randint(50, 100)    # Red
            )
            cv2.circle(img, (x1, y1), radius, color, -1)
        
        # Add some cloud-like features (white)
        for _ in range(5):
            x2, y2 = np.random.randint(0, 512, 2)
            width = np.random.randint(40, 120)
            height = np.random.randint(20, 60)
            angle = np.random.randint(0, 180)
            
            # Create white clouds
            cloud_color = (240, 240, 240)
            
            # Create elliptical cloud
            cv2.ellipse(img, (x2, y2), (width, height), angle, 0, 360, cloud_color, -1)
        
        # Save the image
        cv2.imwrite(os.path.join(output_dir, f'sample_image_{i+1}.jpg'), img)
    
    print(f"Created {num_images} sample images in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Make predictions with cloud detection model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--image_dir', type=str, required=True, help='Directory containing input images')
    parser.add_argument('--output_dir', type=str, default='./predictions', help='Directory to save results')
    parser.add_argument('--height', type=int, default=256, help='Image height')
    parser.add_argument('--width', type=int, default=256, help='Image width')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--create_samples', action='store_true', help='Create sample test images')
    
    args = parser.parse_args()
    
    if args.create_samples:
        prepare_test_images(output_dir=args.image_dir)
    
    predict_clouds(
        model_path=args.model_path,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        img_size=(args.height, args.width),
        batch_size=args.batch_size
    )