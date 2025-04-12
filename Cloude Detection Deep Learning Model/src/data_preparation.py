import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2
from tqdm import tqdm

class CloudDataLoader:
    """
    Data loader for the 38-Cloud dataset
    """
    def __init__(self, data_dir, img_size=(256, 256), batch_size=16, val_split=0.2):
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.val_split = val_split
        
    def prepare_data(self):
        """
        Prepare and split the dataset into training and validation sets
        """
        print("Preparing 38-Cloud dataset...")
        
        # Define paths for train data
        train_img_dir = os.path.join(self.data_dir, 'train_red_blue_green')
        train_mask_dir = os.path.join(self.data_dir, 'train_ground_truth')
        
        # Get all training image and mask file paths
        train_img_paths = sorted(glob.glob(os.path.join(train_img_dir, '*.jpg')))
        train_mask_paths = []
        
        # Match each image with its corresponding mask
        for img_path in train_img_paths:
            img_filename = os.path.basename(img_path)
            # In 38-Cloud, filenames match but masks have "_mask" suffix
            mask_filename = img_filename.replace('.jpg', '_mask.png')
            mask_path = os.path.join(train_mask_dir, mask_filename)
            
            if os.path.exists(mask_path):
                train_mask_paths.append(mask_path)
            else:
                print(f"Warning: Mask not found for {img_filename}")
                train_img_paths.remove(img_path)
        
        print(f"Found {len(train_img_paths)} train images with matching masks")
        
        # Split data into train and validation sets
        train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = train_test_split(
            train_img_paths, train_mask_paths, test_size=self.val_split, random_state=42
        )
        
        print(f"Split into {len(train_img_paths)} training and {len(val_img_paths)} validation samples")
        
        return (train_img_paths, train_mask_paths), (val_img_paths, val_mask_paths)
    
    def data_generator(self, img_paths, mask_paths, augment=False):
        """
        Generator that yields batches of images and masks
        
        Args:
            img_paths: List of image file paths
            mask_paths: List of mask file paths
            augment: Whether to apply data augmentation
            
        Yields:
            tuple: (batch of images, batch of masks)
        """
        num_samples = len(img_paths)
        indices = np.arange(num_samples)
        
        while True:
            np.random.shuffle(indices)
            
            for start_idx in range(0, num_samples, self.batch_size):
                batch_indices = indices[start_idx:start_idx + self.batch_size]
                
                batch_img_paths = [img_paths[i] for i in batch_indices]
                batch_mask_paths = [mask_paths[i] for i in batch_indices]
                
                batch_imgs = []
                batch_masks = []
                
                for img_path, mask_path in zip(batch_img_paths, batch_mask_paths):
                    # Load and preprocess image
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, self.img_size)
                    img = img / 255.0  # Normalize to [0, 1]
                    
                    # Load and preprocess mask (binary: cloud=1, non-cloud=0)
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    mask = cv2.resize(mask, self.img_size)
                    mask = (mask > 127).astype(np.float32)  # Threshold to binary
                    mask = np.expand_dims(mask, axis=-1)  # Add channel dimension
                    
                    if augment:
                        img, mask = self._apply_augmentation(img, mask)
                    
                    batch_imgs.append(img)
                    batch_masks.append(mask)
                
                yield np.array(batch_imgs), np.array(batch_masks)
    
    def _apply_augmentation(self, img, mask):
        """
        Apply data augmentation to image and mask
        
        Args:
            img: Input image
            mask: Input mask
            
        Returns:
            tuple: (augmented image, augmented mask)
        """
        # Horizontal flip with 50% probability
        if np.random.random() > 0.5:
            img = np.fliplr(img)
            mask = np.fliplr(mask)
        
        # Vertical flip with 50% probability
        if np.random.random() > 0.5:
            img = np.flipud(img)
            mask = np.flipud(mask)
        
        # Random rotation by 0, 90, 180, or 270 degrees
        k = np.random.randint(0, 4)  # Number of 90-degree rotations
        if k > 0:
            img = np.rot90(img, k)
            mask = np.rot90(mask, k)
        
        # Random brightness adjustment
        brightness_factor = np.random.uniform(0.8, 1.2)
        img = np.clip(img * brightness_factor, 0, 1)
        
        return img, mask
    
    def get_train_val_generators(self):
        """
        Get train and validation data generators
        
        Returns:
            tuple: (train_generator, val_generator, train_steps, val_steps)
        """
        (train_img_paths, train_mask_paths), (val_img_paths, val_mask_paths) = self.prepare_data()
        
        train_generator = self.data_generator(train_img_paths, train_mask_paths, augment=True)
        val_generator = self.data_generator(val_img_paths, val_mask_paths, augment=False)
        
        train_steps = len(train_img_paths) // self.batch_size
        val_steps = len(val_img_paths) // self.batch_size
        
        if len(train_img_paths) % self.batch_size != 0:
            train_steps += 1
        if len(val_img_paths) % self.batch_size != 0:
            val_steps += 1
        
        return train_generator, val_generator, train_steps, val_steps

    def visualize_samples(self, num_samples=3):
        """
        Visualize a few samples from the dataset
        """
        (train_img_paths, train_mask_paths), _ = self.prepare_data()
        
        # Get random indices
        indices = np.random.choice(len(train_img_paths), num_samples, replace=False)
        
        plt.figure(figsize=(12, 4*num_samples))
        
        for i, idx in enumerate(indices):
            # Load and display image
            img = cv2.imread(train_img_paths[idx])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.img_size)
            
            # Load and display mask
            mask = cv2.imread(train_mask_paths[idx], cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, self.img_size)
            mask = (mask > 127).astype(np.uint8) * 255  # Threshold to binary
            
            # Plot image and mask
            plt.subplot(num_samples, 3, i*3 + 1)
            plt.imshow(img)
            plt.title(f"Image {i+1}")
            plt.axis('off')
            
            plt.subplot(num_samples, 3, i*3 + 2)
            plt.imshow(mask, cmap='gray')
            plt.title(f"Mask {i+1}")
            plt.axis('off')
            
            # Overlay mask on image
            overlay = img.copy()
            mask_rgb = np.zeros_like(img)
            mask_rgb[:, :, 0] = mask  # Red channel
            overlay = cv2.addWeighted(overlay, 0.7, mask_rgb, 0.3, 0)
            
            plt.subplot(num_samples, 3, i*3 + 3)
            plt.imshow(overlay)
            plt.title(f"Overlay {i+1}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()