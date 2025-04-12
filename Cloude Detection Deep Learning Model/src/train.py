import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
import matplotlib.pyplot as plt
import numpy as np
import datetime
import argparse
from model import create_rsnet
from data_preparation import CloudDataLoader

def train_model(data_dir, output_dir, epochs=100, batch_size=16, img_size=(256, 256), 
                input_channels=3, patience=10, initial_lr=1e-4, pretrained_model=None):
    """
    Train the RS-Net model on cloud detection dataset
    
    Args:
        data_dir: Directory containing the dataset
        output_dir: Directory to save model checkpoints and logs
        epochs: Number of training epochs
        batch_size: Batch size for training
        img_size: Image dimensions (height, width)
        input_channels: Number of input channels (3 for RGB, more for multi-spectral)
        patience: Patience for early stopping
        initial_lr: Initial learning rate
        pretrained_model: Path to pretrained model (if any)
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    model_dir = os.path.join(output_dir, 'models')
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Set up data loader
    data_loader = CloudDataLoader(data_dir, img_size=img_size, batch_size=batch_size)
    
    # Get data generators
    train_gen, val_gen, train_steps, val_steps = data_loader.get_train_val_generators()
    
    # Create or load model
    if pretrained_model and os.path.exists(pretrained_model):
        print(f"Loading pretrained model from {pretrained_model}")
        model = tf.keras.models.load_model(
            pretrained_model, 
            custom_objects={
                'dice_coefficient': dice_coefficient
            }
        )
    else:
        print("Creating new RS-Net model")
        model = create_rsnet(input_channels=input_channels, input_size=img_size)
    
    # Display model summary
    model.summary()
    
    # Set up callbacks
    callbacks = [
        ModelCheckpoint(
            os.path.join(model_dir, 'rsnet_cloud_best.h5'),
            monitor='val_dice_coefficient',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_dice_coefficient',
            patience=patience,
            mode='max',
            verbose=1,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_dice_coefficient',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            mode='max',
            verbose=1
        ),
        TensorBoard(
            log_dir=os.path.join(log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")),
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        )
    ]
    
    # Train the model
    print(f"Starting training for {epochs} epochs...")
    history = model.fit(
        train_gen,
        steps_per_epoch=train_steps,
        epochs=epochs,
        validation_data=val_gen,
        validation_steps=val_steps,
        callbacks=callbacks
    )
    
    # Save the final model
    model.save(os.path.join(model_dir, 'rsnet_cloud_final.h5'))
    
    # Plot training history
    plot_training_history(history, output_dir)
    
    print(f"Training completed. Model saved to {model_dir}")
    return model, history

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """
    Dice coefficient for measuring segmentation quality
    """
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def plot_training_history(history, output_dir):
    """
    Plot and save training history
    """
    plt.figure(figsize=(12, 5))
    
    # Plot training & validation loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    # Plot dice coefficient
    plt.subplot(1, 2, 2)
    plt.plot(history.history['dice_coefficient'])
    plt.plot(history.history['val_dice_coefficient'])
    plt.title('Dice Coefficient')
    plt.ylabel('Dice Coefficient')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train RS-Net for cloud detection')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save results')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--height', type=int, default=256, help='Image height')
    parser.add_argument('--width', type=int, default=256, help='Image width')
    parser.add_argument('--channels', type=int, default=3, help='Number of input channels')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--pretrained', type=str, default=None, help='Path to pretrained model')
    
    args = parser.parse_args()
    
    train_model(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=(args.height, args.width),
        input_channels=args.channels,
        initial_lr=args.lr,
        pretrained_model=args.pretrained
    )