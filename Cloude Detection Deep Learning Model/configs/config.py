# config.py

# Dataset configuration
DATA_CONFIG = {
    'dataset_path': 'data/raw/38-Cloud',
    'img_size': (256, 256),
    'val_split': 0.2,
    'test_split': 0.1
}

# Model configuration
MODEL_CONFIG = {
    'input_shape': (256, 256, 3),
    'num_classes': 1,
    'learning_rate': 1e-4,
    'batch_size': 16,
    'epochs': 50
}

# Training configuration
TRAIN_CONFIG = {
    'model_dir': 'data/models',
    'checkpoint_dir': 'data/models/checkpoints',
    'log_dir': 'data/models/logs'
}

# Prediction configuration
PREDICT_CONFIG = {
    'threshold': 0.5,
    'output_dir': 'data/predictions'
}