import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def rsnet_model(input_size=(256, 256, 3), n_filters=64, dropout=0.1, batch_norm=True):
    """
    RS-Net model architecture based on U-Net for cloud detection in satellite imagery.
    
    Args:
        input_size: Input image dimensions and channels (height, width, channels)
        n_filters: Number of filters for the first layer (doubled in each downsampling)
        dropout: Dropout rate
        batch_norm: Whether to use batch normalization
        
    Returns:
        model: TensorFlow Keras model
    """
    # Input layer
    inputs = Input(input_size)
    
    # Encoding path
    # Contracting Path (Downsampling)
    conv1 = conv_block(inputs, n_filters, batch_norm)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = conv_block(pool1, n_filters*2, batch_norm)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = conv_block(pool2, n_filters*4, batch_norm)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = conv_block(pool3, n_filters*8, batch_norm)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
    # Bridge
    conv5 = conv_block(pool4, n_filters*16, batch_norm)
    
    # Decoding path
    # Expansive Path (Upsampling)
    up6 = Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same')(conv5)
    concat6 = concatenate([up6, conv4], axis=-1)
    conv6 = conv_block(concat6, n_filters*8, batch_norm)
    
    up7 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same')(conv6)
    concat7 = concatenate([up7, conv3], axis=-1)
    conv7 = conv_block(concat7, n_filters*4, batch_norm)
    
    up8 = Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same')(conv7)
    concat8 = concatenate([up8, conv2], axis=-1)
    conv8 = conv_block(concat8, n_filters*2, batch_norm)
    
    up9 = Conv2DTranspose(n_filters, (3, 3), strides=(2, 2), padding='same')(conv8)
    concat9 = concatenate([up9, conv1], axis=-1)
    conv9 = conv_block(concat9, n_filters, batch_norm)
    
    # Output layer - binary cloud mask
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

def conv_block(inputs, n_filters, batch_norm=True, dropout_rate=0.1):
    """
    Convolutional block with optional batch normalization and dropout
    """
    conv = Conv2D(n_filters, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    
    if batch_norm:
        conv = tf.keras.layers.BatchNormalization()(conv)
        
    if dropout_rate > 0:
        conv = Dropout(dropout_rate)(conv)
        
    conv = Conv2D(n_filters, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv)
    
    if batch_norm:
        conv = tf.keras.layers.BatchNormalization()(conv)
        
    return conv

def compile_model(model, learning_rate=1e-4):
    """
    Compile the model with appropriate loss function and metrics
    """
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=[
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.AUC(),
            dice_coefficient
        ]
    )
    return model

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """
    Dice coefficient for measuring segmentation quality
    """
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def create_rsnet(input_channels=3, input_size=(256, 256)):
    """
    Create and compile the RS-Net model
    
    Args:
        input_channels: Number of input channels (3 for RGB, more for multi-spectral)
        input_size: Size of input images (height, width)
        
    Returns:
        model: Compiled RS-Net model
    """
    model = rsnet_model(
        input_size=(input_size[0], input_size[1], input_channels),
        n_filters=64,
        dropout=0.1,
        batch_norm=True
    )
    
    return compile_model(model, learning_rate=1e-4)