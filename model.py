# model.py

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    Lambda,
    Dense,
    Flatten,
    MaxPooling2D,
    BatchNormalization,
    Dropout
)
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ReduceLROnPlateau

# Hyperparameters
img_width, img_height = 128, 128
input_shape = (img_width, img_height, 1)

# Fungsi untuk menghitung jarak Euclidean
def euclidean_distance(vectors):
    (featsA, featsB) = vectors
    sum_square = K.sum(K.square(featsA - featsB), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

# Contrastive loss
def contrastive_loss(y_true, y_pred):
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

# Custom accuracy metric
def accuracy(y_true, y_pred):
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

def build_siamese_model(input_shape):
    input = Input(shape=input_shape)
    
    # Feature extraction layers (CNN)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)
    x = Dropout(0.3)(x)
    
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)
    x = Dropout(0.3)(x)
    
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)
    x = Dropout(0.3)(x)
    
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    embedding = Dense(128)(x)  # Embedding layer
    
    # Normalize embeddings
    embedding = Lambda(lambda x: K.l2_normalize(x, axis=-1))(embedding)

    model = Model(inputs=input, outputs=embedding)

    # Define the learning rate reduction callback
    learning_rate_reduction = ReduceLROnPlateau(
        monitor='val_accuracy', 
        patience=2, 
        verbose=1, 
        factor=0.7, 
        min_lr=1e-10
    )
    
    return model, learning_rate_reduction

def get_siamese_model(learning_rate):
    base_model, learning_rate_reduction = build_siamese_model(input_shape)
    # Two inputs for the model
    input_A = Input(shape=input_shape)
    input_B = Input(shape=input_shape)
    # Generate embeddings for both inputs
    embedding_A = base_model(input_A)
    embedding_B = base_model(input_B)
    # Calculate distance between embeddings
    distance = Lambda(euclidean_distance)([embedding_A, embedding_B])
    # Build the final model
    siamese_model = Model(inputs=[input_A, input_B], outputs=distance)
    # Compile the model
    siamese_model.compile(
        loss=contrastive_loss, optimizer=tf.keras.optimizers.Adam(learning_rate),
        metrics=[accuracy]
    )
    return siamese_model, base_model
