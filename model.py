import callbacks
import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.metrics import MeanIoU
import tensorflow.keras.backend as K
from tensorflow.keras.losses import binary_crossentropy, BinaryCrossentropy
from img_utils import sobel_tf

def contraction(n_filters, dropout, input_layer):
    c = layers.Conv2D(filters=n_filters, kernel_size=(3,3), activation='relu', kernel_initializer='he_normal', padding='same')(input_layer)
    if (dropout != None):
        c = layers.Dropout(dropout)(c)
    c = layers.Conv2D(filters=n_filters, kernel_size=(3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c)
    
    return c

def pooling(input_layer):
    return layers.MaxPooling2D((2,2))(input_layer)

def expansion(n_filters, dropout, input_layer, concat_layer):
    e = layers.Conv2DTranspose(filters=n_filters, kernel_size=(2,2), strides=(2,2), padding='same')(input_layer)
    e = layers.concatenate([e, concat_layer])
    c = layers.Conv2D(filters=n_filters, kernel_size=(3,3), activation='relu', kernel_initializer='he_normal', padding='same')(e)
    c = layers.Dropout(dropout)(c)
    c = layers.Conv2D(filters=n_filters, kernel_size=(3,3), activation='relu', kernel_initializer='he_normal', padding='same')(e)
    
    return c

def build_unet(crop_size, img_c):
    inputs = layers.Input((crop_size, crop_size, img_c))
    norm = layers.Lambda(lambda x: sobel_tf(x))(inputs) # Normalize the pixels
    
    # Encoder
    c1 = contraction(32, 0.1, norm)
    p1 = pooling(c1)
    
    c2 = contraction(64, 0.1, p1)
    p2 = pooling(c2)
    
    c3 = contraction(128, 0.2, p2)
    p3 = pooling(c3)
    
    c4 = contraction(256, 0.2, p3)
    p4 = pooling(c4)
    
    # Bottom of U
    c5 = contraction(512, None, p4)
    
    # Decoder
    e4 = expansion(256, 0.2, c5, concat_layer=c4)
    
    e3 = expansion(128, 0.2, e4, concat_layer=c3)
    
    e2 = expansion(64, 0.1, e3, concat_layer=c2)
    
    e1 = expansion(32, 0.1, e2, concat_layer=c1)
    
    # Outputs (classification)
    outputs = layers.Conv2D(1, (1,1), activation='sigmoid')(e1)
    
    return tf.keras.Model(inputs=[inputs], outputs=[outputs])

# Dice Coefficient is equivalent to F1 score. Dice loss is differentiable
def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (
            K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score
    
    return dice_loss

def log_cosh_dice_loss(y_true, y_pred):
        x = dice_loss(y_true, y_pred)
        return tf.math.log((tf.exp(x) + tf.exp(-x)) / 2.0)

def train_unet(train_gen, train_step_count, valid_gen, valid_step_count, crop_size, img_c, lr=1e-4, epochs=30, save_path="airbus_ships_balanced-{epoch:02d}-{val_loss:.2f}.hdf5"):
    train_model = build_unet(crop_size, img_c)
    train_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss=log_cosh_dice_loss, metrics=[MeanIoU(num_classes=2, name='mean_iou')])
    
    train_checkpoint = tf.keras.callbacks.ModelCheckpoint(save_path, verbose=1, monitor='val_loss', save_best_only=True, mode='min')
    train_early_stopping = tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_mean_iou', min_delta=0.005)
    train_per_batch_loss_history = callbacks.PerBatchLossHistory()
    
    train_history = train_model.fit(train_gen, steps_per_epoch=train_step_count, validation_data=valid_gen, validation_steps=valid_step_count,
                              epochs=epochs, 
                              callbacks=[train_per_batch_loss_history, train_checkpoint, train_early_stopping])
    
def train_unet(train_model, train_gen, train_step_count, valid_gen, valid_step_count, crop_size, img_c, lr=1e-4, epochs=30, save_path="airbus_ships_balanced-{epoch:02d}-{val_loss:.2f}.hdf5"):
    train_checkpoint = tf.keras.callbacks.ModelCheckpoint(save_path, verbose=1, monitor='val_loss', save_best_only=True, mode='min')
    train_early_stopping = tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_mean_iou', min_delta=0.005)
    train_per_batch_loss_history = callbacks.PerBatchLossHistory()
    
    train_history = train_model.fit(train_gen, steps_per_epoch=train_step_count, validation_data=valid_gen, validation_steps=valid_step_count,
                              epochs=epochs, 
                              callbacks=[train_per_batch_loss_history, train_checkpoint, train_early_stopping])