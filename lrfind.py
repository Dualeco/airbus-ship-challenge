import model
import tensorflow as tf
import os
from tensorflow.keras.metrics import MeanIoU
import callbacks
import matplotlib.pyplot as plt

# Build a graph to find an optimal learning rate
def lr_find(X, y, crop_size, img_c, train_step_count, valid_step_count, save_path, epochs=1):
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    
    # LrFind Model
    unet = model.build_unet(crop_size, img_c)
    unet.compile(optimizer='adam', loss=model.dice_loss, metrics=[MeanIoU(num_classes=2)])
    
    checkpoint = tf.keras.callbacks.ModelCheckpoint(save_path, verbose=1, save_best_only=True)
    early_stopping = tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss')
    lr_scheduler = callbacks.PerBatchLrScheduler(model, train_step_count, epochs)
    per_batch_loss_history = callbacks.PerBatchLossHistory()
    
    history = unet.fit(X, steps_per_epoch=train_step_count, validation_data=y, validation_steps=valid_step_count,
                                epochs=epochs, callbacks=[lr_scheduler, per_batch_loss_history])
        
    filtered_lr = []
    filtered_loss = []
    skip_first = 20
    skip_last = 5
    
    for n_epoch in range(epochs):
        for i in range(n_epoch * train_step_count, (n_epoch + 1) * train_step_count):
            if i > n_epoch * train_step_count + skip_first and i < (n_epoch + 1) * train_step_count - skip_last:
                filtered_lr.append(lr_scheduler.rates[i])
                filtered_loss.append(per_batch_loss_history.losses[i])
                
    plt.plot(filtered_lr, filtered_loss)
    plt.xscale('log')
    plt.tick_params(axis='y', which='minor')
    plt.show()
    plt.savefig("lr_vs_loss.png")