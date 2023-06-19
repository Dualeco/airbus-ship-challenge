import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from skimage.io import imread
import rle

# Makes vaguely balanced random crop
def balanced_crop_gen(df, train_folder, batch_size, seed, img_size, crop_size, debug = False):
    all_batches = list(df.groupby('ImageId'))
    
    # Setup per-batch random crop
    np.random.seed(seed)
    
    n = 0
    half_batch = batch_size // 2
    size = batch_size + half_batch
    rand_x = np.random.randint(0, img_size - crop_size, size=size).tolist()
    rand_y = np.random.randint(0, img_size - crop_size, size=size).tolist()
    
    rgb = []
    mask = []
    non_zero_means=[]
    non_zero_mean_ids=[]
    zero_mean_ids=[]
    out_rgb=[]
    out_mask=[]

    while True:
        np.random.shuffle(all_batches)
        for c_img_id, c_masks in all_batches:
            rgb_path = os.path.join(train_folder, c_img_id)
            
            # Random crop
            i = rand_x[n]
            j = rand_y[n]
            
            c_img = imread(rgb_path)[i: i + crop_size, j: j + crop_size]
            c_mask = rle.masks_as_image(c_masks['EncodedPixels'].values)[i: i + crop_size, j: j + crop_size]

            rgb.append(c_img)
            mask.append(c_mask)
            m = c_mask.mean()
            if m > 0.001:
                non_zero_means.append(m)
                non_zero_mean_ids.append(n)
                n += 1
            elif m == 0.:
                zero_mean_ids.append(n)
                n += 1
            
            
            if (n == size):
                n = 0
                
                top_ids = non_zero_mean_ids
                top_ids_len = len(top_ids)
                
                if (top_ids_len > half_batch):
                    top_ids_len = half_batch
                    top_ids = non_zero_mean_ids[:half_batch]
                
                btm_ids = zero_mean_ids[:batch_size - top_ids_len]
                
                for i in btm_ids:
                    out_rgb.append(rgb[i])
                    out_mask.append(mask[i])
                    
                for i in top_ids:
                    out_rgb.append(rgb[i])
                    out_mask.append(mask[i])
                    
                out_rgb_a = np.stack(out_rgb, 0)
                out_mask_a = np.stack(out_mask, 0).astype(np.float32)
                
                assert out_rgb_a.shape == (batch_size, crop_size, crop_size, 3)
                assert out_mask_a.shape == (batch_size, crop_size, crop_size, 1)
                
                if (debug):
                    print("Before Augmentation")
                    
                    h = batch_size // 4
                    for i in range(h):
                        f, axarr = plt.subplots(1, 8)
                        f.set_dpi(200)
                        
                        for j in range(4):
                            axarr[2 * j].imshow(out_rgb_a[4 * i + j].astype(int))
                            axarr[2 * j].axison = False  
                            axarr[2 * j + 1].imshow(out_mask_a[4 * i + j])
                            axarr[2 * j + 1].axison = False  
                        
                    plt.show()
                    
                yield out_rgb_a, out_mask_a
                
                rand_x = np.random.randint(0, img_size - crop_size, size=size)
                rand_y = np.random.randint(0, img_size - crop_size, size=size)
                rgb.clear()
                mask.clear()
                out_rgb.clear()
                out_mask.clear()
                non_zero_means.clear()
                non_zero_mean_ids.clear()
                zero_mean_ids.clear()

dg_args = dict(featurewise_center = False, 
                  samplewise_center = False,
                  zoom_range = [0.95, 0.95],
                  shear_range=3,
                  horizontal_flip = True, 
                  vertical_flip = True,
                  fill_mode = 'constant',
                  data_format = 'channels_last')


image_gen = ImageDataGenerator(**dg_args)
label_gen = ImageDataGenerator(**dg_args)
        
        
def create_aug_gen(in_gen, seed, debug=False):
    np.random.seed(seed)
                           
    for in_x, in_y in in_gen:
        
        batch_size = in_x.shape[0]
        
        g_x = image_gen.flow(in_x, 
                             batch_size = batch_size, 
                             seed = seed, 
                             shuffle=True)
        
        g_y = label_gen.flow(in_y, 
                             batch_size = batch_size, 
                             seed = seed, 
                             shuffle=True)

        x, y = next(g_x), (next(g_y) > 0.5).astype(np.float32) # compensate for interpolation of masks
        
        if (debug):
            print("After Augmentation")
            
            h = batch_size // 4
            for i in range(h):
                f, axarr = plt.subplots(1, 8)
                f.set_dpi(200)
                
                for j in range(4):
                    axarr[2 * j].imshow(x[4 * i + j].astype(int))
                    axarr[2 * j].axison = False  
                    axarr[2 * j + 1].imshow(y[4 * i + j])
                    axarr[2 * j + 1].axison = False  
                
                plt.show()
        
        yield x, y  