import pandas as pd
import rle
import tensorflow as tf
from keras.models import load_model
from model import dice_loss
import os
from skimage.io import imread, imsave
import numpy as np
from tensorflow.keras.metrics import MeanIoU
from joblib import Parallel, delayed
import shutil

MODEL = 'airbus_ships_final-14-0.36.hdf5' 

def predict_enc(img_id, model):
    rgb_path = os.path.join("./test_v2", img_id)
    img = imread(rgb_path)
    
    all_im = split_768_into_256(img)
    
    p = np.where(model.predict(np.array(all_im)) > 0.5, 1., 0.)
    
    res = np.empty((768, 768, 1))
    
    put_256_in_768(p, res)
    
    return rle.rle_encode(res)

def predict_mask(img_id, model):
    rgb_path = os.path.join("./test_v2", img_id)
    img = imread(rgb_path)
    
    all_im = split_768_into_256(img)
    
    p = np.where(model.predict(np.array(all_im)) > 0.5, 1., 0.)
    
    res = np.empty((768, 768, 1))
    
    put_256_in_768(p, res)
    
    return res

def build_submission(encodings=True):
    # Select any model
    model = load_model(MODEL, {'dice_loss': dice_loss})
    
    import os
    ids = [f for f in os.listdir("./test_v2") if not f.startswith('.')][:300]
    print(ids)

    RES_DIR = "./test_v2/res/"
    if os.path.exists(RES_DIR):
       shutil.rmtree(RES_DIR)
       os.makedirs(RES_DIR)
    
    if (encodings): 
        # imread could be parallelized using joblib

        encodings = Parallel(n_jobs=1)(delayed(predict_enc)(img_id, model) for img_id in ids)
            
        df = pd.DataFrame({
            'ImageId': ids,
            'EncodedPixels': preds
        }, index = 'ImageId')
            
        df.to_csv("submission.csv")
    else:

        [print(img_id) for img_id in ids]
        masks = Parallel(n_jobs=1)(delayed(predict_mask)(img_id, model) for img_id in ids)
	
        for i in range(len(ids)):
            imsave(os.path.join(RES_DIR, ids[i]), masks[i])
        
    
def split_768_into_256(rgb):
    l = []
    for i in range(3):
        for j in range(3):
            l.append(rgb[i * 256:i * 256 + 256, j * 256:j * 256 + 256])
    
    print(f"Preds {len(l)}")
            
    return l

def put_256_in_768(preds, res):
    for i in range(3):
        for j in range(3):
            res[i * 256:i * 256 + 256, j * 256:j * 256 + 256] = preds[i * 3 + j]
    
def test_on_validation():
    # How many samples to predict
    N = 300
    
    # import test data
    valid_df = pd.read_csv('validation_set.csv')
    data = list(valid_df.groupby('ImageId'))
    rgbs = []
    masks = []
    
    for img_id, img_mask in data[:N]:
        rgb_path = os.path.join("./train_v2", img_id)
        
        img = imread(rgb_path)
        mask = rle.masks_as_image(img_mask['EncodedPixels'].values)
        
        rgbs.append(img)
        masks.append(mask)
    
    all_im = []
    for im in rgbs[:N]:
        all_im += split_768_into_256(im)
        
    all_mask = []
    for im in masks[:N]:
        all_mask += split_768_into_256(im)
        
    all_mask = np.array(all_mask).astype(np.float32)
    
    # Select any model
    model = load_model(MODEL, {'dice_loss': dice_loss})
    predictions = np.where(model.predict(np.array(all_im)) > 0.5, 1, 0).astype(np.float32)
    
    score = MeanIoU(num_classes=2)
    score.update_state(all_mask, predictions)
    
    loss = dice_loss(all_mask, predictions)
    
    print(f"IoU score: {score.result().numpy()}")
    print(f"Dice loss: {loss.numpy()}")
    
    # Patially build submission
    # build_submission(model)
    
def inference():
    #test_on_validation()
    build_submission(encodings=False)
    
if __name__ == '__main__':
    inference()