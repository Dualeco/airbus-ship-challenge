import pandas as pd
import rle
import tensorflow as tf
from keras.models import load_model
from model import log_cosh_dice_loss
import os
from skimage.io import imread, imsave
import numpy as np
from tensorflow.keras.metrics import MeanIoU
from joblib import Parallel, delayed
import shutil
import warnings
warnings.filterwarnings("ignore")

MODEL = 'airbus_ships_iou_score-9+04-0.79.hdf5' 

def predict_enc(img_id, model):
    rgb_path = os.path.join("./test_v2", img_id)
    img = imread(rgb_path)
    
    all_im = split_768_into_256(img)
    
    p = np.where(model.predict(np.array(all_im)) > 0.5, 1., 0.)
    
    res = np.empty((768, 768, 1))
    
    put_256_in_768(p, res)
    
    e = rle.rle_encode(res)
    return e

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
    model = load_model(MODEL, {'log_cosh_dice_loss': log_cosh_dice_loss})
    
    import os
    ids = [f for f in os.listdir("./test_v2") if not f.startswith('.')][:300]

    RES_DIR = "./test_y/"
    if os.path.exists(RES_DIR):
       shutil.rmtree(RES_DIR)
    os.makedirs(RES_DIR)
    
    if (encodings): 

        encodings = Parallel(n_jobs=1)(delayed(predict_enc)(img_id, model) for img_id in ids)
            
        df = pd.DataFrame({
            'ImageId': ids,
            'EncodedPixels': encodings
        })
        df = df.set_index('ImageId')
        df.to_csv("submission.csv")
    else:

        masks = Parallel(n_jobs=1)(delayed(predict_mask)(img_id, model) for img_id in ids)
	
        for i in range(len(ids)):
            imsave(os.path.join(RES_DIR, ids[i]), masks[i])
        
    
def split_768_into_256(rgb):
    l = []
    for i in range(3):
        for j in range(3):
            l.append(rgb[i * 256:i * 256 + 256, j * 256:j * 256 + 256])
            
    return l

def put_256_in_768(preds, res):
    for i in range(3):
        for j in range(3):
            res[i * 256:i * 256 + 256, j * 256:j * 256 + 256] = preds[i * 3 + j]
    
def inference():
    #test_on_validation()
    build_submission(encodings=False)
    
if __name__ == '__main__':
    inference()