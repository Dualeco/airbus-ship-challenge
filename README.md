# Airbus Ship Detection with U-net and Dice score

Hi! My name is Pavlo and this is my solution for the ship semantic segmentation problem.

## Code
This repo contains the following files:
- `requirements.txt`
- `unet-kaggle.ipynb` - EDA, data manipulations, and training results are here.
- `model.py ` - resources for model building and training. It is used in `unet-kaggle.ipynb` to train the model.
- `inference.py` - here is where the model is tested. It alsocontains a function that can encode a solution DataFrame. There weren't strict requirements on this file, so to show some results, I predict some values on the validation set and calculate `IoU` and `Dice Loss` on the results. `N` value can be tweaked to control the sample quantity.
- `image_utils.py` - functions to random crop and augment images. They are probably suboptimal, but in the small timeframe of the work, some things were easier to write.
- `lrfind.py`- my implementation of plotting a learning_rate/loss graph
- `rle.py` - utils for decoding rle-masks
- `callbacks.py` - code for custom learning rate and loss trackers

## EDA
The dataset contains 768x768 images of ships from above. Some images don't have any ships in them and others contain from 1 to several ships.

#### Data Imbalance
The data is imbalanced. Firstly, the number of empty images in the dataset is larger than one of the images with ships:
```
N of images in the dataset: 192556
N of non-empty images: 81723
N of empty images: 110833
```

Secondly, the non-empty images generally have only a few ship pixels, and the rest is background:![enter image description here](https://storage.googleapis.com/kagglesdsdata/competitions/9988/868324/test_v2/01963652b.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2@kaggle-161607.iam.gserviceaccount.com/20230618/auto/storage/goog4_request&X-Goog-Date=20230618T164754Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=8c68222e44000f09f13e3ec728a3524b33a5f8d95f70d915e30436ec764ec633cd7eee76f83de25e2010eba94061653ee8cf7883061c91f5bdef274916f2f735057d2e96603e3787695261a44542f5a4b3ee76353a84baa1f34c5586fc792e94e54d1bae62e52df40a9d0d48fcbff78ba5a03a23ed26d7448befd3d2ed794a05e50f4f8e71ff77f72950e389b4506c47d1e43462de2ea096725b2c11b1d83c9e769ae0859178f766eb43421b5b70d97bff47d99ff419f6fdbccf356b86ff3f8599338f101a8b77f1b979dfe90c3d3886963e0d848c954ed8b410e1eaa72408ec529b3672424a021fe61e287c9fa32a03acb32d53c3fbe632d51041d6b6fe4ce6)

Thirdly, some images contain more ships than other:
![enter image description here](https://imgtr.ee/images/2023/06/18/Y3nI3.png)

To ease the imbalance in data, I have edited the dataset in the following way:
- An arbitrary `N` of empty and non-empty 256x256 images and masks were collected using 768x768 source files and random cropping for each batch.
- From those `N` images, approximately `BATCH_SIZE / 2` empty and non empty images were selected to assemble a single batch.
- The original empty images were discarded to avoid even further category imbalance.

This solution has caused some technical overhead, but in return, it gave my model a visible performance impovement.

#### Augmentation
After successfully cropping and balancing the data, it was a good idea to add some variability to it. To do so, I have used Tensorflow's `ImageDataGenerator` to flip and shear images and masks. I decided to avoid zooming and rotating the data, because the ships in some images were heavily distorted by it, and some small ones  could even disappear. This would have made the resulting dataset visibly more distorted and less balanced.

This image is an exmple of a typical 32 image batch. Note that it is expected to have some fluctuations in favor of empty images.
![enter image description here](https://i.ibb.co/yscCsNy/image.png)

### Model selection

####  Architecture
The task requires the problem to be solved using 256x256 U-net with Dice score. Because of the memory constraints to fit in the standard U-net, I have chosen the reduced architecture without the "Class label" part from https://ieeexplore.ieee.org/document/8363710.
![Universal multi-modal deep network for classification and segmentation of  medical images | Semantic Scholar](https://d3i71xaburhd42.cloudfront.net/c70b21c746d89c4729f22dc16c40dc2f9b3ae8dd/2-Figure2-1.png)
I have also added a normalization layer after the `Input` layer to normalize each input image by 255.
`BATCH_SIZE` value was chosen to be 32.

#### Loss
The training loss was Dice Loss defined in this code:
```
def dice_loss(y_true, y_pred, smooth=1):
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    
    intersection = tf.reduce_sum(y_true * y_pred)
    
    nom = 2.0 * intersection + smooth
    denom = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth
    
    dice_coef = nom / denom
    
    dice_loss = 1.0 - dice_coef
    
    return dice_loss
```
Where `smooth` is the factor that avoids division by `0`.

#### Metric
For the training metric I chose `tensorflow.keras.metrics.MeanIoU`.

### Training
#### Learning rate selection
Training with learning rate exponentially growing from 1e-5 to 10 can be useful to plot a lr/loss graph and select an optimal lr where the loss drops the fastest.
![enter image description here](https://i.ibb.co/GTkQSTd/image.png)
#### Training-validation split
Training/validation split in this work is 0.9 / 0.1 . Validation set DataFrame is saved to `validation.csv`  to be later used in `inference.py`