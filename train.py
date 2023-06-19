from img_utils import create_aug_gen, balanced_crop_gen    
import model
import pandas as pd

CROP_SIZE = 256
IMG_C = 3
BATCH_SIZE = 32
        
def get_aug_gen(df):
    crop_gen = img_utils.balanced_crop_gen(train_df, TRAIN_FOLDER, BATCH_SIZE, RANDOM_STATE, IMG_SIZE, CROP_SIZE)
    return img_utils.create_aug_gen(crop_gen, RANDOM_STATE)

def train():
    train_df = pd.read_csv('training_set.csv')  
    valid_df = pd.read_csv('validation_set.csv')
    
    train_aug_gen = get_aug_gen(train_df)
    valid_aug_gen = get_aug_gen(valid_df)
    
    TRAIN_STEP_COUNT = train_df.shape[0] // BATCH_SIZE
    VALID_STEP_COUNT = valid_df.shape[0] // BATCH_SIZE
    
    # Model
    TRAIN_LEARNING_RATE = 8e-5
    
    SAVE_PATH = "airbus_ships_trained-{epoch:02d}-{val_loss:.2f}.hdf5"
    
    TRAIN_EPOCHS = 30
    
    model.train_unet(train_aug_gen, TRAIN_STEP_COUNT, valid_aug_gen, VALID_STEP_COUNT, 
                     crop_size=CROP_SIZE, img_c=IMG_C,
                     lr=TRAIN_LEARNING_RATE, epochs=TRAIN_EPOCHS, save_path=SAVE_PATH)
    
if __name__ == '__main__':
    train()