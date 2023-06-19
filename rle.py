import numpy as np

def multi_rle_encode(img):
    labels = label(img[:, :, 0])
    return [rle_encode(labels==k) for k in np.unique(labels[labels>0])]

def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def rle_decode(mask_rle_str, shape=(768, 768)):
    '''
    Convert run-length encoded mask to a binary mask
    '''
    # Split the run-length encoded string
    s = mask_rle_str.split()
    
    starts = np.asarray(s[0:][::2], dtype=int) - 1 # even numbers
    lengths = np.asarray(s[1:][::2], dtype=int) # odd numbers
    
    ends = starts + lengths
    
    # Initialize an array for the binary mask
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    
    # Set the pixels corresponding to the mask region to 1
    for start, end in zip(starts, ends):
        img[start:end] = 1
    
    # Reshape the array to the desired shape
    return img.reshape(shape).T

def masks_as_image(in_mask_list):
    '''
    Combine individual ship masks into a single mask array
    
    Args:
        in_mask_list (list): List of ship masks (run-length encoded strings)
    
    Returns:
        numpy.ndarray: Combined mask array
    '''
    # Initialize an array to hold the combined mask
    all_masks = np.zeros((768, 768), dtype=np.int16)
    
    # Iterate over the ship masks and add them to the combined mask
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks += rle_decode(mask)
    
    # Expand dimensions to match the expected shape
    return np.expand_dims(all_masks, -1)