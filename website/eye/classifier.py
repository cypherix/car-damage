
# coding: utf-8

# In[1]:

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import numpy as np
import keras.backend as K
import cv2
import matplotlib.pyplot as plt

def image_reader(path):
    im  = Image.open(path)
    im = im.resize((384,288), Image.ANTIALIAS)
    im = np.expand_dims(np.array(im)/255.0, axis=0)
    return im

def imposer(img_c_array, mask):
    img_m = Image.fromarray(np.uint8(np.squeeze(mask)*255))
    img_m = img_m.convert('RGBA')
    img_c = Image.fromarray(np.uint8(img_c_array*255))
    img_c = img_c.convert('RGBA')
    return Image.blend(img_c, img_m, 0.4)


def prediction():
    
    """
    Function to predict if the retina image has diabetic retinopathy or not.
    
    Parameters
    ----------
    None
    
    Returns
    -------
    y_pred: bool
            Whether or not the retina has diabetic retinopathy.
    percent_chance: float
            Percentage of chance the retina image has diabetic retinopathy.
    """
    
    import os
    PROJECT_PATH = os.path.abspath(os.path.dirname(__file__))

    K.clear_session()
    mod=load_model(PROJECT_PATH+'/car_mask_model_TL.h5')
    CAPTHA_ROOT = os.path.join(PROJECT_PATH,'test_images/uploaded/uploaded.jpg')
    
    test_data = image_reader(CAPTHA_ROOT)
    predicted = mod.predict(test_data, steps=1)

    predicted[predicted>0.4]=1
    predicted[predicted!=1]=0
    
    imposed_image = imposer(test_data[0], predicted[0])
    # imposed_image = imposed_image.convert("RGB")
    SAVE_ROOT = os.path.join(PROJECT_PATH,'static/img/result/result.png')
    imposed_image.save(SAVE_ROOT)
    
    return True, SAVE_ROOT
# In[17]:

if __name__ == '__main__':
    print(prediction())


