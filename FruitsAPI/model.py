import pickle
from tensorflow.keras.models import load_model
import tensorflow as tf
import cv2, base64
import numpy as np

class Model:
    def __init__(self):
        with open('../categories.pkl', 'rb') as output:
            self.categories = pickle.load(output)
        self.model = load_model('../my_model.h5')
        self.IMG_SIZE = 100

    def process(self, b64_string):

        data = base64.b64decode(b64_string)
        nparr = np.fromstring(data, np.uint8)
        img_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        new_array = cv2.resize(img_array, (self.IMG_SIZE, self.IMG_SIZE))
        img = new_array.astype('float32') / 255
        return img
    
    def predict(self, img):
        pred = self.model.predict_classes(np.array([img]))
        probs = self.model.predict_proba(np.array([img]))
        index = int(pred)
        name = self.categories[index]
        probability = probs[0, index]
        return name, probability

        

print(tf.__version__)