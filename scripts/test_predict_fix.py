import sys
sys.path.insert(0, 'D:\\stremlit_brain')
from PIL import Image
from tensorflow import keras
from classifier.classify import classifier

model = keras.models.load_model('Model/model.keras')
img = Image.open('Brain_Tumor_Datasets/Brain_Tumor_Datasets/test/YES/yes_1892.jpg')
label, score = classifier(img, model, {0:'No Brain Tumor',1:'Brain Tumor'})
print(label, score)
