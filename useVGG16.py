"""
vgg16 is a dnn with 16 layers 
state of the art
very std design
university of oxford
"""

# importing all libraries
import numpy as np
import cv2
from keras.preprocessing import image
from keras.applications import vgg16

# importing VGG16 model 
model=vgg16.VGG16()

# tweak image size to input size of VGG16
img=image.load_img("pubg.jpg",target_size=(224,224))
x=image.img_to_array(img)
x=np.expand_dims(x,axis=0)
x= vgg16.preprocess_input(x)

# get top 10 predictions
predictions=model.predict(x)
predicted_classes=vgg16.decode_predictions(predictions, top=10)
print("top predictions for this image are :")
for imagenet_id, name, likelihood in predicted_classes[0]:
	print("Predictions :{}-{:2f}".format(name, likelihood))