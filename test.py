from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import matplotlib.pyplot as plt

model = load_model('./models/simple_CNN_model.h5')  

img_folder = './img/'  

CLASS_LABELS = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sadness', 'Surprise']

for img_file in os.listdir(img_folder):
    img_path = os.path.join(img_folder, img_file)
    
    if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    img = image.load_img(img_path, target_size=(48, 48)) 
    img_array = image.img_to_array(img)  
    img_array = np.expand_dims(img_array, axis=0)  
    img_array /= 255.0  

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    predicted_label = CLASS_LABELS[predicted_class[0]]

    plt.imshow(img)
    plt.title(f'Predicted: {predicted_label}')
    plt.axis('off')
    plt.show()
