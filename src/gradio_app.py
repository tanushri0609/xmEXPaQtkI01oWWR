from tensorflow.keras.models import load_model

import numpy as np
import gradio as gr



def classify_image(resized_img):
    mobile_model_path = '../Final_model/mobile_model.h5'
    final_model=load_model(mobile_model_path)
    prediction = final_model.predict(np.reshape(resized_img, (1,) + resized_img.shape))[0][0]
    probs = {'flip':float(prediction), 'not flip':float(1-prediction)}
    #solu = sorted(probs,reverse=True)
    #print(probs)
    return(probs)
    

interface = gr.Interface(fn=classify_image,
                         inputs=gr.Image(shape=(224, 224)),
                         outputs='label',
                         description = "Example image",
                         title = "Image Classifier",
                         examples=[
                             "../Data/images/testing/flip/0003_000000022.jpg",
                             "../Data/images/testing/flip/0002_000000017.jpg",
                             "../Data/images/testing/notflip/0043_000000024.jpg",
                             "../Data/images/testing/notflip/0038_000000023.jpg"]
                         )
interface.launch()