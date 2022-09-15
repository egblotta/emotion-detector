from deepface import DeepFace
from deepface.basemodels import VGGFace
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from contextlib import redirect_stdout
import pickle

# ruta de las imagenes
img1_path = 'emi.jpg'
db_path   = 'D:/UDA/Tesis/deepface-detector/e'

model = VGGFace.loadModel()

# funcion de verificación
def find(img1_path, db_path):
    # img1 = cv2.imread(img1_path)
    # plt.imshow(img1[:, :, : : -1])

    # find = DeepFace.find(img1_path, db_path,model_name='VGG-Face', model = model)
    find = DeepFace.detectFace(img1_path, detector_backend='opencv')
    plt.imshow(find)
    plt.show()
    # print(pd.DataFrame(find.head()))

    find = find * 255
    cv2.imwrite("face.jpg", find[:, :, ::-1])

# llamda a la función
find(img1_path, db_path)
