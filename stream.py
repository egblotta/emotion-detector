import firebase_admin
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import json

from datetime import datetime
from firebase_admin import credentials, firestore, db
from deepface import DeepFace
from bs4 import BeautifulSoup


# url firebase database
if not firebase_admin._apps:
    cred = credentials.Certificate('credentials/emotion-detector-be028-firebase-adminsdk-1b73m-c7cb3ab9fb.json')
    default_app = firebase_admin.initialize_app(cred, {
    'databaseURL': "https://emotion-detector-be028-default-rtdb.firebaseio.com/"
    })

# current date and time
curDT = datetime.now()

# current time
time = curDT.strftime("%H:%M:%S")
print("time:", time)

# current date and time
date = curDT.strftime("%d-%m-%Y")
print("date:", date)

ref = db.reference('/emotion-detections')

# ruta de la base de datos
db_path   = 'D:/UDA/Tesis/deepface-detector/e'

# ruta de las imagenes
img1_path = 'pexels\pexels3.jpg'
img2_path = 'pexels\pexels1.jpeg'

# funcion de verificación
def analyze(img_path):
    img1 = cv2.imread(img_path)
    plt.imshow(img1[:, :, : : -1])

    analyze = DeepFace.analyze(img_path, actions = ['emotion'])

    obj_dump     = json.dumps(analyze)
    parsed       = json.loads(obj_dump)
    response     = json.dumps(parsed, indent=4)
    print('resultado: ',response)   

    # save by date and then by time
    users_ref = ref.child(date).child('Salida').child('Emiliano').child(time)
    users_ref.set(parsed)

# funcion de verificación
def stream(db_path):
    stream = DeepFace.stream(db_path, detector_backend='opencv', distance_metric='euclidean')   #opencv, ssd, mtcnn, dlib, retinaface
    stream = stream * 255
    cv2.imwrite("face.jpg", stream[:, :, ::-1])

# llamada a la función
analyze(img1_path)
stream(db_path)