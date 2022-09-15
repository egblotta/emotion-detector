from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt
import json

models  = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace"]

# ruta de las imagenes
img1_path = 'Emiliano\Emiliano3.jpg'
img2_path = 'Emiliano\Emiliano4.jpg'

# funcion de verificación
def verify(img_path1,img_path2):
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)
    plt.imshow(img1[:, :, : : -1])
    plt.imshow(img2[:, :, : : -1])

    verification = DeepFace.verify(img_path1, img_path2, model_name=models[0])
    obj_dump     = json.dumps(verification)
    parsed       = json.loads(obj_dump)
    response     = json.dumps(parsed, indent=4)
    print('resultado: ',response)

# llamda a la función
verify(img1_path,img2_path)