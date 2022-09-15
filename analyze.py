from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt
import json

# ruta de las imagenes
img1_path = 'Emiliano\Emiliano2.jpg'
img2_path = 'pexels\pexels1.jpeg'

# detector backend
backends = ['opencv', 'ssd', 'mtcnn', 'retinaface']

# funcion de verificación
def analyze(img_path):
    img1 = cv2.imread(img_path)
    plt.imshow(img1[:, :, : : -1])

    analyze = DeepFace.analyze(img_path, actions = ['age', 'gender', 'emotion'], detector_backend='retinaface')
    obj_dump     = json.dumps(analyze)
    parsed       = json.loads(obj_dump)
    response     = json.dumps(parsed, indent=4)
    print('resultado: ',response)

# llamada a la función
analyze(img1_path)