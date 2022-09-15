from deepface import DeepFace

# ruta de las imagenes
img1_path = 'e\Emi_2.jpg'

# funcion de verificación
def represent(img1_path):

    find = DeepFace.represent(img1_path)
    print(find)

# llamada a la función
represent(img1_path)