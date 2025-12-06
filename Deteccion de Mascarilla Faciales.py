import cv2
import mediapipe as mp
import requests
import numpy as np
import os
import random
# Enlaces de cualquier sitio web (JPG, PNG, JPEG)
ruta_imagenes = [
    ("Persona1", "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS01Zi3y9iIG8HC9_KtCmVHyD95Mw9FVVcONA&s"),
    ("Persona2", "https://www.apeseg.org.pe/wp-content/uploads/2021/06/GettyImages-1215709494-1.jpg"),
    ("Persona3", "https://cloudfront-us-east-1.images.arcpublishing.com/infobae/72B4DFE6ZVHTNGDRRE6IBRMYQ4.jpg"),
]


# Crear carpetas del dataset
os.makedirs("Dataset/Con_Mascarilla", exist_ok=True)
os.makedirs("Dataset/Sin_Mascarilla", exist_ok=True)

# Inicializar MediaPipe Face Detection
mp_face = mp.solutions.face_detection

def descargar_imagen(url):
    try:
        response = requests.get(url, timeout=10)
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        return image
    except requests.RequestException as e:
        print(f"Error al descargar la imagen {url}: {e}")
        return None
    except Exception as e:
        print(f"Error inesperado al procesar imagen {url}: {e}")
        return None

with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    contador = 0

    for nombre, url in ruta_imagenes:
        image = descargar_imagen(url)
        if image is None:
            continue

        # Convertir a RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Procesar con MediaPipe
        results = face_detection.process(image_rgb)

        if not results.detections:
            print(f"No se detectaron rostros en: {nombre}")
            continue

        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            ih, iw, _ = image.shape
            x, y, w, h = int(bbox.xmin * iw), int(bbox.ymin * ih), int(bbox.width * iw), int(bbox.height * ih)

            rostro = image[y:y+h, x:x+w]

            if rostro.size == 0:
                continue

            # Clasificación aleatoria simulada
            tiene_mascarilla = random.choice([True, False])
            label = "Con Mascarilla" if tiene_mascarilla else "Sin Mascarilla"
            color = (0, 255, 0) if tiene_mascarilla else (0, 0, 255)

            # Guardar imagen recortada
            file_path = f"Dataset/{label}/{nombre}_{contador}.jpg"
            cv2.imwrite(file_path, rostro)
            contador += 1

            # Dibujar cuadros en la imagen original
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Mostrar resultado final por cada imagen descargada
        cv2.imshow("Detección Facial con MediaPipe", image)
        cv2.waitKey(3000) # conteo de muestreo de imagenes por segundo

cv2.destroyAllWindows()
print("\n Dataset generado correctamente en la carpeta 'Dataset'")
