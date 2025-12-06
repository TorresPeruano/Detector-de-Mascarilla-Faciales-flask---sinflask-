from flask import Flask, render_template, request, send_file, jsonify
import cv2
from ultralytics import YOLO 
import numpy as np
import os
import random
import base64
import io
import threading
from werkzeug.utils import secure_filename
from datetime import datetime

# --- Configuración del Modelo YOLO con Thread-Safety ---
# Cargar el modelo YOLOv8 para detección de rostros
# El modelo se descargará automáticamente si no existe localmente.
model = None
model_lock = threading.Lock()

try:
    model = YOLO("yolov8n-face.pt")
    print("✓ Modelo YOLO cargado correctamente")
except Exception as e:
    print(f"✗ Error al cargar el modelo YOLO: {e}")
    raise RuntimeError(f"No se pudo cargar el modelo YOLO: {e}")

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif', 'bmp'}

# Crear carpetas del dataset
os.makedirs("Dataset/Con_Mascarilla", exist_ok=True)
os.makedirs("Dataset/Sin_Mascarilla", exist_ok=True)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(image):
    """
    Detecta rostros en la imagen usando YOLOv8 y retorna:
    - Imagen procesada con cuadros y etiquetas
    - Lista de resultados (etiqueta y coordenadas)
    """
    detections_list = []
    results_detections = []
    
    # Thread-safe inference con modelo YOLO
    with model_lock:
        results = model(image, verbose=False)
    
    for r in results:
        # 'r.boxes' contiene todos los cuadros delimitadores detectados
        for box in r.boxes:
            # Obtener coordenadas del cuadro (xmin, ymin, xmax, ymax) en píxeles enteros
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            
            # Calcular ancho y alto (formato x, y, w, h)
            x = x1
            y = y1
            w = x2 - x1
            h = y2 - y1
            
            # Asegurar que las coordenadas son válidas
            if w <= 0 or h <= 0:
                continue
            
            # Recortar el rostro
            rostro = image[y:y+h, x:x+w].copy()
            
            if rostro.size == 0:
                continue
            
            # Clasificación aleatoria simulada
            tiene_mascarilla = random.choice([True, False])
            label = "Con Mascarilla" if tiene_mascarilla else "Sin Mascarilla"
            color = (0, 255, 0) if tiene_mascarilla else (0, 0, 255) # Verde / Rojo
            
            # Guardar imagen recortada en el dataset
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"Dataset/{label}/rostro_yolo_{timestamp}_{random.randint(0, 999)}.jpg"
            cv2.imwrite(file_path, rostro)
            
            # Dibujar cuadros en la imagen original
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            detections_list.append({
                'id': len(detections_list),
                'label': label,
                'coords': {'x': x, 'y': y, 'w': w, 'h': h}
            })
            
            results_detections.append({
                'label': label,
                'color': color,
                'saved_path': file_path
            })
            
    return image, detections_list, results_detections

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed. Use: jpg, jpeg, png, gif, bmp'}), 400
    
    try:
        # Leer imagen del archivo
        file_stream = file.read()
        nparr = np.frombuffer(file_stream, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image file'}), 400
        
        # Procesar imagen con YOLOv8
        processed_image, detections, results = process_image(image)
        
        if not detections:
            return jsonify({
                'error': 'No faces detected in the image',
                'detections': 0
            }), 200
        
        # Convertir imagen procesada a bytes
        success, buffer = cv2.imencode('.jpg', processed_image)
        if not success:
            return jsonify({'error': 'Error encoding image'}), 500
        
        img_bytes = buffer.tobytes()
        
        # Guardar para descarga
        download_filename = f"processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        download_path = os.path.join(app.config['UPLOAD_FOLDER'], download_filename)
        with open(download_path, 'wb') as f:
            f.write(img_bytes)
        
        # Preparar respuesta
        response = {
            'success': True,
            'message': f'{len(detections)} face(s) detected',
            'detections': detections,
            'download_filename': download_filename,
            'image_base64': base64.b64encode(img_bytes).decode('utf-8'),
            'summary': {
                'with_mask': len([d for d in detections if d['label'] == 'Con Mascarilla']),
                'without_mask': len([d for d in detections if d['label'] == 'Sin Mascarilla']),
                'total': len(detections)
            }
        }
        
        return jsonify(response), 200
    
    except FileNotFoundError as e:
        return jsonify({'error': f'File system error: {str(e)}'}), 500
    except ValueError as e:
        return jsonify({'error': f'Invalid value: {str(e)}'}), 400
    except Exception as e:
        print(f"Error en /upload: {str(e)}")
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

@app.route('/download/<filename>')
def download_file(filename):
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        return send_file(file_path, mimetype='image/jpeg', as_attachment=True, download_name=filename)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/dataset-status')
def dataset_status():
    """Retorna información sobre el dataset generado"""
    try:
        con_mascarilla = len(os.listdir("Dataset/Con_Mascarilla"))
        sin_mascarilla = len(os.listdir("Dataset/Sin_Mascarilla"))
        
        return jsonify({
            'con_mascarilla': con_mascarilla,
            'sin_mascarilla': sin_mascarilla,
            'total': con_mascarilla + sin_mascarilla
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)