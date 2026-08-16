"""
Tomato Disease Detector - Flask Backend
Production-Ready API Server
"""

from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# ============================================================================
# LOAD MODEL AND DATA
# ============================================================================

print("Loading model...")
model = tf.keras.models.load_model('model.keras')

print("Loading class names...")
with open('class_names.txt', 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

# ============================================================================
# DISEASE TRANSLATIONS
# ============================================================================

disease_translations = {
    "Bacterial_spot": "بیکٹیریل دھبہ",
    "Early_blight": "ابتدائی جھلساؤ",
    "Late_blight": "دیرینہ جھلساؤ",
    "Leaf_Mold": "پتوں کی پھپھوندی",
    "Septoria_leaf_spot": "سیپٹوریا پتوں کا دھبہ",
    "Spider_mites Two-spotted_spider_mite": "دو دھبے والا مکڑی کا کیڑا",
    "Target_Spot": "ہدف کا دھبہ",
    "Tomato_Yellow_Leaf_Curl_Virus": "ٹماٹر پیلا پتوں کا کرل وائرس",
    "Tomato_mosaic_virus": "ٹماٹر موزیک وائرس",
    "healthy": "صحت مند",
    "powdery_mildew": "پاؤڈری پھپھوندی"
}

# ============================================================================
# TREATMENT SOLUTIONS
# ============================================================================

disease_solutions_en = {
    "Bacterial_spot": "Remove infected leaves. Apply copper-based fungicides. Ensure proper plant spacing for air circulation.",
    "Early_blight": "Use chlorothalonil fungicide. Apply mulch to prevent soil splash. Remove lower leaves.",
    "Late_blight": "Apply copper fungicide immediately. Destroy infected plants. Ensure good drainage.",
    "Leaf_Mold": "Increase ventilation in greenhouse. Reduce humidity. Remove affected leaves.",
    "Septoria_leaf_spot": "Remove affected leaves. Apply copper fungicide. Avoid overhead watering.",
    "Spider_mites Two-spotted_spider_mite": "Spray neem oil. Introduce predatory mites. Maintain adequate humidity.",
    "Target_Spot": "Use azoxystrobin fungicide. Remove crop debris. Avoid overhead watering.",
    "Tomato_Yellow_Leaf_Curl_Virus": "Control whiteflies using insecticides. Remove infected plants immediately.",
    "Tomato_mosaic_virus": "Use virus-free seeds. Disinfect tools with 10% bleach solution.",
    "healthy": "No disease detected. Maintain good agricultural practices and regular monitoring.",
    "powdery_mildew": "Apply sulfur or potassium bicarbonate. Improve air circulation."
}

disease_solutions_ur = {
    "Bacterial_spot": "متاثرہ پتے ہٹائیں۔ تانبے پر مبنی فنگسائیڈ استعمال کریں۔",
    "Early_blight": "کلوروتھالونل استعمال کریں۔ پانی کی چھڑکاؤ سے بچنے کے لیے ملچ کریں۔",
    "Late_blight": "تانبے کی فنگسائیڈ لگائیں۔ متاثرہ پودوں کو تباہ کریں۔",
    "Leaf_Mold": "ہوا کی گردش بڑھائیں۔ نمی کم کریں۔",
    "Septoria_leaf_spot": "متاثرہ پتے ہٹائیں۔ تانبے کی فنگسائیڈ لگائیں۔",
    "Spider_mites Two-spotted_spider_mite": "نیم کا تیل اسپرے کریں۔ شکاری کیڑے متعارف کروائیں۔",
    "Target_Spot": "ایزوکسائسٹروبن استعمال کریں۔ فصل کی باقیات ہٹائیں۔",
    "Tomato_Yellow_Leaf_Curl_Virus": "سفید مکھیوں کو کنٹرول کریں۔ متاثرہ پودے ہٹائیں۔",
    "Tomato_mosaic_virus": "وائرس سے پاک بیج استعمال کریں۔ اوزار جراثیم کش کریں۔",
    "healthy": "کوئی بیماری نہیں پائی گئی۔ اچھی دیکھ بھال جاری رکھیں۔",
    "powdery_mildew": "سلفر یا پوٹاشیم بائی کاربونیٹ لگائیں۔"
}

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================

def predict_disease(image_array, language='en'):
    """
    Predict disease from image array
    
    Args:
        image_array: PIL Image object
        language: 'en' or 'ur'
    
    Returns:
        dict with disease, confidence, solution
    """
    try:
        # Resize image
        img = image_array.resize((224, 224))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Preprocess
        from tensorflow.keras.applications.efficientnet import preprocess_input
        img_array = preprocess_input(img_array)
        
        # Predict
        preds = model.predict(img_array, verbose=0)[0]
        idx = np.argmax(preds)
        disease = class_names[idx]
        confidence = float(preds[idx])
        
        # Get all confidences
        confidences = {class_names[i]: float(preds[i]) for i in range(len(class_names))}
        
        # Translate disease name
        if language == 'ur':
            disease_display = disease_translations.get(disease, disease)
            solution = disease_solutions_ur.get(disease, "ماہر سے مشورہ کریں۔")
        else:
            disease_display = disease
            solution = disease_solutions_en.get(disease, "Consult an expert.")
        
        return {
            'success': True,
            'disease': disease,
            'disease_display': disease_display,
            'confidence': confidence,
            'confidence_percent': f"{confidence * 100:.1f}%",
            'confidences': confidences,
            'solution': solution,
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    API endpoint for disease prediction
    
    Expects:
        - image: base64 encoded image
        - language: 'en' or 'ur'
    
    Returns:
        JSON with prediction results
    """
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({'success': False, 'error': 'No image provided'}), 400
        
        # Decode base64 image
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Ensure RGB format
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Get language
        language = data.get('language', 'en')
        
        # Predict
        result = predict_disease(image, language)
        
        return jsonify(result), 200
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model': 'EfficientNetB0',
        'diseases': len(class_names),
        'timestamp': datetime.now().isoformat()
    }), 200

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def server_error(error):
    return jsonify({'error': 'Server error'}), 500

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("TOMATO DISEASE DETECTOR - BACKEND SERVER")
    print("="*60)
    print(f"Model loaded with {len(class_names)} disease classes")
    print("Starting Flask server at http://localhost:5000")
    print("="*60 + "\n")
    
    # Run development server (use gunicorn for production)
    app.run(
        debug=False,
        host='0.0.0.0',
        port=5000,
        threaded=True
    )
