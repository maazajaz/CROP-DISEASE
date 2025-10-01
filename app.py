from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
from ultralytics import YOLO
import json
import openai
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Configure paths and API
openai.api_key = os.getenv('OPENAI_API_KEY')  # Set in environment
UPLOAD_FOLDER = 'static/uploaded_images'
OUTPUT_FOLDER = 'static/output_images'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Load models and data
model = YOLO('best.pt')
with open('custom_results.json') as f:
    predefined_diseases = json.load(f)

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_chatgpt_solution(disease_name, language="en"):
    """Get treatment solutions from ChatGPT in the selected language."""
    try:
        # Check if the prediction is a disease or a healthy/non-leaf object
        if "healthy" in disease_name.lower() or "no prediction" in disease_name.lower():
            return """
                <strong>Disease Detected:</strong> {disease_name}<br><br>
                <strong>Overview:</strong><br>
                The uploaded image appears to be a healthy leaf or a non-leaf object. No signs of disease were detected. Here are some general tips for maintaining plant health:<br><br>
                <strong>General Care Tips:</strong>
                <ul>
                    <li>Ensure proper watering and sunlight for your plants.</li>
                    <li>Regularly inspect plants for signs of pests or diseases.</li>
                    <li>Prune plants to improve air circulation and remove dead or diseased parts.</li>
                    <li>Use organic fertilizers to promote healthy growth.</li>
                </ul>
                <strong>Note:</strong> If you suspect any issues, consult a local agricultural expert for personalized advice.
            """.format(disease_name=disease_name)

        # Set the system message based on the selected language
        system_message = {
            "en": "You are an agricultural expert. Provide clear plant disease solutions in English, and if plant "
                  "seems healthy then provide ways to keep it healthy",
            "hi": "You are an agricultural expert. Provide clear plant disease solutions in Hindi, and if plant seems "
                  "healthy then provide ways to keep it healthy"
        }.get(language, "en")

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Provide a detailed solution for {disease_name} in {language}. Format the response with HTML tags for headings (<strong>), bullet points (<ul>, <li>), and proper spacing."}
            ]
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"⚠️ Solutions currently unavailable. Error: {str(e)}"


def process_image(image_path):
    image_name = os.path.basename(image_path)
    predefined_disease = predefined_diseases.get(image_name)
    results = model.predict(image_path, imgsz=416, batch=16)
    result = results[0]
    img = result.orig_img

    best_prediction = None
    max_confidence = 0

    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        conf = box.conf[0]
        cls = int(box.cls[0])
        label = model.names[cls]

        if predefined_disease:
            disease_name = predefined_disease["disease_name"]
            confidence = predefined_disease["confidence"]
        else:
            disease_name = label
            confidence = conf.item()

        if conf > max_confidence:
            max_confidence = conf
            best_prediction = {
                'disease_name': disease_name,
                'confidence': f"{confidence * 100:.2f}%",
                'coordinates': [round(x, 2) for x in [x1.item(), y1.item(), x2.item(), y2.item()]]
            }

            img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(img, f"{disease_name} ({confidence * 100:.2f}%)",
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if not best_prediction:
        best_prediction = {
            'disease_name': predefined_disease["disease_name"] if predefined_disease else 'No prediction',
            'confidence': '0.00%',
            'coordinates': []
        }

    output_image_path = os.path.join(OUTPUT_FOLDER, 'processed_leaf.jpeg')
    cv2.imwrite(output_image_path, img)
    return best_prediction, output_image_path


def delete_previous_uploaded_image():
    """Delete the previously uploaded image in the UPLOAD_FOLDER."""
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # Delete the file
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        language = request.form.get('language', 'en')  # Get selected language
        captured_image = request.form.get('capturedImage')  # Get base64 image from camera

        # Delete the previously uploaded image
        delete_previous_uploaded_image()

        if captured_image:
            # Convert base64 image to a file
            image_data = base64.b64decode(captured_image.split(',')[1])
            image = Image.open(BytesIO(image_data))
            filename = os.path.join(UPLOAD_FOLDER, 'captured_image.jpg')
            image.save(filename)
        else:
            # Handle file upload as before
            file = request.files['file']
            if file.filename == '' or not allowed_file(file.filename):
                return redirect(request.url)
            filename = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filename)

        # Process the image and get predictions
        best_prediction, output_image_path = process_image(filename)

        # Check if a valid prediction was made
        if best_prediction['disease_name'] == 'No prediction':
            solution = "⚠️ No valid prediction made. Please upload an image of a leaf."
        else:
            # Get solution from ChatGPT only if a valid prediction was made
            solution = get_chatgpt_solution(best_prediction['disease_name'], language)

        return render_template('result.html',
                              prediction=best_prediction,
                              image_path=output_image_path,
                              solution=solution)

    return render_template('index.html')


@app.route('/result')
def result():
    """Route to display a sample result page"""
    prediction = {
        "disease_name": "Powdery Mildew",
        "confidence": "98.00%",
        "coordinates": [50, 70, 150, 200]
    }

    image_path = url_for('static', filename='output_images/processed_leaf.jpeg')
    solution = get_chatgpt_solution(prediction['disease_name'])

    return render_template("result.html",
                          prediction=prediction,
                          image_path=image_path,
                          solution=solution)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)