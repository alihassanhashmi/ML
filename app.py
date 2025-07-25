from flask import Flask, render_template, request
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
import cv2
import os
import numpy as np
from PIL import Image
import uuid
import pickle
from moviepy import VideoFileClip
import whisper
import joblib
from werkzeug.utils import secure_filename 
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

model = torchvision.models.resnet50(pretrained=False)
model.fc = nn.Linear(2048, 2)
model.load_state_dict(torch.load("models/smoking_alcohol_model3.pt", map_location=torch.device('cpu'))) #download model from link txt file and then upload in models folder and then u good to go
model.eval()

with open("models/svm_abuse_model.pkl", "rb") as f: #less accurate model trained by me ignore it and use the abuse_language moel trained on bert with high accuracy
    svm_model = joblib.load(f)
with open("models/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = joblib.load(f)

whisper_model = whisper.load_model("base")

model_dir = "https://drive.google.com/drive/folders/1Gw7YI-H-vnA_Y03WP7eM0hThOSORqoyX?usp=sharing" # Replace with the actual path to your model directory like models/abuse_language

tokenizer = AutoTokenizer.from_pretrained(model_dir)
abuse_model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)

def predict_abuse1(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = abuse_model(**inputs).logits
        probs = torch.softmax(logits, dim=1)
        label = torch.argmax(probs).item()
    return "Abusive" if label == 1 else "Not Abusive"

def extract_audio(video_path, audio_path='temp_audio.wav'):
    video = VideoFileClip(video_path)
    if video.audio is None:
        return None
    video.audio.write_audiofile(audio_path, logger=None)
    return audio_path

def transcribe_audio(audio_path):
    result = whisper_model.transcribe(audio_path)
    return result['text']

def predict_abuse(text, model, vectorizer):
    text_features = vectorizer.transform([text])
    label = model.predict(text_features)[0]
    if label == 0:
        return "Hate Speech"
    elif label == 1:
        return "Offensive Language"
    else:
        return "Neutral"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict_image(image_path, threshold=0.5):
    img = Image.open(image_path).convert('RGB')
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(img)
        preds = torch.sigmoid(output)
    smoking_prob = preds[0][0].item()
    alcohol_prob = preds[0][1].item()
    return smoking_prob > threshold, alcohol_prob > threshold, smoking_prob, alcohol_prob

def predict_video(video_path, threshold=0.5):
    cap = cv2.VideoCapture(video_path)
    smoking_preds = []
    alcohol_preds = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = transform(img).unsqueeze(0)
        with torch.no_grad():
            output = model(img)
            preds = torch.sigmoid(output)
        smoking_preds.append(preds[0][0].item())
        alcohol_preds.append(preds[0][1].item())
    cap.release()
    smoking_mean = np.mean(smoking_preds)
    alcohol_mean = np.mean(alcohol_preds)
    return smoking_mean > threshold, alcohol_mean > threshold, smoking_mean, alcohol_mean

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        file = request.files['file']
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        ext = filename.split('.')[-1].lower()
        if ext in ['mp4', 'avi', 'mov', 'mkv']:
            smoking_detected, alcohol_detected, smoking_prob, alcohol_prob = predict_video(file_path)
            try:
                audio_path = extract_audio(file_path)
            except Exception as e:
                audio_path = None
            if audio_path is not None:
                transcribed_text = transcribe_audio(audio_path)
                abuse_label = predict_abuse1(transcribed_text)
            else:
                transcribed_text = None
                abuse_label = None
            file_type = "video"
        else:
            smoking_detected, alcohol_detected, smoking_prob, alcohol_prob = predict_image(file_path)
            transcribed_text = None
            abuse_label = None
            file_type = "image"
        is_neutral = not smoking_detected and not alcohol_detected
        result = {
            'filename': filename,
            'file_type': file_type,
            'smoking': smoking_detected,
            'alcohol': alcohol_detected,
            'smoking_prob': f"{smoking_prob*100:.2f}%",
            'alcohol_prob': f"{alcohol_prob*100:.2f}%",
            'neutral': is_neutral,
            'abuse_label': abuse_label,
            'transcribed_text': transcribed_text,
            'file_url': file_path
        }
    return render_template('index.html', result=result)

if __name__ == "__main__":
    app.run(debug=True)
