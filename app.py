import os
import pickle
import librosa
import soundfile as sf
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

with open("emotion_model.pkl", "rb") as f:
    model = pickle.load(f)

def get_agent_response(emotion):
    responses = {
        'neutral': "You seem perfectly calm and steady. Let me know how I can assist you today!",
        'calm': "Your voice sounds serene and peaceful. It's a wonderful state of mind!",
        'happy': "Wow, I detect so much positive energy in your voice! Keep spreading that joy! 🎉",
        'sad': "You sound a bit down. Take it easy, drink some water, and remember that tough times don't last.",
        'angry': "I sense a bit of frustration or tension. Let's take a deep breath. How can I help resolve this?",
        'fearful': "Your voice sounds slightly anxious or worried. Don't worry, you are safe here.",
        'disgust': "I notice some disapproval or dislike in your tone. Let me know what went wrong.",
        'surprised': "Oh, you sound amazed or startled! Did something unexpected happen? 😮"
    }
    return responses.get(emotion, "I processed your audio, but couldn't quite map the exact feeling.")

# ⭐ OPTIMIZED MEMORY-EFFICIENT FEATURE EXTRACTION
def extract_feature(file_path):
    # Read with soundfile natively using a targeted sample rate limit
    X, sample_rate = sf.read(file_path, dtype="float32")
    
    if len(X.shape) > 1:
        X = np.mean(X, axis=1) # Stereo to Mono
        
    # Downsample if the file rate is unnecessarily high (e.g. 44.1kHz down to 16kHz) 
    # This prevents RAM spikes on the free tier hosting environment
    if sample_rate > 16000:
        X = librosa.resample(X, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000

    result = np.array([])
    
    # 1. MFCC
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    result = np.hstack((result, mfccs))
    
    # 2. Chroma (Optimized STFT hop length to conserve memory)
    stft = np.abs(librosa.stft(X, hop_length=512))
    chroma_feat = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_feat))
    
    # 3. Mel Spectrogram
    mel_feat = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel_feat))
    
    return result.reshape(1, -1)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio_file' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
        
    file = request.files['audio_file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        
        try:
            features = extract_feature(file_path)
            prediction = model.predict(features)[0]
            agent_text = get_agent_response(prediction)
            
            os.remove(file_path)
            return jsonify({
                'status': 'success',
                'emotion': prediction.capitalize(),
                'response': agent_text
            })
        except Exception as e:
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
