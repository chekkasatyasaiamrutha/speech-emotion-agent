import os
import pickle
import librosa
import soundfile as sf
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained model
with open("emotion_model.pkl", "rb") as f:
    model = pickle.load(f)

# Helper function to generate agent responses based on emotion
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

def extract_feature(file_path):
    with sf.SoundFile(file_path) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if len(X.shape) > 1:
            X = np.mean(X, axis=1) # Convert Stereo to Mono
            
        result = np.array([])
        # MFCC
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))
        # Chroma
        stft = np.abs(librosa.stft(X))
        chroma_feat = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        result = np.hstack((result, chroma_feat))
        # Mel Spectrogram
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
            # Extract features and predict
            features = extract_feature(file_path)
            prediction = model.predict(features)[0]
            agent_text = get_agent_response(prediction)
            
            # Clean up the uploaded file after processing
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