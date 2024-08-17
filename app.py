# from flask import Flask, request, render_template, redirect, url_for
# import numpy as np
# import librosa
# import pickle
# import os

# app = Flask(__name__)

# # Load the trained model
# with open('dog_bark_classifier.pkl', 'rb') as model_file:
#     loaded_model = pickle.load(model_file)

# # Function to extract features from an audio file
# def extract_features(file_path):
#     y, sr = librosa.load(file_path, sr=22050)  # Load audio
#     mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
#     chroma = librosa.feature.chroma_stft(y=y, sr=sr)
#     spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
#     return np.concatenate((np.mean(mfccs.T, axis=0), 
#                            np.mean(chroma.T, axis=0),
#                            np.mean(spectral_contrast.T, axis=0)))

# # Route for the home page
# @app.route('/')
# def index():
#     return render_template('index.html')

# # Route to handle file upload and prediction
# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return redirect(url_for('index'))

#     file = request.files['file']
#     if file.filename == '':
#         return redirect(url_for('index'))
    
#     # Save the uploaded file
#     file_path = os.path.join('uploads', file.filename)
#     file.save(file_path)
    
#     # Extract features and predict
#     feature = extract_features(file_path).reshape(1, -1)
#     prediction = loaded_model.predict(feature)
    
#     # Remove the uploaded file
#     os.remove(file_path)
    
#     return render_template('result.html', prediction=prediction[0])

# if __name__ == '__main__':
#     os.makedirs('uploads', exist_ok=True)  # Create uploads directory if not exists
#     app.run(debug=True)














# from flask import Flask, request, jsonify
# import numpy as np
# import librosa
# import pickle
# import os

# app = Flask(__name__)

# # Load the trained model when the app starts
# with open('dog_bark_classifier.pkl', 'rb') as model_file:
#     loaded_model = pickle.load(model_file)

# # Function to extract features from an audio file
# def extract_features(file_path):
#     y, sr = librosa.load(file_path, sr=22050)  # Load audio
#     mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
#     chroma = librosa.feature.chroma_stft(y=y, sr=sr)
#     spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
#     return np.concatenate((np.mean(mfccs.T, axis=0), 
#                            np.mean(chroma.T, axis=0),
#                            np.mean(spectral_contrast.T, axis=0)))

# # Route to handle file upload and prediction
# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file provided'}), 400

#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({'error': 'Empty file'}), 400
    
#     # Save the uploaded file
#     file_path = os.path.join('uploads', file.filename)
#     file.save(file_path)
    
#     # Extract features and predict
#     feature = extract_features(file_path).reshape(1, -1)
#     prediction = loaded_model.predict(feature)
    
#     # Remove the uploaded file
#     os.remove(file_path)
    
#     return jsonify({'prediction': prediction[0]})

# if __name__ == '__main__':
#     os.makedirs('uploads', exist_ok=True)  # Create uploads directory if not exists
#     app.run(debug=True)






























from flask import Flask, request, render_template, redirect, url_for, jsonify
import numpy as np
import librosa
import pickle
import os

app = Flask(__name__)

# Load the trained model
with open('dog_bark_classifier.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# Function to extract features from an audio file
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=22050)  # Load audio
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    return np.concatenate((np.mean(mfccs.T, axis=0), 
                           np.mean(chroma.T, axis=0),
                           np.mean(spectral_contrast.T, axis=0)))

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle file upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))
    
    # Save the uploaded file
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)
    
    # Extract features and predict
    feature = extract_features(file_path).reshape(1, -1)
    prediction = loaded_model.predict(feature)
    
    # Remove the uploaded file
    os.remove(file_path)
    
    return render_template('result.html', prediction=prediction[0])

# API route to handle file upload and prediction
@app.route('/api/predict', methods=['POST'])
def api_predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    
    # Save the uploaded file
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)
    
    # Extract features and predict
    feature = extract_features(file_path).reshape(1, -1)
    prediction = loaded_model.predict(feature)
    
    # Remove the uploaded file
    os.remove(file_path)
    
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)  # Create uploads directory if not exists
    app.run(debug=True)

