import streamlit as st
import numpy as np
import librosa
import joblib
import json
from tensorflow import keras
from keras.models import load_model
import pandas as pd
import plotly.graph_objects as go
from io import BytesIO
import os
import matplotlib.pyplot as plt
import librosa.display

# Page configuration
st.set_page_config(
    page_title="Urdu Deepfake Audio Detection",
    page_icon="🎵",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4ECDC4;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .result-box {
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        margin: 2rem 0;
    }
    .bonafide {
        background-color: #d4edda;
        color: #155724;
        border: 3px solid #28a745;
    }
    .spoofed {
        background-color: #f8d7da;
        color: #721c24;
        border: 3px solid #dc3545;
    }
    .model-card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px solid #e0e0e0;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s ease;
        background-color: white;
        margin: 0.5rem;
    }
    .model-card:hover {
        border-color: #4ECDC4;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }
    .model-card.selected {
        border-color: #4ECDC4;
        background-color: #e8f8f7;
        box-shadow: 0 4px 8px rgba(78,205,196,0.3);
    }
</style>
""", unsafe_allow_html=True)

# AudioPreprocessor class
class AudioPreprocessor:
    """Preprocessing pipeline for audio feature extraction"""

    def __init__(self, sr=16000, duration=4.0, n_mfcc=13, n_mel=128, n_chroma=12):
        self.sr = sr
        self.duration = duration
        self.n_mfcc = n_mfcc
        self.n_mel = n_mel
        self.n_chroma = n_chroma
        self.max_samples = int(sr * duration)

    def pad_or_truncate(self, audio):
        if len(audio) < self.max_samples:
            audio = np.pad(audio, (0, self.max_samples - len(audio)), mode='constant')
        else:
            audio = audio[:self.max_samples]
        return audio

    def extract_mfcc(self, audio):
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=self.n_mfcc)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        return np.concatenate([mfcc_mean, mfcc_std])

    def extract_mel_spectrogram(self, audio):
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=self.sr, n_mels=self.n_mel)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_mean = np.mean(mel_spec_db, axis=1)
        mel_std = np.std(mel_spec_db, axis=1)
        return np.concatenate([mel_mean, mel_std])

    def extract_chroma(self, audio):
        chroma = librosa.feature.chroma_stft(y=audio, sr=self.sr, n_chroma=self.n_chroma)
        chroma_mean = np.mean(chroma, axis=1)
        chroma_std = np.std(chroma, axis=1)
        return np.concatenate([chroma_mean, chroma_std])

    def extract_zero_crossing_rate(self, audio):
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        return np.array([np.mean(zcr), np.std(zcr)])

    def extract_spectral_centroid(self, audio):
        spec_cent = librosa.feature.spectral_centroid(y=audio, sr=self.sr)[0]
        return np.array([np.mean(spec_cent), np.std(spec_cent)])

    def extract_features(self, audio):
        audio = self.pad_or_truncate(audio)
        mfcc_feat = self.extract_mfcc(audio)
        mel_feat = self.extract_mel_spectrogram(audio)
        chroma_feat = self.extract_chroma(audio)
        zcr_feat = self.extract_zero_crossing_rate(audio)
        spec_cent_feat = self.extract_spectral_centroid(audio)
        
        features = np.concatenate([
            mfcc_feat,
            mel_feat,
            chroma_feat,
            zcr_feat,
            spec_cent_feat
        ])
        return features

@st.cache_resource
def load_models_and_data():
    """Load all models and configuration files"""
    try:
        # Load models
        svm_model = joblib.load('svm_model.pkl')
        lr_model = joblib.load('logistic_regression_model.pkl')
        perceptron_model = joblib.load('perceptron_model.pkl')
        dnn_model = load_model('dnn_model.keras')
        
        # Load scaler
        scaler = joblib.load('scaler.pkl')
        
        # Load configurations
        with open('preprocessor_config.json', 'r') as f:
            preprocessor_config = json.load(f)
        
        with open('label_mapping.json', 'r') as f:
            label_mapping = json.load(f)
        
        # Initialize preprocessor (exclude max_samples as it's calculated, not a parameter)
        preprocessor_params = {k: v for k, v in preprocessor_config.items() if k != 'max_samples'}
        preprocessor = AudioPreprocessor(**preprocessor_params)
        
        models = {
            'SVM': svm_model,
            'Logistic Regression': lr_model,
            'Single-Layer Perceptron': perceptron_model,
            'Deep Neural Network': dnn_model
        }
        
        return models, scaler, preprocessor, label_mapping
    
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.info("Please ensure all model files are in the same directory as this app.")
        return None, None, None, None

def predict_audio(audio_data, sr, model_name, models, scaler, preprocessor):
    """Make prediction on audio data"""
    try:
        # Resample if needed
        if sr != preprocessor.sr:
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=preprocessor.sr)
        
        # Extract features
        features = preprocessor.extract_features(audio_data)
        features = features.reshape(1, -1)
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Get model
        model = models[model_name]
        
        # Make prediction
        if model_name == 'Deep Neural Network':
            prediction_proba = model.predict(features_scaled, verbose=0)[0][0]
            prediction = 1 if prediction_proba >= 0.5 else 0
        else:
            prediction = model.predict(features_scaled)[0]
            if hasattr(model, 'predict_proba'):
                prediction_proba = model.predict_proba(features_scaled)[0][1]
            else:
                # For perceptron, use decision function
                decision = model.decision_function(features_scaled)[0]
                prediction_proba = 1 / (1 + np.exp(-decision))  # Sigmoid
        
        return prediction, prediction_proba
    
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return None, None

def plot_waveform(audio_data, sr):
    """Plot audio waveform as static image"""
    fig, ax = plt.subplots(figsize=(10, 4))
    time = np.arange(0, len(audio_data)) / sr
    ax.plot(time, audio_data, color='#4ECDC4', linewidth=0.5)
    ax.set_title('Audio Waveform', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (seconds)', fontsize=11)
    ax.set_ylabel('Amplitude', fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def plot_spectrogram(audio_data, sr):
    """Plot mel spectrogram as static image"""
    mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    img = librosa.display.specshow(mel_spec_db, x_axis='time', y_axis='mel', 
                                    sr=sr, cmap='viridis', ax=ax)
    ax.set_title('Mel Spectrogram', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (seconds)', fontsize=11)
    ax.set_ylabel('Mel Frequency (Hz)', fontsize=11)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    plt.tight_layout()
    return fig

# Main app
def main():
    # Header
    st.markdown('<div class="main-header">🎵 Urdu Deepfake Audio Detection</div>', unsafe_allow_html=True)
    st.markdown('---')
    
    # Load models and data
    models, scaler, preprocessor, label_mapping = load_models_and_data()
    
    if models is None:
        st.stop()
    
    # Initialize session state
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'current_file' not in st.session_state:
        st.session_state.current_file = None
    
    # Main content
    st.markdown('<div class="sub-header">Upload Audio File for Detection</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['wav', 'mp3', 'm4a', 'ogg'],
        help="Upload an audio file to detect if it's real or deepfake",
        key='audio_uploader'
    )
    
    # Model selection
    st.markdown("### Select Detection Model")
    
    # Initialize session state for model selection if not exists
    model_options = list(models.keys())
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = model_options[0]
    
    # Create 4 columns for model buttons
    cols = st.columns(4)
    
    # Create buttons for each model
    for col, model in zip(cols, model_options):
        with col:
            is_selected = st.session_state.selected_model == model
            button_type = "primary" if is_selected else "secondary"
            
            if st.button(model, key=f"model_{model}", use_container_width=True, type=button_type):
                st.session_state.selected_model = model
                st.rerun()
    
    model_name = st.session_state.selected_model
    
    st.markdown("---")
    
    if uploaded_file is not None:
        # Check if this is a new file
        file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state.current_file != file_id:
            st.session_state.current_file = file_id
            st.session_state.analysis_results = None
        
        # Load audio
        try:
            # Save uploaded file temporarily for librosa to read
            temp_file_path = f"temp_audio_{uploaded_file.name}"
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Load audio from temporary file
            audio_data, sr = librosa.load(temp_file_path, sr=None)
            
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            
            # Display audio player
            uploaded_file.seek(0)  # Reset file pointer
            st.audio(uploaded_file, format='audio/wav')
            
            # Audio info
            duration = len(audio_data) / sr
            st.info(f"📝 **Audio Info**: Duration: {duration:.2f}s | Sample Rate: {sr} Hz | Samples: {len(audio_data)}")
            
            # Predict button
            if st.button("🔍 Analyze Audio", type="primary", use_container_width=True):
                with st.spinner(f'Analyzing with {model_name}...'):
                    prediction, confidence = predict_audio(
                        audio_data, sr, model_name, models, scaler, preprocessor
                    )
                    
                    if prediction is not None:
                        # Store results in session state
                        st.session_state.analysis_results = {
                            'prediction': prediction,
                            'confidence': confidence,
                            'model_name': model_name,
                            'audio_data': audio_data,
                            'sr': sr
                        }
            
            # Display results if available
            if st.session_state.analysis_results is not None:
                with st.spinner('Loading analysis results...'):
                    results = st.session_state.analysis_results
                    prediction = results['prediction']
                    confidence = results['confidence']
                    result_model_name = results['model_name']
                    result_audio_data = results['audio_data']
                    result_sr = results['sr']
                    
                    # Display result
                    label_text = label_mapping[str(prediction)]
                    
                    # Calculate actual confidence (for bonafide, invert the probability)
                    if prediction == 0:
                        result_class = "bonafide"
                        icon = "✅"
                        color = "#28a745"
                        actual_confidence = 1 - confidence  # Invert for bonafide
                        confidence_label = "Bonafide Confidence"
                    else:
                        result_class = "spoofed"
                        icon = "⚠️"
                        color = "#dc3545"
                        actual_confidence = confidence
                        confidence_label = "Deepfake Confidence"
                    
                    st.markdown(f'<div class="result-box {result_class}">{icon} {label_text}</div>', unsafe_allow_html=True)
                    
                    # Confidence score
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Prediction", "Bonafide" if prediction == 0 else "Spoofed")
                    with col2:
                        st.metric(confidence_label, f"{actual_confidence:.2%}")
                    with col3:
                        st.metric("Model Used", result_model_name)
                    
                    # Confidence gauge
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=actual_confidence * 100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': f"{confidence_label} (%)"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': color},
                            'steps': [
                                {'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 75], 'color': "gray"},
                                {'range': [75, 100], 'color': "darkgray"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 90
                            }
                        }
                    ))
                    fig_gauge.update_layout(height=300)
                    st.plotly_chart(fig_gauge, use_container_width=True)
                    
                    # Audio visualizations
                    st.markdown('<div class="sub-header">Audio Analysis</div>', unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        fig_wave = plot_waveform(result_audio_data, result_sr)
                        st.pyplot(fig_wave, use_container_width=True)
                        plt.close(fig_wave)
                    
                    with col2:
                        fig_spec = plot_spectrogram(result_audio_data, result_sr)
                        st.pyplot(fig_spec, use_container_width=True)
                        plt.close(fig_spec)
            
        except Exception as e:
            st.error(f"Error loading audio file: {str(e)}")
    
    else:
        # Instructions when no file is uploaded
        st.info("👆 Please upload an audio file to begin detection")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### How to use:")
            st.markdown("""
            1. **Upload an audio file** using the file uploader above
            2. **Select a model** for detection
            3. **Click 'Analyze Audio'** to get the detection result
            4. **View the results** including confidence score and visualizations
            """)
        
        with col2:
            st.markdown("### Tips for best results:")
            st.markdown("""
            - Use high-quality audio files
            - WAV format provides the most accurate predictions
            - Audio should be at least 1 second long
            - Clear speech without background noise works best
            """)
            
            st.markdown("### Supported Formats:")
            st.markdown("""
            - WAV (recommended)
            - MP3
            - M4A
            - OGG
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Urdu Deepfake Audio Detection System | Built with Streamlit & TensorFlow</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
