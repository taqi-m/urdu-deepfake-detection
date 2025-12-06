# Urdu Deepfake Audio Detection

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

A machine learning application for detecting deepfake audio in Urdu using four trained models: SVM, Logistic Regression, Perceptron, and Deep Neural Network.

## 🚀 Live Demo

Try the live application: **[Launch App](https://your-app-url.streamlit.app)**

## 🎯 Features

- **Multiple ML Models**: Choose from 4 different trained models
- **Real-time Detection**: Upload and analyze audio files instantly
- **Visual Analysis**: Interactive waveform and spectrogram visualization
- **Confidence Scoring**: Get prediction confidence levels
- **User-friendly Interface**: Clean, intuitive design

## 📋 Prerequisites

- Python 3.10
- Audio files in WAV, MP3, M4A, or OGG format

## 🚀 Quick Start

### Local Installation

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/urdu-deepfake-detection.git
cd urdu-deepfake-detection
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run the application**:
```bash
streamlit run app_deploy.py
```

4. **Open your browser** to `http://localhost:8501`

## 🖥️ Using the App

1. **Select a Model**: Choose from SVM, Logistic Regression, Perceptron, or DNN
2. **Upload Audio**: Upload an audio file (WAV, MP3, M4A, or OGG)
3. **Analyze**: Click the "Analyze Audio" button
4. **View Results**: Get prediction with confidence score and audio visualizations

## 🎵 Supported Audio Formats

- WAV (recommended for best accuracy)
- MP3
- M4A
- OGG

## 📈 Available Models

| Model | Description | Best For |
|-------|-------------|----------|
| **SVM** | Support Vector Machine with RBF kernel | High accuracy, excellent generalization |
| **Logistic Regression** | Linear probabilistic classifier | Fast predictions, interpretable |
| **Perceptron** | Single-layer neural network | Baseline performance, fast training |
| **DNN** | Deep neural network (4 hidden layers) | Complex patterns, highest accuracy |

## 🔬 Technology Stack

- **Python 3.10**
- **Streamlit** - Web application framework
- **TensorFlow** - Deep learning models
- **Scikit-learn** - Machine learning algorithms
- **Librosa** - Audio feature extraction
- **Plotly** - Interactive visualizations

## 📊 Audio Features

The system extracts **310 audio features**:
- MFCC (26 features)
- Mel Spectrogram (256 features)
- Chroma (24 features)
- Zero Crossing Rate (2 features)
- Spectral Centroid (2 features)

## 🗂️ Project Structure

```
urdu-deepfake-detection/
├── app_deploy.py                  # Main Streamlit application
├── requirements.txt               # Python dependencies
├── packages.txt                   # System dependencies
├── README.md                      # This file
├── .streamlit/
│   └── config.toml               # Streamlit theme configuration
├── Models/
│   ├── svm_model.pkl
│   ├── logistic_regression_model.pkl
│   ├── perceptron_model.pkl
│   └── dnn_model.keras
└── Configs/
    ├── scaler.pkl
    ├── preprocessor_config.json
    └── label_mapping.json
```

## 🌐 Deployment

This app is deployed on Streamlit Community Cloud. To deploy your own:

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Select `app_deploy.py` as the main file
5. Click Deploy!

## 🛠️ Troubleshooting

### Audio file not processing
- Use WAV format for best results
- Ensure audio is at least 1 second long
- Check that the file is not corrupted

### App running slow
- Processing time depends on audio length
- Larger files take longer to analyze
- Consider using shorter audio clips

## 📝 Usage Tips

- **Audio Quality**: Use clear speech audio for best results
- **Duration**: Minimum 1 second, recommended 3-5 seconds
- **Format**: WAV provides the most accurate predictions
- **Background Noise**: May affect accuracy

## ⚠️ Limitations

- Trained specifically on Urdu language audio
- Performance may vary with heavy background noise
- Not suitable for real-time streaming analysis

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/yourusername/urdu-deepfake-detection/issues).

## 📄 License

This project is open source and available for educational and research purposes.

## 🙏 Acknowledgments

- Urdu Deepfake Detection Dataset
- Streamlit Community
- TensorFlow and Scikit-learn teams
- Librosa audio processing library

---

**Made with ❤️ for deepfake detection research**

For questions or support, please [open an issue](https://github.com/yourusername/urdu-deepfake-detection/issues).
