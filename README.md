# Plant Disease Detection using YOLOv8

A web-based application for detecting plant diseases using YOLOv8 deep learning model. The application provides real-time disease detection with AI-powered treatment recommendations via OpenAI's GPT-3.5.

## 🌟 Features

- **Real-time Disease Detection**: Upload images or capture photos to detect plant diseases
- **YOLOv8 Integration**: Uses state-of-the-art YOLOv8 model for accurate disease classification
- **AI-Powered Solutions**: Get treatment recommendations using OpenAI GPT-3.5
- **Multi-language Support**: Available in English and Hindi
- **Web Camera Support**: Capture images directly from your device's camera
- **Responsive Design**: Works on desktop and mobile devices
- **Visual Detection**: Bounding boxes and confidence scores for detected diseases

## 🔬 Supported Disease Detection

The model can detect various plant diseases including:
- Powdery Mildew
- Colorado Potato Beetle Larva
- Milli Bugs
- Dried and Wilted Leaves
- Chili Plant diseases
- Healthy plant detection

## 🛠️ Technology Stack

- **Backend**: Flask (Python)
- **AI/ML**: YOLOv8 (Ultralytics), OpenCV
- **AI Assistant**: OpenAI GPT-3.5 Turbo
- **Frontend**: HTML, CSS, JavaScript
- **Image Processing**: PIL, OpenCV
- **File Handling**: Base64 encoding for camera captures

## 📋 Prerequisites

Before setting up the project, ensure you have:

- Python 3.8 or higher
- OpenAI API key (for treatment recommendations)
- Git (for cloning the repository)
- A webcam or camera-enabled device (optional, for live capture)

## 🚀 Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/maazajaz/CROP-DISEASE.git
cd CROP-DISEASE
```

### 2. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

Install the required packages:

```bash
pip install Flask==2.3.3
pip install ultralytics==8.0.196
pip install opencv-python==4.8.0.76
pip install Pillow==10.0.0
pip install openai==0.28.1
pip install numpy==1.24.3
```

Or create a `requirements.txt` file with the dependencies and install:

```bash
pip install -r requirements.txt
```

### 4. Set Up OpenAI API Key

1. Get your OpenAI API key from [OpenAI Platform](https://platform.openai.com/api-keys)
2. Set it as an environment variable:

**Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY="your_openai_api_key_here"
```

**Windows (Command Prompt):**
```cmd
set OPENAI_API_KEY=your_openai_api_key_here
```

**macOS/Linux:**
```bash
export OPENAI_API_KEY="your_openai_api_key_here"
```

### 5. Create Required Directories

The application will automatically create these directories, but you can create them manually:

```bash
mkdir static/uploaded_images
mkdir static/output_images
mkdir uploads
```

## 🎯 Usage

### 1. Start the Application

```bash
python app.py
```

The application will start on `http://localhost:5000`

### 2. Using the Web Interface

1. **Open your browser** and navigate to `http://localhost:5000`
2. **Upload an image** of a plant leaf or **capture a photo** using your camera
3. **Select language** (English or Hindi) for treatment recommendations
4. **Click "Detect Disease"** to process the image
5. **View results** including:
   - Disease name and confidence score
   - Processed image with bounding boxes
   - AI-generated treatment recommendations

## 📁 Project Structure

```
CROP-DISEASE/
│
├── app.py                          # Main Flask application
├── best.pt                         # YOLOv8 trained model (22MB)
├── custom_results.json            # Predefined disease mappings
├── README.md                      # Project documentation
│
├── static/
│   ├── uploaded_images/           # User uploaded images
│   └── output_images/             # Processed images with detections
│
├── templates/
│   ├── index.html                 # Main upload page
│   └── result.html                # Results display page
│
├── uploads/                       # Additional upload directory
│
└── sample_images/
    ├── leaf1.jpg                  # Sample test images
    └── leaf2.jpg
```

## ⚙️ Configuration

### Model Configuration

- **Model**: YOLOv8 (`best.pt`)
- **Image Size**: 416x416 pixels
- **Batch Size**: 16
- **Confidence Threshold**: Automatically determined by model

### Application Configuration

- **Host**: `0.0.0.0` (accessible from network)
- **Port**: `5000`
- **Debug Mode**: `True` (disable in production)
- **Allowed File Types**: PNG, JPG, JPEG

## 🐛 Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Install missing dependencies using pip
2. **OpenAI API Error**: Check your API key and internet connection
3. **Model Loading Error**: Ensure `best.pt` is in the correct location
4. **Camera Access**: Allow camera permissions in your browser
5. **File Upload Error**: Check file format (PNG, JPG, JPEG only)

## 🚀 Deployment

### Local Network Access

The app runs on `0.0.0.0:5000`, making it accessible from other devices on your network at:
```
http://YOUR_LOCAL_IP:5000
```

### Production Deployment

For production deployment:

1. **Disable debug mode**:
   ```python
   app.run(debug=False, host='0.0.0.0', port=5000)
   ```

2. **Use a production WSGI server** like Gunicorn:
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```

3. **Set environment variables** properly
4. **Use HTTPS** in production
5. **Add error handling** and logging

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and commit: `git commit -m 'Add feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 👨‍💻 Author

**Maaz Ajaz**
- GitHub: [@maazajaz](https://github.com/maazajaz)

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for the detection model
- [OpenAI](https://openai.com/) for AI-powered treatment recommendations
- [Flask](https://flask.palletsprojects.com/) for the web framework

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/maazajaz/CROP-DISEASE/issues) page
2. Create a new issue with detailed description
3. Include error messages and system information

---

**Happy Disease Detection! 🌱**