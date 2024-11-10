# Video Recognition System

A powerful video recognition system that uses LLaMA Vision model to analyze video content and detect specific objects or scenes.

## Features

- 🎥 Video frame extraction and analysis
- 🧠 Powered by LLaMA 3.2 Vision model
- 🚀 GPU acceleration support
- 🌐 Web-based interface
- 🔄 Real-time analysis feedback
- 📊 Visual results display

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- [Ollama](https://ollama.ai/) installed and running
- LLaMA 3.2 Vision model

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/video-recognition.git
cd video-recognition
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Install Ollama following the instructions at [ollama.ai](https://ollama.ai)

## Project Structure

```
video-recognition/
├── templates/
│   └── index.html
├── uploads/          # Directory for uploaded videos
├── frames/           # Directory for extracted frames
├── app.py           # FastAPI application code
├── main.py          # Main application logic
├── requirements.txt  # Project dependencies
└── README.md        # Project documentation
```

## Usage

1. Start the application:
```bash
python main.py
```

2. Open your web browser and navigate to `http://localhost:8000`

3. Upload a video and specify the object or scene you want to find

4. Wait for the analysis results - the system will process the video and show matching frames

## Configuration

The system automatically detects and uses GPU if available. You can modify the following parameters in `main.py`:

- `MODEL_NAME`: The vision model to use (default: 'llama3.2-vision')
- `UPLOAD_DIR`: Directory for uploaded videos
- `FRAMES_DIR`: Directory for extracted frames

## Technical Details

- Backend: FastAPI
- Video Processing: OpenCV
- AI Model: LLaMA 3.2 Vision
- GPU Acceleration: CUDA via PyTorch

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- LLaMA Vision model by Meta
- Ollama for model serving
- FastAPI framework
- OpenCV for video processing
