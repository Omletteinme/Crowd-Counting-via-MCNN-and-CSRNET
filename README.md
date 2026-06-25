# CrowdSense: Crowd Counting & Density Estimation via MCNN and CSRNet

A clean, minimal, professional dark-themed web application that performs crowd density estimation and counting using deep learning. The application allows users to upload an image and choose between two distinct model architectures (**MCNN** and **CSRNet**) to view the estimated number of people, a generated density heatmap, and the inference time.

---

## 🚀 Features

- **Dual-Model Inference:** Select between:
  - **MCNN (Multi-Column CNN):** Utilizes multiple parallel convolutional branches with different kernel sizes to capture multi-scale crowd features.
  - **CSRNet (Congested Scene Recognition Network):** Uses a dilated convolutional neural network backbone (built on VGG-16 frontend) for high-resolution density mapping.
- **Dynamic Density Mapping:** Generates a custom jet-colormap density heatmap using Matplotlib.
- **Sleek UX/UI:** Dark mode interface with glassmorphism styling, drag-and-drop file upload, loading animations, and interactive metrics.
- **Production-Ready Docker Config:** Configured out-of-the-box to run locally or deploy to **Hugging Face Spaces** (exposes port `7860`).

---

## 🛠️ Tech Stack

- **Backend:** Flask (Python 3.11), PyTorch, Torchvision, OpenCV, NumPy, Matplotlib, SciPy
- **Frontend:** Vanilla HTML5, Vanilla CSS3 (with Outfit Font & Phosphor Icons), Vanilla JavaScript (ES6)
- **Deployment:** Docker, Gunicorn

---

## 📂 Project Structure

```
├── app.py                      # Flask backend & model inference logic
├── Dockerfile                  # Container configuration (HF Spaces optimized)
├── requirements.txt            # Python dependencies
├── templates/
│   └── index.html              # Frontend user interface
├── static/
│   ├── style.css               # Modern glassmorphism dark-theme styling
│   └── main.js                 # Drag & drop upload, API communications, counters
├── crowd_counting.pth          # Trained MCNN model weights
├── csrnet_best.pth             # Trained CSRNet model weights
└── README.md                   # Project documentation
```

---

## 💻 Local Setup

### 1. Manual Setup (Local Environment)
Make sure you have Python 3.11 installed.

```bash
# Clone the repository
git clone https://github.com/Omletteinme/Crowd-Counting-via-MCNN-and-CSRNET.git
cd Crowd-Counting-via-MCNN-and-CSRNET

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the Flask app
python app.py
```
Open `http://localhost:7860` in your web browser.

### 2. Docker Setup
```bash
# Build the Docker image
docker build -t crowdsense .

# Run the container
docker run -p 7860:7860 crowdsense
```
Open `http://localhost:7860` in your web browser.

---

## ☁️ Deployment

For deployment options (including a **100% free hosting guide** on **Hugging Face Spaces**), please refer to the detailed local [Deployment Guide](deployment_guide.md).

---

## 📜 License

This project is licensed under the MIT License.
