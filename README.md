# Khadrawi
AI-powered smart plant health monitoring system for disease detection and personalized care plans. 🌱

This Streamlit application uses a Convolutional Neural Network (CNN) to identify plant diseases from leaf images. It incorporates Grad-CAM for visual explanations and integrates Google Gemini for bilingual medical reports and expert chat assistance.

## Features

* **Disease Classification:** Identifies plant diseases using a PyTorch-based CNN.
* **Explainable AI (XAI):** Uses Grad-CAM to generate heatmaps showing which parts of the leaf influenced the model's decision.
* **Bilingual Support:** Full interface and AI reports available in both English and Arabic.
* **AI Expert Assistant:** Integrated Gemini chat for follow-up questions regarding treatment and prevention.
* **Genetic Insights:** Provides simulated genetic pattern analysis based on model confidence.

## Prerequisites

* Python 3.8 or higher
* A Google Gemini API Key

## Installation

1. Clone this repository or save the source code.
2. Install the required dependencies:

```bash
pip install streamlit torch torchvision numpy opencv-python Pillow matplotlib google-genai python-dotenv

```

## Configuration

1. Create a file named `.env` in the root directory.
2. Add your Gemini API key to the file:

```text
GEMINI_API_KEY=your_api_key_here

```

3. Ensure the following files are present in the directory:
* `best_crop_model.pth`: The trained PyTorch model weights.
* `class_names.pth`: A saved list of the disease class names.



## Usage

1. Run the Streamlit application:

```bash
streamlit run app.py

```

2. Upload an image of a plant leaf.
3. View the prediction, confidence levels, and Grad-CAM visualizations.
4. Read the AI-generated report and use the chat interface to ask for specific advice.

## Project Structure

* `app.py`: The main application script.
* `GeneralizedPlantCNN`: The neural network architecture class.
* `GradCAM`: Class for generating activation heatmaps.
* `texts`: Dictionary containing all bilingual UI strings and prompts.
