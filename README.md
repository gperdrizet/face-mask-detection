# Face Mask Detection

A deep learning project for detecting face masks in images using a PyTorch CNN model, with a simple Streamlit web application for deployment.

## Project structure

- `notebooks/` - Jupyter notebook for model training and evaluation
- `app/` - Streamlit web application for inference
- `data/` - Training data (with_mask / without_mask)
- `models/` - Trained model checkpoints

## Model

- **Architecture**: 4-layer CNN with batch normalization and dropout (~500K parameters)
- **Input size**: 128x128 pixels RGB
- **Output**: Binary classification (with_mask / without_mask)
- **Training**: PyTorch 2.0+ with data augmentation

## Getting started

### Using the dev container (recommended)

This project includes a dev container configuration with all dependencies pre-installed.

1. **Open in VS Code**: Make sure you have Docker and the Dev Containers extension installed
2. **Reopen in Container**: VS Code will prompt you to reopen in the container, or use Command Palette → "Dev Containers: Reopen in Container"
3. **Train the Model**: Open and run all cells in `notebooks/face_mask_detection.ipynb` to train the model and save it to `models/face_mask_detector_production.pth`
4. **Run the App**: From the `app/` directory, run:
```bash
streamlit run streamlit_app.py
```

The dev container has everything configured, so you don't need to install any dependencies manually!

### Manual installation (without dev container)

If you prefer not to use the dev container:

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Train the model by running the notebook at `notebooks/face_mask_detection.ipynb`

3. Run the app from the `app/` directory:
```bash
streamlit run streamlit_app.py
```

## Web application

A simple Streamlit app for real-time face mask detection.

### Features

- **Image Upload**: Upload JPG, JPEG, or PNG images
- **Webcam Capture**: Take photos directly with your camera
- **Real-time Inference**: Get instant mask detection results
- **Probability Display**: Shows mask detection probability (0.0 - 1.0)
- **CPU-only**: Runs on CPU, no GPU required

### Usage

The app will open in your default browser at `http://localhost:8501`.

1. **Upload an image**: Click "Browse files" to select an image from your computer
2. **Or use your camera**: Click "Take a photo" to capture an image with your webcam
3. **View results**: The app will display:
   - The input image (fixed 600px width)
   - Mask detection probability
   - Visual indicator (Mask detected / No mask detected)

## Error handling

The app includes comprehensive error handling for:
- Model loading failures
- Invalid image formats
- Image processing errors
- Inference errors

All errors are displayed to the user with helpful messages.

## License

See LICENSE file for details.