'''
Face Mask Detection Web Application
A simple Streamlit app for detecting face masks in images.
'''

import base64
import io

from io import BytesIO
from pathlib import Path

import streamlit as st
import torch
from PIL import Image
from model_utils import load_model, preprocess_image, predict

PATH = Path(__file__).parent.resolve()

# Page configuration
st.set_page_config(
    page_title='Mask Detector',
    page_icon='',
    layout='centered'
)


@st.cache_resource
def load_cached_model():
    '''Load and cache the model to avoid reloading on every interaction.'''

    try:
        # Use CPU for inference
        device = torch.device('cpu')
        
        # Load the model
        model_path = f'{PATH}/face_mask_detector_production.pth'
        model, metadata = load_model(model_path, device)
        
        return model, metadata, device

    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()


def main():

    # Title
    st.title('Mask detector')
    
    # Load model (cached)
    model, metadata, device = load_cached_model()
    
    # Create placeholder for results (will be populated after image is processed)
    result_placeholder = st.container()
    
    # Input options
    st.subheader('Upload an image or use your camera')
    
    # File uploader
    uploaded_file = st.file_uploader(
        'Choose an image file',
        type=['jpg', 'jpeg', 'png'],
        help='Upload a photo to detect mask presence'
    )
    
    # Camera input
    camera_photo = st.camera_input('Or take a photo with your camera')
    
    # Determine which input to use (camera takes precedence)
    image_source = camera_photo if camera_photo is not None else uploaded_file
    
    if image_source is not None:
        try:
            # Load image
            image = Image.open(image_source)
            
            # Preprocess image
            image_tensor = preprocess_image(
                image,
                metadata['target_size'],
                metadata['normalization_mean'],
                metadata['normalization_std']
            )
            
            # Run inference
            import time
            with st.spinner('Analyzing image...'):
                start_time = time.perf_counter()
                mask_probability = predict(model, image_tensor, device)
                inference_time = time.perf_counter() - start_time
            
            # Display results in the placeholder (above the upload controls)
            with result_placeholder:
                
                # Resize image to 200px width for display
                display_width = 200
                aspect_ratio = image.height / image.width
                display_height = int(display_width * aspect_ratio)
                display_image = image.resize((display_width, display_height), Image.LANCZOS)
                
                # Convert resized image to base64 for embedding in HTML                
                buffered = BytesIO()
                display_image.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                # Display image centered in dark gray background
                st.markdown(
                    f"""
                    <div style="
                        background-color: #2b2b2b;
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        margin: 0;
                        padding: 0;
                    ">
                        <img src="data:image/png;base64,{img_str}" 
                             style="width: 200px; height: auto; margin: 0; padding: 0; display: block;"
                             alt="Input image">
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                st.write("")  # Add spacing
                
                # Show call & predicted probability
                if mask_probability >= 0.5:
                    st.info(f'Mask detected: P(mask) = {mask_probability:.2f} | Inference time: {inference_time*1000:.1f}ms')

                else:
                    st.info(f'No mask detected: P(mask) = {mask_probability:.2f} | Inference time: {inference_time*1000:.1f}ms')
                
        except Exception as e:
            with result_placeholder:
                st.error(f'Error processing image: {str(e)}')
                st.error('Please try a different image or check the image format.')
    
    # GitHub repository link at the bottom
    st.markdown('**GitHub repository:** [face-mask-detection](https://github.com/gperdrizet/face-mask-detection)')


if __name__ == "__main__":
    main()
