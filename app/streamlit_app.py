'''
Face Mask Detection Web Application
A simple Streamlit app for detecting face masks in images.
'''

# Standard library imports
import base64
import time

from io import BytesIO
from pathlib import Path

# Third-party imports
import streamlit as st
import torch
from PIL import Image

# Local imports
from model_utils import load_model, preprocess_image, predict

PATH = Path(__file__).parent.resolve()

# Page configuration
st.set_page_config(
    page_title='Mask detector',
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

    # Hide Streamlit header and menu
    st.markdown(
        """
        <style>
        /* Hide Streamlit header */
        header[data-testid="stHeader"] {
            display: none;
        }
        
        /* Hide the main menu button */
        #MainMenu {
            visibility: hidden;
        }
        
        /* Hide footer */
        footer {
            visibility: hidden;
        }
        
        /* Reduce top padding of main container */
        .stMainBlockContainer {
            padding-top: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Title
    st.title('Mask detector')
    
    # Load model (cached)
    model, metadata, device = load_cached_model()
    
    # Input options
    tab1, tab2 = st.tabs(['Camera', 'Upload file'])
    
    with tab1:
        # Camera input
        camera_photo = st.camera_input('Take a photo', label_visibility='collapsed')
        
        # Process camera photo
        if camera_photo is not None:
            try:
                # Load image
                image = Image.open(camera_photo)
                
                # Preprocess image
                image_tensor = preprocess_image(
                    image,
                    metadata['target_size'],
                    metadata['normalization_mean'],
                    metadata['normalization_std']
                )
                
                # Run inference
                with st.spinner('Analyzing image...'):
                    start_time = time.perf_counter()
                    mask_probability = predict(model, image_tensor, device)
                    inference_time = time.perf_counter() - start_time
                
                st.write('')  # Add spacing
                
                # Show result
                if mask_probability >= 0.5:
                    st.info(f'Mask detected: P(mask) = {mask_probability:.2f} | Inference time: {inference_time:.2f} s')

                else:
                    st.info(f'No mask detected: P(mask) = {mask_probability:.2f} | Inference time: {inference_time:.2f} s')
                    
            except Exception as e:
                st.error(f'Error processing image: {str(e)}')
                st.error('Please try a different image or check the image format.')
    
    with tab2:

        # File uploader
        uploaded_file = st.file_uploader(
            'Select an image file',
            label_visibility='collapsed',
            type=['jpg', 'jpeg', 'png'],
            help='Upload a photo to detect mask presence'
        )
        
        # Process uploaded file
        if uploaded_file is not None:
            try:

                # Load image
                image = Image.open(uploaded_file)
                
                # Preprocess image
                image_tensor = preprocess_image(
                    image,
                    metadata['target_size'],
                    metadata['normalization_mean'],
                    metadata['normalization_std']
                )
                
                # Run inference
                with st.spinner('Analyzing image...'):
                    start_time = time.perf_counter()
                    mask_probability = predict(model, image_tensor, device)
                    inference_time = time.perf_counter() - start_time
                
                # Resize image to 300px width for display
                display_width = 300
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
                             style="width: 300px; height: auto; margin: 0; padding: 0; display: block;"
                             alt="Input image">
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                st.write('')  # Add spacing
                
                # Show result
                if mask_probability >= 0.5:
                    st.info(f'Mask detected: P(mask) = {mask_probability:.2f} | Inference time: {inference_time:.2f} s')
                else:
                    st.info(f'No mask detected: P(mask) = {mask_probability:.2f} | Inference time: {inference_time:.2f} s')
                    
            except Exception as e:
                st.error(f'Error processing image: {str(e)}')
                st.error('Please try a different image or check the image format.')
    
    # GitHub repository link at the bottom
    st.markdown('**GitHub repository:** [gperdrizet/face-mask-detection](https://github.com/gperdrizet/face-mask-detection)')


if __name__ == "__main__":
    main()
