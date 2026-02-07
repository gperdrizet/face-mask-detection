"""
Model utilities for face mask detection.
Contains model architecture, loading, preprocessing, and inference functions.
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image


def create_model(device):
    '''
    Create the face mask detection CNN model.
    
    Args:
        device: torch.device for model placement
        
    Returns:
        model: PyTorch model instance
    '''

    model = nn.Sequential(

        # Conv block: RGB input (3 channels)
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Dropout(0.25),

        # Conv block
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Dropout(0.25),

        # Conv block
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Dropout(0.25),

        # Conv block
        nn.Conv2d(128, 256, kernel_size=3, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.Conv2d(256, 256, kernel_size=3, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Dropout(0.25),
        
        # Classifier
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, 2)  # Binary classification: 2 outputs for CrossEntropyLoss

    ).to(device)
    
    return model


def load_model(model_path, device):
    '''
    Load a trained model from checkpoint.
    
    Args:
        model_path: Path to the model checkpoint file
        device: torch.device for model placement
        
    Returns:
        tuple: (model, metadata_dict) where metadata contains class_names, 
               target_size, normalization parameters, etc.
    '''

    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Create model architecture
        model = create_model(device)
        
        # Load trained weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Extract metadata
        metadata = {
            'class_names': checkpoint.get('class_names', ['with_mask', 'without_mask']),
            'target_size': checkpoint.get('target_size', (128, 128)),
            'normalization_mean': checkpoint.get('normalization_mean', [0.485, 0.456, 0.406]),
            'normalization_std': checkpoint.get('normalization_std', [0.229, 0.224, 0.225])
        }
        
        return model, metadata
        
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")


def preprocess_image(image, target_size, normalization_mean, normalization_std):
    '''
    Preprocess an image for model inference.
    
    Args:
        image: PIL Image
        target_size: tuple (height, width) for resizing
        normalization_mean: list of mean values for normalization
        normalization_std: list of std values for normalization
        
    Returns:
        torch.Tensor: Preprocessed image tensor with shape (1, 3, H, W)
    '''

    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Define preprocessing transforms
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalization_mean, std=normalization_std)
    ])
    
    # Apply transforms and add batch dimension
    image_tensor = transform(image).unsqueeze(0)
    
    return image_tensor


def predict(model, image_tensor, device):
    '''
    Run inference on a preprocessed image.
    
    Args:
        model: PyTorch model
        image_tensor: Preprocessed image tensor
        device: torch.device for inference
        
    Returns:
        float: Probability of mask detection (index 0 = with_mask)
    '''

    with torch.no_grad():
        # Move tensor to device and run inference
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        
        # Apply softmax to get probabilities
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Return probability for "with_mask" class (index 0)
        mask_probability = probabilities[0, 0].item()
        
    return mask_probability
