"""
Utility functions for face mask detection model training and evaluation.
"""

import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image


def convert_to_rgb(img):
    '''Convert image to RGB, handling palette images with transparency properly.
    
    Args:
        img: PIL Image to convert
    
    Returns:
        PIL Image in RGB mode
    '''
 
    if img.mode == 'P':
        # Convert palette images to RGBA first to handle transparency
        img = img.convert('RGBA')
    
    if img.mode != 'RGB':
        # Then convert to RGB, which will discard the alpha channel
        img = img.convert('RGB')
    
    return img


def plot_image_grid(image_data, title_func, max_cols=6, cell_size=1.5):
    '''Plot a grid of images with custom titles.
    
    Args:
        image_data: List of tuples, first element is image path, second is label
        title_func: Function that takes a data tuple and returns the title string
        max_cols: Maximum number of columns in the grid
        cell_size: Size multiplier for each cell
    '''

    if not image_data:
        return
    
    # Calculate grid dimensions
    n_images = len(image_data)
    ncols = min(max_cols, n_images)
    nrows = (n_images + ncols - 1) // ncols  # Ceiling division
    
    _, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*cell_size, nrows*cell_size))
    
    # Handle single image case
    if n_images == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if nrows > 1 else [axes] if ncols == 1 else axes
    
    for idx, data in enumerate(image_data):
        img_path = data[0]
        img = Image.open(img_path)
        axes[idx].imshow(img)
        axes[idx].set_title(title_func(data), fontsize=10)
        axes[idx].axis('off')
    
    # Hide unused subplots
    for idx in range(len(image_data), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()


def train_one_epoch(model, data_loader, criterion, optimizer, device, cyclic_scheduler=None, history=None):
    '''Run one training epoch, tracking metrics per batch.
    
    Args:
        model: PyTorch model to train
        data_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to move batches to (None = already on device)
        cyclic_scheduler: Optional CyclicLR scheduler to step after each batch
        history: Optional history dictionary to record batch-level metrics
    
    Returns:
        Tuple of (average_loss, accuracy_percentage) for the epoch
    '''

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(data_loader):
        if device is not None:
            images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Step cyclic scheduler after each batch
        if cyclic_scheduler is not None:
            cyclic_scheduler.step()
        
        # Track batch-level metrics
        batch_loss = loss.item()
        _, predicted = torch.max(outputs, 1)
        batch_correct = (predicted == labels).sum().item()
        batch_total = labels.size(0)
        batch_acc = 100 * batch_correct / batch_total
        
        # Record batch metrics if history is provided
        if history is not None:
            history['batch_train_loss'].append(batch_loss)
            history['batch_train_accuracy'].append(batch_acc)
            history['batch_learning_rates'].append(optimizer.param_groups[0]['lr'])
            
            if cyclic_scheduler is not None:
                history['batch_base_lrs'].append(cyclic_scheduler.base_lrs[0])
                history['batch_max_lrs'].append(cyclic_scheduler.max_lrs[0])
        
        running_loss += batch_loss
        correct += batch_correct
        total += batch_total
    
    return running_loss / len(data_loader), 100 * correct / total


def evaluate(model, data_loader, criterion, device):
    '''Evaluate model on a dataset.
    
    Args:
        model: PyTorch model to evaluate
        data_loader: Data loader (validation or test set)
        criterion: Loss function
        device: Device to move batches to (None = already on device)
    
    Returns:
        Tuple of (average_loss, accuracy_percentage) for the dataset
    '''

    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in data_loader:
            if device is not None:
                images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return running_loss / len(data_loader), 100 * correct / total


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader = None,
    criterion: nn.Module = None,
    optimizer: optim.Optimizer = None,
    cyclic_scheduler = None,
    lr_schedule: dict = None,
    epochs: int = 10,
    early_stopping_patience: int = 10,
    print_every: int = 1,
    device: torch.device = None
) -> dict[str, list[float]]:
    '''Training loop with optional validation and early stopping.
    
    Tracks metrics at both batch and epoch levels.
    
    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader (None for training without validation)
        criterion: Loss function
        optimizer: Optimizer
        cyclic_scheduler: CyclicLR scheduler (steps per batch)
        lr_schedule: Optional dict with scheduled LR bounds reduction:
                    {'initial_base_lr', 'initial_max_lr', 'final_base_lr', 
                     'final_max_lr', 'schedule_epochs'}
        epochs: Maximum number of epochs
        early_stopping_patience: Stop if val_loss doesn't improve for this many epochs
                                (ignored if val_loader is None)
        print_every: Print progress every N epochs
        device: Device to move batches to (None = already on device)
    
    Returns:
        Dictionary containing training history with epoch and batch-level metrics
    '''

    history = {
        # Epoch-level metrics
        'train_loss': [], 'val_loss': [],
        'train_accuracy': [], 'val_accuracy': [],
        'learning_rates': [],
        'base_lrs': [],
        'max_lrs': [],
        
        # Batch-level metrics
        'batch_train_loss': [],
        'batch_train_accuracy': [],
        'batch_learning_rates': [],
        'batch_base_lrs': [],
        'batch_max_lrs': []
    }
    
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model_state = None

    for epoch in range(epochs):

        # Train and validate (now passing history to track batch metrics)
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, cyclic_scheduler, history)
        
        # Only evaluate on validation set if provided
        if val_loader is not None:
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        else:
            val_loss, val_acc = None, None
        # Only evaluate on validation set if provided
        if val_loader is not None:
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        else:
            val_loss, val_acc = None, None
        
        # Record epoch-level metrics
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss if val_loss is not None else float('nan'))
        history['train_accuracy'].append(train_acc)
        history['val_accuracy'].append(val_acc if val_acc is not None else float('nan'))
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # Record base and max LR if using cyclic scheduler
        if cyclic_scheduler is not None:
            history['base_lrs'].append(cyclic_scheduler.base_lrs[0])
            history['max_lrs'].append(cyclic_scheduler.max_lrs[0])
        
        # Early stopping (only if validation data is provided)
        if val_loader is not None and val_loss is not None:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                epochs_without_improvement += 1
        
        # Update LR bounds based on schedule
        if lr_schedule is not None and cyclic_scheduler is not None and epoch < lr_schedule['schedule_epochs']:
            # Linear interpolation of base and max LR
            progress = (epoch + 1) / lr_schedule['schedule_epochs']
            new_base_lr = lr_schedule['initial_base_lr'] * (1 - progress) + lr_schedule['final_base_lr'] * progress
            new_max_lr = lr_schedule['initial_max_lr'] * (1 - progress) + lr_schedule['final_max_lr'] * progress
            
            # Update the cyclic scheduler's base and max LRs
            cyclic_scheduler.base_lrs = [new_base_lr]
            cyclic_scheduler.max_lrs = [new_max_lr]
        
        # Print progress
        if (epoch + 1) % print_every == 0 or epoch == 0:
            lr = optimizer.param_groups[0]['lr']
            base_lr = cyclic_scheduler.base_lrs[0] if cyclic_scheduler else lr
            max_lr = cyclic_scheduler.max_lrs[0] if cyclic_scheduler else lr
            
            if val_loader is not None:
                print(
                    f'Epoch {epoch+1:3d}/{epochs} - '
                    f'loss: {train_loss:.4f} - acc: {train_acc:5.2f}% - '
                    f'val_loss: {val_loss:.4f} - val_acc: {val_acc:5.2f}% - '
                    f'lr: {lr:.2e} (base: {base_lr:.2e}, max: {max_lr:.2e})'
                )
            else:
                print(
                    f'Epoch {epoch+1:3d}/{epochs} - '
                    f'loss: {train_loss:.4f} - acc: {train_acc:5.2f}% - '
                    f'lr: {lr:.2e} (base: {base_lr:.2e}, max: {max_lr:.2e})'
                )
        
        # Check early stopping (only if validation data is provided)
        if val_loader is not None and epochs_without_improvement >= early_stopping_patience:
            print(f'\nEarly stopping at epoch {epoch + 1}')
            print(f'Best val_loss: {best_val_loss:.4f} at epoch {epoch + 1 - epochs_without_improvement}')
            break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f'Restored best model weights')
    
    return history


def generate_augmented_data(X_train, y_train, augmentation_transforms, augmentations_per_image, 
                           save_path=None, force_reaugment=False):
    '''Generate augmented training data with optional saving and loading.
    
    Args:
        X_train: Training images tensor (on GPU or CPU)
        y_train: Training labels tensor
        augmentation_transforms: nn.Sequential containing augmentation transforms
        augmentations_per_image: Number of augmented versions to create per image
        save_path: Optional path to save/load augmented data
        force_reaugment: If True, regenerate even if saved data exists
    
    Returns:
        Tuple of (X_train_final, y_train_final) on CPU
    '''
    
    # Move data to CPU for augmentation
    X_train_cpu = X_train.cpu()
    y_train_cpu = y_train.cpu()
    
    # Check if saved augmented data exists
    if save_path and os.path.exists(save_path) and not force_reaugment:
        print(f'Loading pre-generated augmented data from {save_path}...')
        saved_data = torch.load(save_path)
        X_train_final = saved_data['X_train']
        y_train_final = saved_data['y_train']
        
        print(f'\nLoaded augmented training set:')
        print(f'  Total size: {len(X_train_final)}')
        print(f'  Original: {len(X_train_cpu)}')
        print(f'  Added: {len(X_train_final) - len(X_train_cpu)}')
        print(f'  Memory location: {X_train_final.device}')
        print(f'  Augmentation factor: {len(X_train_final) / len(X_train_cpu):.1f}x')
        
    else:
        if force_reaugment:
            print('Forcing re-augmentation...')
        else:
            print('No saved augmented data found. Generating augmentations...')
        
        # Lists to collect augmented data on CPU
        X_train_aug = [X_train_cpu]  # Start with original training data
        y_train_aug = [y_train_cpu]
        
        # Generate augmented versions on CPU
        # Apply transforms to each image individually to ensure independent transformations
        for aug_idx in range(augmentations_per_image):
            print(f'Creating augmentation batch {aug_idx + 1}/{augmentations_per_image}...')
            
            # Apply augmentations to each training image individually
            X_aug_list = []
            for img in X_train_cpu:
                # Add batch dimension, apply transform, remove batch dimension
                img_aug = augmentation_transforms(img.unsqueeze(0)).squeeze(0)
                X_aug_list.append(img_aug)
            
            X_aug = torch.stack(X_aug_list)
            
            X_train_aug.append(X_aug)
            y_train_aug.append(y_train_cpu)
        
        # Concatenate all training data (original + augmented) - stays on CPU
        X_train_final = torch.cat(X_train_aug, dim=0)
        y_train_final = torch.cat(y_train_aug, dim=0)
        
        print(f'\nAugmented training set size: {len(X_train_final)}')
        print(f'  Added: {len(X_train_final) - len(X_train_cpu)}')
        print(f'  Original: {len(X_train_cpu)}')
        print(f'  Memory location: {X_train_final.device}')
        print(f'  Augmentation factor: {len(X_train_final) / len(X_train_cpu):.1f}x')
        
        # Save augmented data for future use
        if save_path:
            print(f'\nSaving augmented data to {save_path}...')
            torch.save({
                'X_train': X_train_final,
                'y_train': y_train_final,
                'augmentations_per_image': augmentations_per_image,
                'original_train_size': len(X_train_cpu)
            }, save_path)
            print('Augmented data saved successfully!')
    
    return X_train_final, y_train_final

