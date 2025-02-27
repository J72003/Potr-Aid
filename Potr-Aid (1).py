"""
PortrAid: An AI-Driven Portrait Assistant for Professional-Quality Image Composition

This module implements the PortrAid system for analyzing and improving portrait photo composition.
For end users, see the PortrAid class for simple image analysis.
For researchers/developers, the training pipeline is available through command-line arguments.
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Union, List
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
from captum.attr import IntegratedGradients, LayerGradCam, GuidedBackprop
import cv2

# Custom exception classes
class PortrAidError(Exception):
    """Base exception class for PortrAid errors"""
    pass

class DataError(PortrAidError):
    """Exception raised for data-related errors"""
    pass

class ModelError(PortrAidError):
    """Exception raised for model-related errors"""
    pass

def setup_logging(output_dir: Path) -> Tuple[Path, Path]:
    """Setup logging and create necessary directories"""
    dirs = ['logs', 'plots', 'models', 'analysis']
    paths = {}
    
    for dir_name in dirs:
        dir_path = output_dir / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)
        paths[dir_name] = dir_path
    
    log_file = paths['logs'] / f'portraid_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return paths

class PortraitDataset(Dataset):
    """Dataset class for portrait images"""
    def __init__(self, root_dir: Union[str, Path], transform=None, is_training: bool = True):
        self.root_dir = Path(root_dir)
        self.is_training = is_training
        self.transform = transform or self._get_default_transforms()
        self.samples = []
        self._setup_samples()
    
    def _get_default_transforms(self):
        """Get default transforms based on training/testing mode"""
        if self.is_training:
            return transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    
    def _setup_samples(self):
        """Set up image samples and their labels"""
        for class_name in ['uncropped', 'cropped']:
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                raise DataError(f"Required directory not found: {class_dir}")
            
            label = 1 if class_name == 'cropped' else 0
            for img_path in class_dir.glob('*.png'):
                self.samples.append((img_path, label))
        
        if len(self.samples) == 0:
            raise DataError(f"No valid images found in {self.root_dir}")
        
        logging.info(f"Found {len(self.samples)} images in {self.root_dir}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            logging.error(f"Error loading image {img_path}: {e}")
            return self._get_dummy_item()
    
    def _get_dummy_item(self):
        """Return a dummy item in case of loading errors"""
        dummy_tensor = torch.zeros((3, 224, 224))
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return (dummy_tensor - mean) / std, 0

class MultiScaleResNet(nn.Module):
    """Multi-scale ResNet model for portrait analysis"""
    def __init__(self):
        super().__init__()
        try:
            self._setup_encoders()
            self._setup_attention()
            self._setup_classifier()
            self.activation_maps = {}
            self.gradients = {}
            self._register_hooks()
        except Exception as e:
            raise ModelError(f"Error initializing model: {e}")
    
    def _setup_encoders(self):
        """Setup encoder networks"""
        self.micro_encoder = models.resnet50(weights=None)
        self.meso_encoder = models.resnet50(weights=None)
        self.macro_encoder = models.resnet50(weights=None)
        
        num_ftrs = self.micro_encoder.fc.in_features
        for encoder in [self.micro_encoder, self.meso_encoder, self.macro_encoder]:
            encoder.fc = nn.Sequential(
                nn.Linear(num_ftrs, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 128)
            )
    
    def _setup_attention(self):
        """Setup attention modules"""
        self.attention = nn.ModuleDict({
            'micro': self._create_attention_block(),
            'meso': self._create_attention_block(),
            'macro': self._create_attention_block()
        })
    
    def _create_attention_block(self):
        """Create an attention block"""
        return nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def _setup_classifier(self):
        """Setup the final classifier"""
        self.classifier = nn.Sequential(
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )
    
    def _register_hooks(self):
        """Register hooks for feature visualization"""
        def get_activation(name):
            def hook(module, input, output):
                self.activation_maps[name] = output
            return hook
        
        def get_gradient(name):
            def hook(module, grad_input, grad_output):
                self.gradients[name] = grad_output[0]
            return hook
        
        for name, encoder in [('micro', self.micro_encoder),
                            ('meso', self.meso_encoder),
                            ('macro', self.macro_encoder)]:
            encoder.layer4[-1].register_forward_hook(get_activation(f'{name}_layer4'))
            encoder.layer4[-1].register_full_backward_hook(get_gradient(f'{name}_layer4'))
    
    def forward(self, x):
        try:
            # Get features from each encoder
            micro_features = self.micro_encoder(x)
            meso_features = self.meso_encoder(x)
            macro_features = self.macro_encoder(x)
            
            # Apply attention
            micro_weights = self.attention['micro'](micro_features)
            meso_weights = self.attention['meso'](meso_features)
            macro_weights = self.attention['macro'](macro_features)
            
            # Weight features
            micro_features = micro_features * micro_weights
            meso_features = meso_features * meso_weights
            macro_features = macro_features * macro_weights
            
            # Combine features
            combined_features = torch.cat([micro_features, meso_features, macro_features], dim=1)
            return self.classifier(combined_features)
        except RuntimeError as e:
            if "out of memory" in str(e):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise ModelError("GPU out of memory. Try reducing batch size.")
            raise ModelError(f"Forward pass error: {e}")

class PortrAid:
    """Main interface for portrait analysis"""
    def __init__(self, weights_path: str = "best_model.pth"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = MultiScaleResNet().to(self.device)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.load_weights(weights_path)
    
    def load_weights(self, weights_path: str):
        """Load model weights"""
        try:
            checkpoint = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
        except Exception as e:
            raise ModelError(f"Error loading weights from {weights_path}: {e}")
    
    def analyze_image(self, image_path: Union[str, Path]) -> Dict:
        """
        Analyze a single portrait image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dict containing:
                - composition_score: Float between 0 and 1
                - is_well_composed: Boolean
                - confidence: Float between 0 and 1
                - feature_importance: Dict of attribution maps
        """
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(image_tensor)
                probabilities = torch.softmax(output, dim=1)[0]
                score = probabilities[0].item()
                
                # Get feature importance
                attributions = self._analyze_feature_importance(image_tensor)
                
                return {
                    'composition_score': score,
                    'is_well_composed': score > 0.5,
                    'confidence': float(abs(score - 0.5) * 2),
                    'feature_importance': attributions
                }
        except Exception as e:
            raise PortrAidError(f"Error analyzing image: {e}")
    
    def _analyze_feature_importance(self, image_tensor: torch.Tensor) -> Dict:
        """Analyze feature importance using multiple attribution methods"""
        integrated_gradients = IntegratedGradients(self.model)
        layer_gradcam = LayerGradCam(self.model, self.model.macro_encoder.layer4[-1])
        guided_backprop = GuidedBackprop(self.model)
        
        with torch.no_grad():
            ig_attr = integrated_gradients.attribute(image_tensor, target=0)
            gradcam_attr = layer_gradcam.attribute(image_tensor, target=0)
            gb_attr = guided_backprop.attribute(image_tensor, target=0)
        
        return {
            'integrated_gradients': ig_attr.cpu().numpy(),
            'gradcam': gradcam_attr.cpu().numpy(),
            'guided_backprop': gb_attr.cpu().numpy()
        }



def train_epoch(model: nn.Module, 
              loader: DataLoader, 
              criterion: nn.Module, 
              optimizer: torch.optim.Optimizer, 
              scaler: GradScaler, 
              device: torch.device) -> Tuple[float, float]:
    """
    Train for one epoch
    
    Returns:
        tuple: (average loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(loader, desc='Training'):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(loader), 100. * correct / total


        


def evaluate_model(model: nn.Module, 
                  loader: DataLoader, 
                  criterion: nn.Module, 
                  device: torch.device) -> Tuple[float, float, np.ndarray, str]:
    """
    Evaluate model performance
    
    Returns:
        tuple: (average loss, accuracy, confusion matrix, classification report)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc='Evaluating'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            _, predicted = outputs.max(1)
            running_loss += loss.item() * inputs.size(0)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    conf_matrix = confusion_matrix(all_labels, all_preds)
    class_report = classification_report(all_labels, all_preds, digits=4)
    
    return (running_loss / len(loader), 100. * correct / total,
            conf_matrix, class_report)




def visualize_attributions(image: torch.Tensor, attributions: Dict, save_path: Path):
    """Visualize feature attribution maps"""
    plt.figure(figsize=(15, 5))
    
    # Original image
    plt.subplot(141)
    img_np = image.cpu().numpy().transpose(1, 2, 0)
    img_np = (img_np * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]).clip(0, 1)
    plt.imshow(img_np)
    plt.title('Original Image')
    plt.axis('off')
    
    # Integrated Gradients
    plt.subplot(142)
    ig_attr = attributions['integrated_gradients'].squeeze().mean(axis=0)
    plt.imshow(ig_attr, cmap='seismic', center=0)
    plt.title('Integrated Gradients')
    plt.axis('off')
    
    # GradCAM
    plt.subplot(143)
    gradcam_attr = attributions['gradcam'].squeeze()
    plt.imshow(img_np)
    plt.imshow(gradcam_attr, cmap='jet', alpha=0.5)
    plt.title('GradCAM')
    plt.axis('off')
    
    # Guided Backprop
    plt.subplot(144)
    gb_attr = attributions['guided_backprop'].squeeze().mean(axis=0)
    plt.imshow(gb_attr, cmap='gray')
    plt.title('Guided Backprop')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='PortrAid: Portrait Composition Analysis')
    parser.add_argument('--mode', type=str, choices=['train', 'analyze'], required=True,
                      help='Operation mode: train model or analyze images')
    parser.add_argument('--data_dir', type=str,
                      help='Base directory containing train/val/test splits')
    parser.add_argument('--image_path', type=str,
                      help='Path to single image for analysis')
    parser.add_argument('--weights', type=str, default='best_model.pth',
                      help='Path to model weights')
    parser.add_argument('--output_dir', type=str, default='outputs',
                      help='Directory to save outputs')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                      help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                      help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Number of data loading workers')
    parser.add_argument('--resume', action='store_true',
                      help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    try:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = setup_logging(output_dir)
        
        if args.mode == 'analyze':
            if not args.image_path:
                raise ValueError("--image_path is required for analyze mode")
            
            # Initialize PortrAid for analysis
            portraid = PortrAid(weights_path=args.weights)
            
            # Analyze single image
            logging.info(f"Analyzing image: {args.image_path}")
            result = portraid.analyze_image(args.image_path)
            
            # Save results
            output_path = output_dir / 'analysis_result.json'
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=4)
            
            # Generate visualization
            if 'feature_importance' in result:
                viz_path = output_dir / 'feature_importance.png'
                image = Image.open(args.image_path).convert('RGB')
                image_tensor = portraid.transform(image)
                visualize_attributions(image_tensor, result['feature_importance'], viz_path)
            
            logging.info(f"Analysis complete. Results saved to {output_dir}")
            
        elif args.mode == 'train':
            if not args.data_dir:
                raise ValueError("--data_dir is required for train mode")
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logging.info(f"Using device: {device}")
            
            # Setup data paths
            data_dir = Path(args.data_dir)
            train_dir = data_dir / 'train'
            val_dir = data_dir / 'val'
            test_dir = data_dir / 'test'
            
            # Create datasets
            train_dataset = PortraitDataset(train_dir, is_training=True)
            val_dataset = PortraitDataset(val_dir, is_training=False)
            test_dataset = PortraitDataset(test_dir, is_training=False)
            
            # Create dataloaders
            train_loader = DataLoader(
                train_dataset, 
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True
            )
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True
            )
            
            # Initialize model
            model = MultiScaleResNet().to(device)
            
            # Setup training components
            criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
            optimizer = optim.AdamW([
                {'params': model.micro_encoder.parameters(), 'lr': args.lr},
                {'params': model.meso_encoder.parameters(), 'lr': args.lr},
                {'params': model.macro_encoder.parameters(), 'lr': args.lr},
                {'params': model.attention.parameters(), 'lr': args.lr * 10},
                {'params': model.classifier.parameters(), 'lr': args.lr * 10}
            ], weight_decay=0.01)
            
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=2
            )
            
            scaler = GradScaler()
            
            # Resume from checkpoint if specified
            start_epoch = 0
            best_val_acc = 0
            if args.resume:
                checkpoint_path = output_dir / 'latest_model.pth'
                if checkpoint_path.exists():
                    checkpoint = torch.load(checkpoint_path, map_location=device)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    start_epoch = checkpoint['epoch']
                    best_val_acc = checkpoint['best_acc']
                    logging.info(f"Resumed from epoch {start_epoch}")
                else:
                    logging.warning(f"No checkpoint found at {checkpoint_path}")
            
            # Training loop
            logging.info("Starting training...")
            for epoch in range(start_epoch, args.epochs):
                # Train
                train_loss, train_acc = train_epoch(
                    model, train_loader, criterion, optimizer, scaler, device
                )
                
                # Evaluate
                val_loss, val_acc, conf_matrix, class_report = evaluate_model(
                    model, val_loader, criterion, device
                )
                
                scheduler.step()
                
                # Log progress
                logging.info(
                    f"Epoch {epoch+1}/{args.epochs}: "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
                )
                
                # Save checkpoints
                is_best = val_acc > best_val_acc
                best_val_acc = max(val_acc, best_val_acc)
                
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_acc': best_val_acc,
                }
                
                # Save latest checkpoint
                torch.save(checkpoint, output_dir / 'latest_model.pth')
                
                # Save best model
                if is_best:
                    torch.save(checkpoint, output_dir / 'best_model.pth')
                    logging.info(f"New best model saved with accuracy {best_val_acc:.2f}%")
            
            # Final evaluation
            logging.info("Training completed. Running final evaluation...")
            checkpoint = torch.load(output_dir / 'best_model.pth', map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            test_loss, test_acc, final_conf_matrix, final_class_report = evaluate_model(
                model, test_loader, criterion, device
            )
            
            # Save results
            results = {
                'test_accuracy': float(test_acc),
                'confusion_matrix': final_conf_matrix.tolist(),
                'classification_report': final_class_report,
            }
            
            with open(output_dir / 'final_results.json', 'w') as f:
                json.dump(results, f, indent=4)
            
            logging.info(f"Final Test Accuracy: {test_acc:.2f}%")
            logging.info("Training and evaluation completed successfully!")
    
    except Exception as e:
        logging.error(f"Error in execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()