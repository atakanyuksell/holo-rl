#!/usr/bin/env python3
"""
ResNet18 Obstacle Detection Model - Training Script with Simple GUI
NO MATPLOTLIB - Only Tkinter GUI for stability
Obstacle_data klasöründeki verilerle ResNet18 modelini eğitir
Basit GUI ile real-time monitoring ve log tutma
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms

# Simple Tkinter GUI - No matplotlib
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import numpy as np
import os
import glob
from datetime import datetime
import threading
import time
import json
import cv2
from helpers.resnet18_obstacle_cnn import ResNet18ObstacleCNN

# Optional imports with fallbacks
try:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. Some metrics will be limited.")

# Fallback functions for when scikit-learn is not available
def simple_accuracy_score(y_true, y_pred):
    """Simple accuracy calculation without sklearn"""
    return (y_true == y_pred).mean()

def simple_confusion_matrix(y_true, y_pred):
    """Simple confusion matrix without sklearn"""
    # Convert to binary classification
    y_true_bin = (y_true > 0.5).astype(int)
    y_pred_bin = (y_pred > 0.5).astype(int)
    
    tp = ((y_true_bin == 1) & (y_pred_bin == 1)).sum()
    tn = ((y_true_bin == 0) & (y_pred_bin == 0)).sum()
    fp = ((y_true_bin == 0) & (y_pred_bin == 1)).sum()
    fn = ((y_true_bin == 1) & (y_pred_bin == 0)).sum()
    
    return np.array([[tn, fp], [fn, tp]])

def simple_precision_recall_fscore(y_true, y_pred):
    """Simple precision, recall, f1 calculation without sklearn"""
    y_true_bin = (y_true > 0.5).astype(int)
    y_pred_bin = (y_pred > 0.5).astype(int)
    
    tp = ((y_true_bin == 1) & (y_pred_bin == 1)).sum()
    tn = ((y_true_bin == 0) & (y_pred_bin == 0)).sum()
    fp = ((y_true_bin == 0) & (y_pred_bin == 1)).sum()
    fn = ((y_true_bin == 1) & (y_pred_bin == 0)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1, None

class ObstacleDataset(Dataset):
    """Obstacle detection için custom dataset"""
    
    def __init__(self, data_dir, transform=None, max_samples=None):
        self.transform = transform
        self.samples = []
        self.labels = []
        
        # With_obstacle klasörü (label=1)
        obstacle_path = os.path.join(data_dir, 'with_obstacle')
        if os.path.exists(obstacle_path):
            obstacle_files = glob.glob(os.path.join(obstacle_path, '*.npy'))
            if max_samples:
                obstacle_files = obstacle_files[:max_samples//2]
            
            for file_path in obstacle_files:
                try:
                    sample = np.load(file_path, allow_pickle=True).item()
                    if 'image' in sample:
                        self.samples.append(sample['image'])
                        self.labels.append(1.0)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
        
        # No_obstacle klasörü (label=0)
        no_obstacle_path = os.path.join(data_dir, 'no_obstacle')
        if os.path.exists(no_obstacle_path):
            no_obstacle_files = glob.glob(os.path.join(no_obstacle_path, '*.npy'))
            if max_samples:
                no_obstacle_files = no_obstacle_files[:max_samples//2]
            
            for file_path in no_obstacle_files:
                try:
                    sample = np.load(file_path, allow_pickle=True).item()
                    if 'image' in sample:
                        self.samples.append(sample['image'])
                        self.labels.append(0.0)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
        
        print(f"✅ Dataset loaded: {len(self.samples)} samples")
        print(f"   🔴 Obstacle: {sum(self.labels)} samples")
        print(f"   🟢 No Obstacle: {len(self.labels) - sum(self.labels)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image = self.samples[idx]
        label = self.labels[idx]
        
        # Image preprocessing
        if isinstance(image, np.ndarray):
            # Safe copy to avoid memory issues
            image = image.copy()
            
            if image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.0
            else:
                image = image.astype(np.float32)
            
            # BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[-1] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Ensure CHW format
            if len(image.shape) == 3 and image.shape[-1] == 3:
                image = torch.FloatTensor(image).permute(2, 0, 1)
            else:
                image = torch.FloatTensor(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.FloatTensor([label])

class TrainingConfig:
    """Training hyperparameters"""
    def __init__(self):
        self.batch_size = 4  # Small batch size for stability
        self.learning_rate = 1e-4
        self.weight_decay = 1e-4
        self.num_epochs = 100
        self.patience = 15
        self.min_delta = 1e-4
        self.train_split = 0.8
        self.val_split = 0.1
        self.test_split = 0.1
        self.save_interval = 5
        self.log_interval = 10

class ResNet18TrainerGUI:
    """ResNet18 training with simple GUI monitoring (No plots)"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("ResNet18 Obstacle Detection Trainer - Simple GUI")
        self.root.geometry("1000x700")
        
        # Training state
        self.training_active = False
        self.current_epoch = 0
        self.config = TrainingConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model and training objects
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.BCELoss()
        
        # Data loaders
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        # Training logs
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.learning_rates = []
        
        # Setup GUI
        self.setup_gui()
        self.load_data()
        
    def setup_gui(self):
        """Setup the simple GUI components"""
        
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # === CONFIGURATION SECTION ===
        config_frame = ttk.LabelFrame(main_frame, text="Training Configuration")
        config_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Config controls
        ttk.Label(config_frame, text="Batch Size:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.batch_size_var = tk.StringVar(value=str(self.config.batch_size))
        ttk.Entry(config_frame, textvariable=self.batch_size_var, width=10).grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(config_frame, text="Learning Rate:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=2)
        self.lr_var = tk.StringVar(value=str(self.config.learning_rate))
        ttk.Entry(config_frame, textvariable=self.lr_var, width=15).grid(row=0, column=3, padx=5, pady=2)
        
        ttk.Label(config_frame, text="Epochs:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.epochs_var = tk.StringVar(value=str(self.config.num_epochs))
        ttk.Entry(config_frame, textvariable=self.epochs_var, width=10).grid(row=1, column=1, padx=5, pady=2)
        
        ttk.Label(config_frame, text="Patience:").grid(row=1, column=2, sticky=tk.W, padx=5, pady=2)
        self.patience_var = tk.StringVar(value=str(self.config.patience))
        ttk.Entry(config_frame, textvariable=self.patience_var, width=10).grid(row=1, column=3, padx=5, pady=2)
        
        # === CONTROL BUTTONS ===
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.start_btn = ttk.Button(control_frame, text="🚀 Start Training", command=self.start_training)
        self.start_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        self.stop_btn = ttk.Button(control_frame, text="⏹️ Stop Training", command=self.stop_training, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        self.save_btn = ttk.Button(control_frame, text="💾 Save Model", command=self.save_model)
        self.save_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        self.test_btn = ttk.Button(control_frame, text="🧪 Test Model", command=self.test_model)
        self.test_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        # === PROGRESS SECTION ===
        progress_frame = ttk.LabelFrame(main_frame, text="Training Progress")
        progress_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.progress_var = tk.StringVar(value="Ready to start training")
        ttk.Label(progress_frame, textvariable=self.progress_var).pack(anchor=tk.W, padx=5, pady=2)
        
        self.progress_bar = ttk.Progressbar(progress_frame, mode='determinate')
        self.progress_bar.pack(fill=tk.X, padx=5, pady=5)
        
        # === METRICS SECTION ===
        metrics_frame = ttk.LabelFrame(main_frame, text="Current Metrics")
        metrics_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Create metrics grid
        metrics_grid = ttk.Frame(metrics_frame)
        metrics_grid.pack(fill=tk.X, padx=5, pady=5)
        
        # Left column
        left_metrics = ttk.Frame(metrics_grid)
        left_metrics.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.epoch_var = tk.StringVar(value="Epoch: 0/0")
        self.train_loss_var = tk.StringVar(value="Train Loss: 0.000")
        self.val_loss_var = tk.StringVar(value="Val Loss: 0.000")
        
        ttk.Label(left_metrics, textvariable=self.epoch_var, font=("Arial", 10, "bold")).pack(anchor=tk.W)
        ttk.Label(left_metrics, textvariable=self.train_loss_var).pack(anchor=tk.W)
        ttk.Label(left_metrics, textvariable=self.val_loss_var).pack(anchor=tk.W)
        
        # Right column
        right_metrics = ttk.Frame(metrics_grid)
        right_metrics.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.train_acc_var = tk.StringVar(value="Train Acc: 0.000%")
        self.val_acc_var = tk.StringVar(value="Val Acc: 0.000%")
        self.lr_display_var = tk.StringVar(value="Learning Rate: 0.000")
        
        ttk.Label(right_metrics, textvariable=self.train_acc_var).pack(anchor=tk.W)
        ttk.Label(right_metrics, textvariable=self.val_acc_var).pack(anchor=tk.W)
        ttk.Label(right_metrics, textvariable=self.lr_display_var).pack(anchor=tk.W)
        
        # === STATISTICS SECTION ===
        stats_frame = ttk.LabelFrame(main_frame, text="Training Statistics")
        stats_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Statistics display
        stats_grid = ttk.Frame(stats_frame)
        stats_grid.pack(fill=tk.X, padx=5, pady=5)
        
        # Best metrics
        self.best_val_loss_var = tk.StringVar(value="Best Val Loss: N/A")
        self.best_val_acc_var = tk.StringVar(value="Best Val Acc: N/A")
        self.total_time_var = tk.StringVar(value="Total Time: 0s")
        
        ttk.Label(stats_grid, textvariable=self.best_val_loss_var).pack(anchor=tk.W)
        ttk.Label(stats_grid, textvariable=self.best_val_acc_var).pack(anchor=tk.W)
        ttk.Label(stats_grid, textvariable=self.total_time_var).pack(anchor=tk.W)
        
        # === LOG SECTION ===
        log_frame = ttk.LabelFrame(main_frame, text="Training Log")
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=15, font=("Courier", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # === STATUS BAR ===
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(status_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W).pack(fill=tk.X)
        
    def log_message(self, message):
        """Add message to log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        self.log_text.insert(tk.END, log_entry)
        self.log_text.see(tk.END)
        self.root.update_idletasks()
        
    def load_data(self):
        """Load and prepare dataset"""
        try:
            self.log_message("🔄 Loading dataset...")
            self.status_var.set("Loading dataset...")
            
            # No transforms for stability
            train_transform = None
            val_transform = None
            
            # Load dataset
            dataset = ObstacleDataset('obstacle_data', transform=None)
            
            if len(dataset) == 0:
                raise ValueError("No data found in obstacle_data directory!")
            
            # Split dataset
            total_size = len(dataset)
            train_size = int(self.config.train_split * total_size)
            val_size = int(self.config.val_split * total_size)
            test_size = total_size - train_size - val_size
            
            train_dataset, val_dataset, test_dataset = random_split(
                dataset, [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(42)
            )
            
            # Create data loaders (GUI threading safe)
            self.train_loader = DataLoader(
                train_dataset, batch_size=self.config.batch_size,
                shuffle=True, num_workers=0, pin_memory=False
            )
            
            self.val_loader = DataLoader(
                val_dataset, batch_size=self.config.batch_size,
                shuffle=False, num_workers=0, pin_memory=False
            )
            
            self.test_loader = DataLoader(
                test_dataset, batch_size=self.config.batch_size,
                shuffle=False, num_workers=0, pin_memory=False
            )
            
            self.log_message(f"✅ Dataset loaded successfully!")
            self.log_message(f"   📊 Train: {train_size}, Val: {val_size}, Test: {test_size}")
            self.log_message(f"   🎯 Device: {self.device}")
            self.status_var.set("Dataset loaded - Ready to train")
            
        except Exception as e:
            self.log_message(f"❌ Error loading dataset: {e}")
            self.status_var.set("Error loading dataset")
            messagebox.showerror("Error", f"Failed to load dataset: {e}")
    
    def init_model(self):
        """Initialize model and training components"""
        try:
            self.log_message("🤖 Initializing model...")
            
            # Update config from GUI
            self.config.batch_size = int(self.batch_size_var.get())
            self.config.learning_rate = float(self.lr_var.get())
            self.config.num_epochs = int(self.epochs_var.get())
            self.config.patience = int(self.patience_var.get())
            
            # Initialize model
            self.model = ResNet18ObstacleCNN().to(self.device)
            self.log_message(f"✅ ResNet18 model initialized on {self.device}")
            
            # Initialize optimizer
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
            
            # Initialize scheduler
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=8, verbose=True
            )
            
            # Reset training logs
            self.train_losses = []
            self.val_losses = []
            self.train_accuracies = []
            self.val_accuracies = []
            self.learning_rates = []
            
            self.log_message("✅ Model and optimizer initialized")
            
        except Exception as e:
            self.log_message(f"❌ Error initializing model: {e}")
            raise e
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        try:
            for batch_idx, (data, target) in enumerate(self.train_loader):
                if not self.training_active:
                    break
                    
                # Safe GPU transfer
                data = data.to(self.device, non_blocking=False)
                target = target.to(self.device, non_blocking=False)
                target = target.squeeze()
                
                self.optimizer.zero_grad()
                output = self.model(data)
                
                # Handle tensor dimensions
                if output.dim() > 1:
                    output = output.squeeze()
                
                loss = self.criterion(output, target)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                total_loss += loss.item()
                predicted = (output > 0.5).float()
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
                # Update progress safely
                if batch_idx % self.config.log_interval == 0:
                    progress = 100. * batch_idx / len(self.train_loader)
                    self.progress_bar['value'] = progress
                    self.root.update_idletasks()
                
                # Clear cache periodically
                if batch_idx % 5 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Additional safety check
                if not self.training_active:
                    break
                    
        except Exception as e:
            self.log_message(f"❌ Training batch error: {e}")
            raise e
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        try:
            with torch.no_grad():
                for data, target in self.val_loader:
                    # Safe GPU transfer
                    data = data.to(self.device, non_blocking=False)
                    target = target.to(self.device, non_blocking=False)
                    target = target.squeeze()
                    
                    output = self.model(data)
                    
                    # Handle tensor dimensions
                    if output.dim() > 1:
                        output = output.squeeze()
                    
                    loss = self.criterion(output, target)
                    
                    total_loss += loss.item()
                    predicted = (output > 0.5).float()
                    total += target.size(0)
                    correct += (predicted == target).sum().item()
        except Exception as e:
            self.log_message(f"❌ Validation error: {e}")
            raise e
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def training_loop(self):
        """Main training loop"""
        best_val_loss = float('inf')
        best_val_acc = 0.0
        patience_counter = 0
        start_training_time = time.time()
        
        try:
            self.log_message("🚀 Starting training loop...")
            
            for epoch in range(1, self.config.num_epochs + 1):
                if not self.training_active:
                    break
                
                self.current_epoch = epoch
                start_time = time.time()
                
                # Train epoch
                train_loss, train_acc = self.train_epoch()
                
                # Validate epoch
                val_loss, val_acc = self.validate_epoch()
                
                # Update scheduler
                self.scheduler.step(val_loss)
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # Store metrics
                self.train_losses.append(train_loss)
                self.val_losses.append(val_loss)
                self.train_accuracies.append(train_acc)
                self.val_accuracies.append(val_acc)
                self.learning_rates.append(current_lr)
                
                # Update GUI
                epoch_time = time.time() - start_time
                total_time = time.time() - start_training_time
                
                self.epoch_var.set(f"Epoch: {epoch}/{self.config.num_epochs}")
                self.train_loss_var.set(f"Train Loss: {train_loss:.4f}")
                self.val_loss_var.set(f"Val Loss: {val_loss:.4f}")
                self.train_acc_var.set(f"Train Acc: {train_acc:.2f}%")
                self.val_acc_var.set(f"Val Acc: {val_acc:.2f}%")
                self.lr_display_var.set(f"Learning Rate: {current_lr:.2e}")
                self.total_time_var.set(f"Total Time: {total_time:.0f}s")
                
                # Update best metrics
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.best_val_loss_var.set(f"Best Val Loss: {best_val_loss:.4f}")
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self.best_val_acc_var.set(f"Best Val Acc: {best_val_acc:.2f}%")
                
                self.log_message(
                    f"Epoch {epoch:3d}/{self.config.num_epochs} | "
                    f"Train: {train_loss:.4f} ({train_acc:.1f}%) | "
                    f"Val: {val_loss:.4f} ({val_acc:.1f}%) | "
                    f"LR: {current_lr:.2e} | "
                    f"Time: {epoch_time:.1f}s"
                )
                
                self.status_var.set(f"Training Epoch {epoch}/{self.config.num_epochs}")
                
                # Early stopping
                if val_loss < best_val_loss - self.config.min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    self.save_model(best=True)
                else:
                    patience_counter += 1
                
                if patience_counter >= self.config.patience:
                    self.log_message(f"🛑 Early stopping at epoch {epoch}")
                    break
                
                # Periodic save
                if epoch % self.config.save_interval == 0:
                    self.save_model()
                
                # Reset progress bar
                self.progress_bar['value'] = 0
            
            self.log_message("🎉 Training completed!")
            self.status_var.set("Training completed successfully")
            
        except Exception as e:
            self.log_message(f"❌ Training error: {e}")
            self.status_var.set("Training failed")
        finally:
            self.training_active = False
            self.start_btn['state'] = tk.NORMAL
            self.stop_btn['state'] = tk.DISABLED
            self.progress_var.set("Training finished")
    
    def start_training(self):
        """Start training in separate thread"""
        if self.training_active:
            return
        
        try:
            self.init_model()
            self.training_active = True
            self.start_btn['state'] = tk.DISABLED
            self.stop_btn['state'] = tk.NORMAL
            self.progress_var.set("Training started...")
            self.status_var.set("Training started")
            
            # Start training thread
            training_thread = threading.Thread(target=self.training_loop, daemon=True)
            training_thread.start()
            
        except Exception as e:
            self.log_message(f"❌ Failed to start training: {e}")
            self.status_var.set("Failed to start training")
            messagebox.showerror("Error", f"Failed to start training: {e}")
    
    def stop_training(self):
        """Stop training"""
        self.training_active = False
        self.start_btn['state'] = tk.NORMAL
        self.stop_btn['state'] = tk.DISABLED
        self.progress_var.set("Training stopped")
        self.status_var.set("Training stopped by user")
        self.log_message("⏹️ Training stopped by user")
    
    def save_model(self, best=False):
        """Save model"""
        if self.model is None:
            return
        
        try:
            os.makedirs('obstacle_models', exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if best:
                filename = f"obstacle_models/resnet18_best_{timestamp}.pth"
            else:
                filename = f"obstacle_models/resnet18_epoch{self.current_epoch}_{timestamp}.pth"
            
            # Save model state
            save_data = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epoch': self.current_epoch,
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'train_accuracies': self.train_accuracies,
                'val_accuracies': self.val_accuracies,
                'config': self.config.__dict__
            }
            
            torch.save(save_data, filename)
            self.log_message(f"💾 Model saved: {filename}")
            
        except Exception as e:
            self.log_message(f"❌ Error saving model: {e}")
    
    def test_model(self):
        """Test model on test set - Fixed tensor dimension handling"""
        if self.model is None:
            messagebox.showwarning("Warning", "No model available for testing")
            return
        
        try:
            self.log_message("🧪 Testing model...")
            self.status_var.set("Testing model...")
            
            self.model.eval()
            all_predictions = []
            all_targets = []
            test_loss = 0
            
            with torch.no_grad():
                for data, target in self.test_loader:
                    data = data.to(self.device)
                    target = target.to(self.device)
                    
                    # Ensure target is properly shaped
                    if target.dim() > 1:
                        target = target.squeeze()
                    
                    output = self.model(data)
                    
                    # Ensure output is properly shaped
                    if output.dim() > 1:
                        output = output.squeeze()
                    
                    loss = self.criterion(output, target)
                    test_loss += loss.item()
                    
                    # Convert to predictions
                    predicted = (output > 0.5).float()
                    
                    # Safe conversion to numpy with proper handling
                    # Convert tensor to numpy and ensure 1D
                    pred_numpy = predicted.cpu().numpy()
                    target_numpy = target.cpu().numpy()
                    
                    # Handle scalar tensors (0-d arrays)
                    if pred_numpy.ndim == 0:
                        pred_numpy = np.array([pred_numpy])
                    if target_numpy.ndim == 0:
                        target_numpy = np.array([target_numpy])
                    
                    # Flatten to ensure 1D
                    pred_numpy = pred_numpy.flatten()
                    target_numpy = target_numpy.flatten()
                    
                    all_predictions.extend(pred_numpy)
                    all_targets.extend(target_numpy)
                    
                    # Clear cache after each batch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            # Convert to numpy arrays
            all_predictions = np.array(all_predictions)
            all_targets = np.array(all_targets)
            
            # Calculate metrics
            if SKLEARN_AVAILABLE:
                accuracy = accuracy_score(all_targets, all_predictions)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    all_targets, all_predictions, average='binary'
                )
            else:
                accuracy = simple_accuracy_score(all_targets, all_predictions)
                precision, recall, f1, _ = simple_precision_recall_fscore(all_targets, all_predictions)
            
            avg_test_loss = test_loss / len(self.test_loader)
            
            # Show results
            results_text = (
                f"🧪 TEST RESULTS\n"
                f"Test Loss: {avg_test_loss:.4f}\n"
                f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n"
                f"Precision: {precision:.4f}\n"
                f"Recall: {recall:.4f}\n"
                f"F1-Score: {f1:.4f}"
            )
            
            self.log_message(results_text)
            self.status_var.set("Testing completed")
            messagebox.showinfo("Test Results", results_text)
            
            # Simple confusion matrix display
            if SKLEARN_AVAILABLE:
                cm = confusion_matrix(all_targets, all_predictions)
            else:
                cm = simple_confusion_matrix(all_targets, all_predictions)
            
            cm_text = (f"Confusion Matrix:\n"
                      f"TN: {cm[0,0]}  FP: {cm[0,1]}\n"
                      f"FN: {cm[1,0]}  TP: {cm[1,1]}")
            
            self.log_message(cm_text)
            
        except Exception as e:
            self.log_message(f"❌ Error testing model: {e}")
            self.status_var.set("Testing failed")
            messagebox.showerror("Error", f"Testing failed: {e}")
            # Debug information
            import traceback
            self.log_message(f"🔍 Debug traceback: {traceback.format_exc()}")
    
    def run(self):
        """Run the GUI"""
        self.log_message("🚀 ResNet18 Obstacle Detection Trainer started")
        self.log_message(f"📁 Data directory: obstacle_data/")
        self.log_message(f"🎯 Device: {self.device}")
        self.status_var.set("Application started")
        self.root.mainloop()

def main():
    """Main function"""
    print("🚀 Starting ResNet18 Obstacle Detection Trainer (Simple GUI)...")
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
    else:
        print("⚠️ CUDA not available, using CPU")
    
    # Check data directory
    if not os.path.exists('obstacle_data'):
        print("❌ Error: obstacle_data directory not found!")
        print("Please ensure obstacle_data/ with with_obstacle/ and no_obstacle/ subdirectories exists")
        return
    
    # Start GUI
    app = ResNet18TrainerGUI()
    app.run()

if __name__ == "__main__":
    main()
