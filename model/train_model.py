import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from dataloaders import GetSingleStepDataLoaders
import model
import os
import time
import matplotlib.pyplot as plt

from train import train_epoch, test_epoch, test_batch
import utils



class TrainingWrapper:
    """Wrapper to add modern training features to existing train_model function"""
    
    def __init__(self, scheduler, best_model_path, patience=30, min_delta=1e-5, 
                 gradient_clip=1.0, resume_epoch=0, resume_best_loss=None):
        self.scheduler = scheduler
        self.best_model_path = best_model_path
        self.patience = patience
        self.min_delta = min_delta
        self.gradient_clip = gradient_clip
        
        # Early stopping tracking
        self.early_stop_best_loss = resume_best_loss  # Initialize with resumed value
        self.counter = 0
        self.early_stop_best_epoch = resume_epoch if resume_best_loss else 0
        self.early_stop = False
        
        # Best model tracking
        self.best_loss = resume_best_loss if resume_best_loss else float('inf')
        self.best_epoch = resume_epoch if resume_best_loss else 0
        
        # Tensorboard
        self.writer = SummaryWriter(log_dir=os.path.join(output_folder, 'logs'))
        
        if resume_epoch > 0:
            print(f"Resuming training wrapper from epoch {resume_epoch}")
            print(f"  Best loss so far: {self.best_loss:.6f}")
    
    def on_epoch_end(self, epoch, train_loss, test_loss, net):
        """Call this at the end of each epoch"""
        
        # Log to tensorboard
        current_lr = self.scheduler.get_last_lr()
        print(current_lr)
        self.writer.add_scalar('Loss/train', train_loss, epoch)
        self.writer.add_scalar('Loss/test', test_loss, epoch)
        self.writer.add_scalar('Learning_Rate', current_lr[0], epoch)
        
        print(f"  LR: {current_lr[0]:.6f}")
        
        # Save best model
        if test_loss < self.best_loss:
            self.best_loss = test_loss
            self.best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': self.scheduler.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),  # Save scheduler too!
                'train_loss': train_loss,
                'test_loss': test_loss,
            }, self.best_model_path)
            print(f"  ✓ Saved best model (test_loss: {test_loss:.6f})")
        
        # Early stopping check
        if self.early_stop_best_loss is None:
            self.early_stop_best_loss = test_loss
            self.early_stop_best_epoch = epoch
            self.counter = 0
            print(f"  Early stopping: initialized")
        elif test_loss < (self.early_stop_best_loss - self.min_delta):
            improvement = self.early_stop_best_loss - test_loss
            self.early_stop_best_loss = test_loss
            self.early_stop_best_epoch = epoch
            self.counter = 0
            print(f"  Early stopping: improvement of {improvement:.6f}, counter reset")
        else:
            self.counter += 1
            print(f"  Early stopping: {self.counter}/{self.patience} (best: epoch {self.early_stop_best_epoch}, loss {self.early_stop_best_loss:.6f})")
            
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"\n⚠ Early stopping triggered!")
                print(f"  Best epoch: {self.early_stop_best_epoch} with test_loss: {self.early_stop_best_loss:.6f}")
        
        # Update learning rate
        self.scheduler.step(test_loss)
        
        # Clip learning rate
        for param_group in self.scheduler.optimizer.param_groups:
            param_group['lr'] = max(param_group['lr'], MIN_LEARNING_RATE)

    
    def close(self):
        self.writer.close()

def enhanced_train_model(train_loader, test_loader, net, epochs, optimizer, device, 
                        save_results, output_folder, start_epoch=0, 
                        train_loss_list=None, test_loss_list=None):
    """
    Enhanced version of train_model with modern features and resume support
    
    Args:
        start_epoch: Epoch to start/resume from (0 for new training)
        train_loss_list: Previous training losses (None for new training)
        test_loss_list: Previous test losses (None for new training)
    """
    
    # Initialize or continue loss lists
    if train_loss_list is None:
        train_loss_list = []
    if test_loss_list is None:
        test_loss_list = []
    
    print("\n" + "="*80)
    if start_epoch > 0:
        print(f"RESUMING TRAINING FROM EPOCH {start_epoch}")
        print(f"Previous training: {len(train_loss_list)} epochs")
    else:
        print("STARTING TRAINING")
    print("="*80 + "\n")
    
    for i in range(start_epoch, epochs):  # Note: range starts from start_epoch
        startTime = time.time()
        
        train_loss = train_epoch(train_loader, net, optimizer, device)
        train_loss_list.append(train_loss)
        test_loss = test_epoch(test_loader, net, device)
        test_loss_list.append(test_loss)
        
        epoch_time = time.time() - startTime
        
        writeString = (
            f"epoch {i} train loss: {train_loss:.7f} test loss: {test_loss:.7f} "
            f"epoch time: {epoch_time:.2f}s"
        )
        print(writeString)
        
        # Enhanced features
        training_wrapper.on_epoch_end(i, train_loss, test_loss, net)
        
        # Plot losses
        plt.figure(figsize=(10, 6))
        plt.plot(train_loss_list, label="Train", linewidth=2)
        plt.plot(test_loss_list, label="Test", linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_results:
            with open(output_folder + "prints.txt", "a") as file: 
                file.write(writeString + "\n")
            plt.savefig(output_folder + "loss_512.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Visualization checkpoints
        if i % SAVE_EVERY == 0:
            test_samples, test_actions, test_deltas, test_samples_next = next(iter(test_loader))
            loss, test_output = test_batch(test_samples, test_actions, test_deltas, net, device)
            utils.plotPCbatch(
                test_samples,
                test_samples_next[:, -1, :, :],
                test_output,
                show=False,
                save=True,
                name=(output_folder + f"epoch_{i}")
            )
        
        # Early stopping check
        if training_wrapper.early_stop:
            print(f"\nStopping training at epoch {i}")
            break
    
    training_wrapper.close()
    
    # Final summary
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Best test loss: {training_wrapper.best_loss:.7f} at epoch {training_wrapper.best_epoch}")
    print(f"Final train loss: {train_loss_list[-1]:.7f}")
    print(f"Final test loss: {test_loss_list[-1]:.7f}")
    
    return train_loss_list, test_loss_list

def load_checkpoint(checkpoint_path, net, optimizer=None, scheduler=None, device='cuda'):
    """
    Load a saved checkpoint and optionally restore optimizer/scheduler state
    
    Args:
        checkpoint_path: Path to the .pth checkpoint file
        net: The model to load weights into
        optimizer: (Optional) Optimizer to restore state
        scheduler: (Optional) Scheduler to restore state
        device: Device to load model onto
    
    Returns:
        start_epoch: The epoch to resume from
        train_loss: Training loss from checkpoint
        test_loss: Test loss from checkpoint
    """
    print(f"\n{'='*80}")
    print(f"Loading checkpoint from: {checkpoint_path}")
    print('='*80)
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model weights
    if isinstance(net, torch.nn.DataParallel):
        # Handle DataParallel models
        net.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        net.load_state_dict(checkpoint['model_state_dict'])
    
    print("✓ Loaded model weights")
    
    # Optionally restore optimizer state
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("✓ Restored optimizer state")
    
    # Optionally restore scheduler state
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print("✓ Restored scheduler state")
    
    # Get training info
    start_epoch = checkpoint.get('epoch', 0) + 1  # Resume from next epoch
    train_loss = checkpoint.get('train_loss', None)
    test_loss = checkpoint.get('test_loss', None)
    
    print(f"\nCheckpoint info:")
    print(f"  Epoch: {checkpoint.get('epoch', 0)}")
    print(f"  Train loss: {train_loss:.6f}" if train_loss else "  Train loss: N/A")
    print(f"  Test loss: {test_loss:.6f}" if test_loss else "  Test loss: N/A")
    print(f"  Resuming from epoch: {start_epoch}")
    print('='*80 + '\n')
    
    return start_epoch, train_loss, test_loss

def load_training_history(history_path):
    """
    Load previous training history to continue plotting
    
    Args:
        history_path: Path to training_history.npz
    
    Returns:
        train_losses: List of training losses
        test_losses: List of test losses
    """
    if os.path.exists(history_path):
        print(f"Loading training history from: {history_path}")
        data = np.load(history_path)
        train_losses = data['train_losses'].tolist()
        test_losses = data['test_losses'].tolist()
        print(f"✓ Loaded {len(train_losses)} epochs of history\n")
        return train_losses, test_losses
    else:
        print(f"No existing history found, starting fresh\n")
        return [], []

if __name__ == "__main__":
            
    # ============================================================================
    # HYPERPARAMETERS
    # ============================================================================
    batch_size = 256
    output_folder = "./output_cogging/"
    save_results = True
    use_GPU = True
    latent_size = 512
    dropout = 0.1
    epochs = 500

    # Training hyperparameters (based on PointNet best practices)
    BASE_LEARNING_RATE = 1e-3
    MIN_LEARNING_RATE = 1e-4
    LR_DECAY_RATE = 1e-3 / 1000
    LR_DECAY_STEP = 1
    WEIGHT_DECAY = 1e-2
    GRADIENT_CLIP = 1.0

    # Early stopping
    PATIENCE = 250
    MIN_DELTA = 1e-6

    # Checkpoint settings
    SAVE_EVERY = 50  # Match your existing code
    BEST_MODEL_PATH = os.path.join(output_folder, 'best_model.pth')

    os.makedirs(output_folder, exist_ok=True)

    # ============================================================================
    # DATA LOADING
    # ============================================================================
    print("Loading data...")
    data = np.load('/local/scratch/groves/jax-forgeRL/models/forging_autoencoder/data/test_comb_FOR_large_512_p_degx.npz')
    c_t = data['coords_t']
    c_tp1 = data['coords_tp1']
    steps = data['steps']
    positions = data['positions']
    rotations = data['rotations']

    # Choose action representation
    # actions = np.hstack((steps, positions, rotations))  # Full 8D action
    actions = np.hstack((steps, rotations))  # 2D action - steps and deg_about_x axis
    actions = np.hstack((steps, positions[:,0].reshape(-1,1), rotations)) # 3D action - steps and pos_x and deg_about_x

    print(f"Data shapes:")
    print(f"  coords_t: {c_t.shape}")
    print(f"  coords_tp1: {c_tp1.shape}")
    print(f"  actions: {actions.shape}")


    train_loader, test_loader = GetSingleStepDataLoaders(
    coords_t=c_t,       
    coords_tp1=c_tp1,
    actions=actions,
    batch_size=batch_size
    )

    # ============================================================================
    # MODEL SETUP
    # ============================================================================
    # Infer point_size from data
    point_size = c_t.shape[1]

    print(f"\nInitializing model with point_size={point_size}, latent_size={latent_size}")
    
    # net = model.PCTransitionModel(point_size, latent_size)

    net = model.ImprovedPCTransitionModel(
        point_size=point_size,
        latent_size=latent_size, 
        dropout=dropout   
    )

    # GPU setup
    if use_GPU and torch.cuda.is_available():
        device = torch.device("cuda:0")
        # if torch.cuda.device_count() > 1:
        #     print(f"Using {torch.cuda.device_count()} GPUs")
        #     net = torch.nn.DataParallel(net)
        # else:
        print("Using single GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    net = net.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # ============================================================================
    # OPTIMIZER & SCHEDULER (IMPROVED)
    # ============================================================================
    optimizer = optim.Adam(
        net.parameters(),
        lr=BASE_LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    # optimizer,
    # T_0=100,          # Epochs in first cycle
    # T_mult=1,        # Multiply cycle length after restart (1 = same length)
    # eta_min=MIN_LEARNING_RATE     # Minimum learning rate
    # )

    # Initialize wrapper
    training_wrapper = TrainingWrapper(
        scheduler=scheduler,
        best_model_path=BEST_MODEL_PATH,
        patience=PATIENCE,
        min_delta=MIN_DELTA,
        gradient_clip=GRADIENT_CLIP
    )


    # ============================================================================
    # CONFIGURATION - SET RESUME OPTIONS HERE
    # ============================================================================
    RESUME_TRAINING = False  # Set to True to resume from checkpoint
    RESUME_CHECKPOINT = "./output_cogging/best_model.pth"  # Path to checkpoint
    ADDITIONAL_EPOCHS = 100  # How many more epochs to train


    # Train from scratch
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    train_losses, test_losses = enhanced_train_model(
        train_loader, 
        test_loader, 
        net, 
        epochs, 
        optimizer, 
        device, 
        save_results, 
        output_folder
    )

    # Save final model
    torch.save(net.state_dict(), './transition_resmodel_comb_FOR_large_512.pth')
    print(f"\n✓ Saved final model to ./transition_resmodel_comb_FOR_large_512.pth")

    # Save training history
    if save_results:
        np.savez(
            os.path.join(output_folder, 'training_history.npz'),
            train_losses=train_losses,
            test_losses=test_losses
        )
        print(f"✓ Saved training history to {output_folder}/training_history.npz")

    print("\nAll done! 🎉")

    # Train from checkpoint
    if RESUME_TRAINING and os.path.exists(RESUME_CHECKPOINT):
        # Load checkpoint
        start_epoch, _, resume_best_loss = load_checkpoint(
            RESUME_CHECKPOINT, 
            net, 
            optimizer, 
            scheduler,  # Also restore scheduler state
            device
        )
        
        # Load training history
        history_path = os.path.join(output_folder, 'training_history.npz')
        train_loss_history, test_loss_history = load_training_history(history_path)
        
        # Update total epochs
        total_epochs = start_epoch + ADDITIONAL_EPOCHS
        print(f"Will train for {ADDITIONAL_EPOCHS} more epochs (until epoch {total_epochs})")
    else:
        total_epochs = epochs
        print(f"Starting fresh training for {total_epochs} epochs")