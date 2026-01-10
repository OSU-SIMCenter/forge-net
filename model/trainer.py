import os
import time
import torch
import torch.optim as optim
from model.model import * 
from utils.plotting import * 
from utils.utils import * 




from torch.utils.tensorboard import SummaryWriter

class Trainer:
   
    def __init__(self, config, train_loader, test_loader, resume_epoch=0, resume_best_loss=None):
        
        self.config = config
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.output_folder = self.config["output_folder"]
        
        # Early stopping tracking
        self.early_stop_best_loss = resume_best_loss  # Initialize with resumed value
        self.counter = 0
        self.early_stop_best_epoch = resume_epoch if resume_best_loss else 0
        self.early_stop = self.config["run"]["early_stop"]
        self.patience = self.config["run"]["patience"]
        self.min_delta = self.config["run"]["min_delta"]
        self.gradient_clip = self.config["network"]["gradient_clip"]

        # Best model tracking
        self.best_loss = resume_best_loss if resume_best_loss else float('inf')
        self.best_epoch = resume_epoch if resume_best_loss else 0
        
        # Tensorboard
        self.writer = SummaryWriter(log_dir=os.path.join(self.output_folder, 'logs'))
        
        if resume_epoch > 0:
            print(f"Resuming training wrapper from epoch {resume_epoch}")
            print(f"  Best loss so far: {self.best_loss:.6f}")
        
        self._make_network()
    
    def _make_network(self):
  
        point_size = self.config["datasets"]["points_per_mesh"]
        latent_size = self.config["network"]["latent_size"]
        model_type = self.config["network"]["model_type"]
        use_gpu = self.config["network"]["use_gpu"]
        print(f"\nInitializing model with point_size={point_size}, latent_size={latent_size}")
        
        if model_type =="ResNetPointAE":
            dropout = self.config["network"]["dropout"]
            self.net = ResPCTransitionModel(
            point_size=point_size,
            latent_size=latent_size, 
            dropout=dropout) 
        
        else:
            self.PCTransitionModel(point_size=point_size, latent_size=latent_size)

        if use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            print("Using single GPU")
        else:
            self.device = torch.device("cpu")
            print("Using CPU")

        self.net = self.net.to(self.device)

        total_params = sum(p.numel() for p in self.net.parameters())
        trainable_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        self.optimizer = optim.Adam(
            self.net.parameters(),
            lr=self.config["network"]["optimizer"]["base_learning_rate"],
            weight_decay=self.config["network"]["optimizer"]["weight_decay"]
        )

        #TODO decide on final scheduler configuration
        self.scheduler = optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.05, total_iters=40)

    def load(self, model_path):

        state_dict = torch.load(model_path, weights_only=True)
        plot_network_weights(state_dict)

        self.net.load_state_dict(state_dict)
        self.net.eval()

    def on_epoch_end(self, epoch, train_loss, test_loss, net):
 
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
        self.scheduler.step()
        
        # Clip learning rate
        for param_group in self.scheduler.optimizer.param_groups:
            param_group['lr'] = max(param_group['lr'], MIN_LEARNING_RATE)

    def train(self):

        if self.train_loss_list is None:
            self.train_loss_list = []
        if self.test_loss_list is None:
            self.test_loss_list = []
        
        print("\n" + "="*80)
        if self.start_epoch > 0:
            print(f"RESUMING TRAINING FROM EPOCH {self.start_epoch}")
            print(f"Previous training: {len(self.train_loss_list)} epochs")
        else:
            print("STARTING TRAINING")
        print("="*80 + "\n")
        
        for i in range(self.start_epoch, self.epochs):  # Note: range starts from start_epoch
            startTime = time.time()
            
            self.train_loss = self.train_epoch(self.train_loader, 
                                               self.net, 
                                               self.optimizer, 
                                               self.device)
            
            self.train_loss_list.append(self.train_loss)
            self.test_loss = self.test_epoch(self.test_loader, 
                                             self.net, 
                                             self.device)
            
            self.test_loss_list.append(self.test_loss)
            
            epoch_time = time.time() - startTime
            
            writeString = (
                f"epoch {i} train loss: {self.train_loss:.7f} test loss: {self.test_loss:.7f} "
                f"epoch time: {epoch_time:.2f}s"
            )
            print(writeString)
            
            # Enhanced features
            self.training_wrapper.on_epoch_end(i, self.train_loss, self.test_loss, self.net)
            
            plot_loss(self.train_loss_list, self.test_loss_list, self.output_folder, save_results=True)
            
            # Visualization checkpoints
            if i % self.config["network"]["save_every"] == 0:
                test_samples, test_actions, test_deltas, test_samples_next = next(iter(self.test_loader))
                loss, test_output = self.test_batch(test_samples, test_actions, test_deltas, self.net, self.device)
                plotPCbatch(
                    test_samples,
                    test_samples_next[:, -1, :, :],
                    test_output,
                    show=False,
                    save=True,
                    name=(self.output_folder + f"epoch_{i}")
                )
            
            # Early stopping check
            if self.early_stop:
                print(f"\nStopping training at epoch {i}")
                break
        
        self.close()
        
        # Final summary
        print("\n" + "="*80)
        print("TRAINING COMPLETE")
        print("="*80)
        print(f"Best test loss: {self.best_loss:.7f} at epoch {self.best_epoch}")
        print(f"Final train loss: {self.train_loss_list[-1]:.7f}")
        print(f"Final test loss: {self.test_loss_list[-1]:.7f}")
    
    def close(self):
        self.writer.close()