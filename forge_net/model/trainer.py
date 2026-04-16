from pathlib import Path
import time
import torch
import torch.nn.functional as F
import torch.optim as optim

from forge_net.model.model import ForgeNet

from forge_net.utils.plotting import plot_loss
# from forge_net.utils.common import 

def l1_penalty(net):
    l1_loss = 0.0
    for param in net.parameters():
        l1_loss += torch.sum(torch.abs(param))
    
    return(l1_loss)

from torch.utils.tensorboard import SummaryWriter

class ForgeNetTrainer:
   
    def __init__(self, config, train_loader=None, test_loader=None, 
    resume_epoch=0, resume_best_loss=None, log_to_tb=True):
        
        self.config = config
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.output_folder = Path(self.config["run"]["run_folder"])
        self.best_model_path = self.output_folder / "best_model.pth"
        
        self.train_loss_list = []
        self.test_loss_list = []
        self.start_epoch = 0

        self.early_stop_best_loss = resume_best_loss  # Initialize with resumed value
        self.counter = 0
        self.early_stop_best_epoch = resume_epoch if resume_best_loss else 0
        self.early_stop = self.config["run"]["early_stop"]
        self.patience = self.config["run"]["patience"]
        self.min_delta = self.config["run"]["min_delta"]
        self.gradient_clip = self.config["network"]["gradient_clip"]
        self.min_learning_rate = self.config["network"]["optimizer"]["min_learning_rate"]


        self.num_epochs = self.config["run"]["num_epochs"]
        self.best_loss = resume_best_loss if resume_best_loss else float('inf')
        self.best_epoch = resume_epoch if resume_best_loss else 0
        
        # Tensorboard
        if log_to_tb:
            self.writer = SummaryWriter(log_dir=self.output_folder / 'logs')
        
        if resume_epoch > 0:
            print(f"Resuming training wrapper from epoch {resume_epoch}")
            print(f"  Best loss so far: {self.best_loss:.6f}")
        
        self._make_network()
        self.loss_fn = self._get_loss_fn()
    
    def _make_network(self):
  
        sample_batch = next(iter(self.train_loader))
        point_size = sample_batch[0].shape[1] # get the states shape
        self.config["network"]["point_size"] = point_size #include in the output config

        latent_size = self.config["network"]["latent_size"]
        action_dims = self.config["network"]["action_dims"] 
        use_gpu = self.config["network"]["use_gpu"]   
        dropout = self.config["network"]["dropout"]
        use_res = self.config["network"]["use_res"]

        print(f"\nInitializing model with point_size={point_size}, latent_size={latent_size}")
        
        self.net = ForgeNet(point_size=point_size,
                            latent_size=latent_size,
                            action_dims=action_dims,
                            dropout=dropout,
                            use_res=use_res)

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

        # #TODO decide on final scheduler configuration
        # self.scheduler = optim.lr_scheduler.LinearLR(self.optimizer, 
        #                                              start_factor=1.0, 
        #                                              end_factor=0.05, 
        #                                              total_iters=40)
    
        warmup_epochs = 20
        from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LambdaLR, LinearLR
        warmup = LambdaLR(self.optimizer, lambda e: (e + 1) / warmup_epochs)
        cosine = CosineAnnealingLR(self.optimizer, T_max=90, eta_min=1e-6)
        factor = self.config["network"]["optimizer"]["min_learning_rate"]/ \
                    self.config["network"]["optimizer"]["base_learning_rate"]
        total_steps = self.config["run"]["num_epochs"]
        linear = LinearLR(self.optimizer, start_factor=1.0, end_factor=factor, total_iters=total_steps)
        self.scheduler = SequentialLR(self.optimizer, [warmup, linear], milestones=[warmup_epochs])

    def load(self, model_path):

        self.state_dict = torch.load(model_path, weights_only=True)["model_state_dict"]
        self.net.load_state_dict(self.state_dict)
        self.net.eval()
        return (self.state_dict)

    def on_epoch_end(self, epoch, train_loss, test_loss, net):
 
        current_lr = self.scheduler.get_last_lr()
        print(current_lr)
        self.writer.add_scalar('Loss/train', train_loss, epoch)
        self.writer.add_scalar('Loss/test', test_loss, epoch)
        self.writer.add_scalar('Learning_Rate', current_lr[0], epoch)
        
        print(f"  LR: {current_lr[0]:.6f}")
        
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
            param_group['lr'] = max(param_group['lr'], self.min_learning_rate)

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
        
        for i in range(self.start_epoch, self.num_epochs):  # Note: range starts from start_epoch
            startTime = time.time()
            
            self.train_loss = self.train_epoch()
            self.train_loss_list.append(self.train_loss)
            self.test_loss = self.test_epoch()
            self.test_loss_list.append(self.test_loss)
            
            epoch_time = time.time() - startTime
        
            write_string = (
                f"epoch {i} train loss: {self.train_loss:.7f} test loss: {self.test_loss:.7f} "
                f"epoch time: {epoch_time:.2f}s"
            )
            print(write_string)
            
            self.on_epoch_end(i, self.train_loss, self.test_loss, self.net)
            
            plot_loss(self.train_loss_list, self.test_loss_list, 
                      write_string, self.output_folder, save_results=True)
            
            # Visualization checkpoints
            if i % self.config["run"]["save_every"] == 0:
                test_samples, test_actions, test_deltas, test_samples_next = next(iter(self.test_loader))
                loss, test_output = self.test_batch(test_samples, test_actions, test_deltas, test_samples_next)
            
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

    def train_epoch(self):
        epoch_loss = 0
        
        for i, (x_t, a, delta_t, x_tp1) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            x_t = x_t.to(self.device) # [B, N, 3]
            x_t_perm = x_t.permute(0, 2, 1) # [B, 3, N]
            x_tp1 = x_tp1.to(self.device)
            a = a.to(self.device) # [B, 1, A]
            delta_t = delta_t.to(self.device) # [B, 1, N, 3]
            delta_hat = self.net(x_t_perm, a[:, 0, :]) # [B, N, 3]
            delta_gt = delta_t[:, 0, :, :] # [B, N, 3]
            loss = self.loss_fn(delta_hat, delta_gt)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()

        return epoch_loss/(i+1)

    def test_batch(self, x_t, a, delta_t, x_tp1):
        with torch.no_grad():
            x_t = x_t.to(self.device)
            x_tp1 = x_tp1.to(self.device)
            a = a.to(self.device)
            delta_t = delta_t.to(self.device)
            x_t_perm = x_t.permute(0, 2, 1)
            delta_hat = self.net(x_t_perm, a[:, 0, :])
            delta_gt = delta_t[:, 0, :, :]
            x_tp1_hat = x_t + delta_hat / 100
            loss = self.loss_fn(delta_hat, delta_gt)
            
        return loss.item(), x_tp1_hat.cpu()

    def test_epoch(self):
        with torch.no_grad():
            epoch_loss = 0
            for i, (x_t, a, delta_t, x_tp1) in enumerate(self.test_loader):
                loss, _ = self.test_batch(x_t, a, delta_t, x_tp1)
                epoch_loss += loss
        return epoch_loss/(i+1)
    
    def _get_loss_fn(self):
        
        if self.config["network"]["loss"] == "mse":
            print("Using MSE loss function")
            return lambda delta_pred, delta_gt: torch.mean((delta_pred - delta_gt) ** 2)
        
        elif self.config["network"]["loss"] == "chamfer":
            print("Using Chamfer discrepancy loss function")
            from pytorch3d.loss import chamfer_distance #original implementation uses a chamfer distance
            return lambda delta_pred, delta_gt: chamfer_distance(delta_gt, delta_pred)[0]
        
        elif self.config["network"]["loss"] == "wsd": 
            print("Using Adaptive Wasserstein Distance loss function")
            from forge_net.loss.loss import AdaptiveSlicedWasserstein 
            return lambda delta_pred, delta_gt: \
                    torch.mean(AdaptiveSlicedWasserstein(device=self.device)(delta_pred,delta_gt))
        
        else:
            raise ValueError(f"Unknown loss: {self.config['network']['loss']}")
    

    def close(self):
        self.writer.close()

