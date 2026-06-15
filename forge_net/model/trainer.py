import time
from pathlib import Path
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax
import orbax.checkpoint as ocp
# from torch.utils.tensorboard import SummaryWriter
import numpy as np

_ARCH_MODULES = {
    "large": "forge_net.model.model",
    "small": "forge_net.model.small_model",
}

# We need a custom TrainState to handle BatchNorm statistics
class TrainState(train_state.TrainState):
    batch_stats: dict

class ForgeNetTrainer:
   
    def __init__(self, config, train_loader=None, test_loader=None, 
                 resume_epoch=0, resume_best_loss=None, log_to_tb=True, seed=42):
        
        self.config = config
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.output_folder = Path(self.config["run"]["run_folder"])
        
        # Orbax checkpoint manager setup
        self.ckpt_dir = self.output_folder / "checkpoints"
        self.ckpt_manager = ocp.CheckpointManager(
            self.ckpt_dir.resolve(),
            item_names=('state',),
            options=ocp.CheckpointManagerOptions(max_to_keep=1, create=True)
        )
        
        self.train_loss_list = []
        self.test_loss_list = []
        self.start_epoch = 0
        self.seed = seed

        self.early_stop_best_loss = resume_best_loss
        self.counter = 0
        self.early_stop_best_epoch = resume_epoch if resume_best_loss else 0
        self.early_stop = self.config["run"]["early_stop"]
        self.patience = self.config["run"]["patience"]
        self.min_delta = self.config["run"]["min_delta"]
        self.gradient_clip = self.config["network"]["gradient_clip"]
        self.base_lr = self.config["network"]["optimizer"]["base_learning_rate"]
        self.min_lr = self.config["network"]["optimizer"]["min_learning_rate"]
        self.num_epochs = self.config["run"]["num_epochs"]

        self.best_loss = resume_best_loss if resume_best_loss else float('inf')
        self.best_epoch = resume_epoch if resume_best_loss else 0
        
        if log_to_tb:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=self.output_folder / 'logs')
        
        if resume_epoch > 0:
            print(f"Resuming training wrapper from epoch {resume_epoch}")
            print(f"  Best loss so far: {self.best_loss:.6f}")
        
        self.loss_fn = self._get_loss_fn()
        self._make_network()
        
        # Load checkpoint if resuming
        if resume_epoch > 0:
            self.load()
    
    def _make_network(self):
        # 1. Peek at data to get shapes
        sample_batch = next(iter(self.train_loader()))
        x_dummy = sample_batch[0]
        a_dummy = sample_batch[1][:, 0, :]
        point_size = x_dummy.shape[1]
        
        self.config["network"]["point_size"] = point_size

        print(f"\nInitializing model with point_size={point_size}, latent_size={self.config['network']['latent_size']}")
        
        # 2. Instantiate Flax Model
        arch = self.config["network"].get("arch", "large")
        module_path = _ARCH_MODULES.get(arch)
        if module_path is None:
            raise ValueError(f"Unknown arch '{arch}'. Choose from: {list(_ARCH_MODULES)}")

        import importlib
        ForgeNet = importlib.import_module(module_path).ForgeNet

        net_kwargs = dict(
            latent_size=self.config["network"]["latent_size"],
            action_dims=self.config["network"]["action_dims"],
            dropout=self.config["network"]["dropout"],
        )
        if arch == "large":
            net_kwargs["use_res"] = self.config["network"]["use_res"]

        print(f"Architecture: {arch} ({module_path})")
        self.net = ForgeNet(**net_kwargs)

        # 3. Initialize Variables
        rng = jax.random.PRNGKey(self.seed)
        rng, init_rng = jax.random.split(rng)
        
        variables = self.net.init(init_rng, x_dummy, a_dummy, train=False)
        params = variables['params']
        batch_stats = variables.get('batch_stats', {})

        # Parameter counting
        flat_params = jax.tree_util.tree_leaves(params)
        total_params = sum(p.size for p in flat_params)
        print(f"Total trainable parameters: {total_params:,}")

        # 4. Optax Scheduler & Optimizer
        # Estimate steps per epoch (assuming dataloader returns total length)
        steps_per_epoch = 1000 # FALLBACK: Replace this with len(self.train_loader()) if your wrapper supports len()
        
        warmup_epochs = 20
        warmup_steps = warmup_epochs * steps_per_epoch
        total_steps = self.num_epochs * steps_per_epoch
        
        # Warmup (0 to base_lr) then Linear decay (base_lr to min_lr)
        warmup_fn = optax.linear_schedule(init_value=0.0, end_value=self.base_lr, transition_steps=warmup_steps)
        decay_fn = optax.linear_schedule(init_value=self.base_lr, end_value=self.min_lr, transition_steps=total_steps - warmup_steps)
        self.lr_schedule = optax.join_schedules([warmup_fn, decay_fn], [warmup_steps])

        # Chain gradient clipping, weight decay, and the Adam optimizer
        optimizer = optax.chain(
            optax.clip_by_global_norm(self.gradient_clip),
            optax.adamw(learning_rate=self.lr_schedule, weight_decay=self.config["network"]["optimizer"]["weight_decay"])
        )

        # 5. Create TrainState
        self.state = TrainState.create(
            apply_fn=self.net.apply,
            params=params,
            tx=optimizer,
            batch_stats=batch_stats
        )

        # 6. Define JIT-compiled Step Functions
        # We define them here so they capture `self.loss_fn` without needing `self` passed into JAX
        loss_fn = self.loss_fn

        @jax.jit
        def train_step(state, x_t, a, delta_gt, dropout_rng):
            def compute_loss(params):
                # Forward pass needs both params and batch_stats
                variables = {'params': params, 'batch_stats': state.batch_stats}
                
                delta_hat, mutated_vars = state.apply_fn(
                    variables, x_t, a, train=True, 
                    mutable=['batch_stats'], rngs={'dropout': dropout_rng}
                )
                loss = loss_fn(delta_hat, delta_gt)
                return loss, mutated_vars

            # Calculate gradients
            (loss, mutated_vars), grads = jax.value_and_grad(compute_loss, has_aux=True)(state.params)
            
            # Apply updates
            state = state.apply_gradients(grads=grads)
            # Update batch stats
            state = state.replace(batch_stats=mutated_vars['batch_stats'])
            
            return state, loss

        @jax.jit
        def test_step(state, x_t, a, delta_gt):
            variables = {'params': state.params, 'batch_stats': state.batch_stats}
            delta_hat = state.apply_fn(variables, x_t, a, train=False)
            loss = loss_fn(delta_hat, delta_gt)
            
            # For visualization/metrics
            x_tp1_hat = x_t + (delta_hat / 100.0)
            return loss, x_tp1_hat

        # Bind these to the instance
        self._train_step = train_step
        self._test_step = test_step
        self.rng = rng

    def load(self, model_path=None):
        if model_path is not None:
            model_path = Path(model_path)
            # Extract the step (e.g., '248') from the end of the path
            try:
                step = int(model_path.name)
                
                # 1. Wrap the TrainState in StandardRestore
                # 2. Wrap that in Composite with the key 'state'
                restore_args = ocp.args.Composite(
                    state=ocp.args.StandardRestore(self.state)
                )
                
                # The manager needs the step index, not the full path
                restored = self.ckpt_manager.restore(step, args=restore_args)
                self.state = restored['state']
                
                print(f"Successfully loaded model step {step}")
            except (ValueError, TypeError) as e:
                print(f"Failed to load specific path: {e}. Attempting latest...")
                return self.load(None)
        else:
            step = self.ckpt_manager.latest_step()
            if step is not None:
                restore_args = ocp.args.Composite(
                    state=ocp.args.StandardRestore(self.state)
                )
                restored = self.ckpt_manager.restore(step, args=restore_args)
                self.state = restored['state']
                print(f"Loaded latest checkpoint (epoch {step})")
                
        return self.state

    def save(self, epoch):
 
        print(f"  ✓ Saved best model (test_loss: {self.best_loss:.6f})")
        save_args = ocp.args.Composite(
        state=ocp.args.StandardSave(self.state)
        )
        self.ckpt_manager.save(epoch, args=save_args)
        print(f"  ✓ Saved best model (test_loss: {self.best_loss:.6f})")

    def on_epoch_end(self, epoch, train_loss, test_loss):
        # Get current learning rate based on total steps taken
        current_step = self.state.step
        current_lr = float(self.lr_schedule(current_step))
        
        self.writer.add_scalar('Loss/train', train_loss, epoch)
        self.writer.add_scalar('Loss/test', test_loss, epoch)
        self.writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        print(f"  LR: {current_lr:.6f}")
        
        if test_loss < self.best_loss:
            self.best_loss = test_loss
            self.best_epoch = epoch
            self.save(epoch)
        
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

    def train(self):
        print("\n" + "="*80)
        if self.start_epoch > 0:
            print(f"RESUMING TRAINING FROM EPOCH {self.start_epoch}")
            print(f"Previous training: {len(self.train_loss_list)} epochs")
        else:
            print("STARTING TRAINING")
        print("="*80 + "\n")
        
        for i in range(self.start_epoch, self.num_epochs):
            startTime = time.time()
            
            self.train_loss = self.train_epoch()
            self.train_loss_list.append(self.train_loss)
            
            self.test_loss, test_output = self.test_epoch()
            self.test_loss_list.append(self.test_loss)
            
            epoch_time = time.time() - startTime
        
            write_string = (
                f"epoch {i} train loss: {self.train_loss:.7f} test loss: {self.test_loss:.7f} "
                f"epoch time: {epoch_time:.2f}s"
            )
            print(write_string)
            
            self.on_epoch_end(i, self.train_loss, self.test_loss)
            self.plot_loss()

            if self.early_stop:
                print(f"\nStopping training at epoch {i}")
                break
        
        self.close()
        
        print("\n" + "="*80)
        print("TRAINING COMPLETE")
        print("="*80)
        print(f"Best test loss: {self.best_loss:.7f} at epoch {self.best_epoch}")

    def train_epoch(self):
        epoch_loss = 0
        batch_count = 0
        
        for x_t, a, delta_t, x_tp1 in self.train_loader():
            # Generate a new random key for dropout for this specific step
            self.rng, dropout_rng = jax.random.split(self.rng)
            
            # Action and Ground truth parsing
            a_input = a[:, 0, :] 
            delta_gt = delta_t[:, 0, :, :] 

            # Note: JAX arrays don't need permuting. Native shape is (B, N, 3)
            self.state, loss = self._train_step(self.state, x_t, a_input, delta_gt, dropout_rng)
            
            # Convert JAX array loss back to float
            epoch_loss += loss.item()
            batch_count += 1

        return epoch_loss / max(1, batch_count)

    def test_epoch(self):
        epoch_loss = 0
        batch_count = 0
        latest_output = None
        
        for x_t, a, delta_t, x_tp1 in self.test_loader():
            a_input = a[:, 0, :]
            delta_gt = delta_t[:, 0, :, :]
            
            loss, x_tp1_hat = self._test_step(self.state, x_t, a_input, delta_gt)
            
            epoch_loss += loss.item()
            batch_count += 1
            
            # Save the last batch's output for visualization mapping to `test_batch` behavior
            if latest_output is None:
                # Use jax.device_get to copy JAX arrays back to standard NumPy CPU arrays
                latest_output = jax.device_get(x_tp1_hat)

        return epoch_loss / max(1, batch_count), latest_output
    
    def predict(self, x_t, a):
        """Simple inference helper"""
        variables = {'params': self.state.params, 'batch_stats': self.state.batch_stats}
        # We call apply with train=False to disable Dropout and use Batchnorm global statistics
        return self.net.apply(variables, x_t, a, train=False)

    def _get_loss_fn(self):
        if self.config["network"]["loss"] == "mse":
            print("Using MSE loss function")
            # JAX native implementation
            return lambda delta_pred, delta_gt: jnp.mean((delta_pred - delta_gt) ** 2)
        
        elif self.config["network"]["loss"] == "chamfer":
            # WARNING: PyTorch3D cannot run inside JAX. 
            print("CRITICAL: You must rewrite Chamfer distance in JAX. Using MSE as fallback.")
            # return custom_jax_chamfer_distance
            return lambda delta_pred, delta_gt: jnp.mean((delta_pred - delta_gt) ** 2)
            
        elif self.config["network"]["loss"] == "wsd": 
            # WARNING: Custom PyTorch loss cannot run inside JAX.
            print("CRITICAL: You must rewrite Adaptive WSD in JAX. Using MSE as fallback.")
            # return custom_jax_wsd_distance
            return lambda delta_pred, delta_gt: jnp.mean((delta_pred - delta_gt) ** 2)
        else:
            raise ValueError(f"Unknown loss: {self.config['network']['loss']}")

    def plot_loss(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        epochs = range(self.start_epoch, self.start_epoch + len(self.train_loss_list))
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(epochs, self.train_loss_list, label='Train Loss')
        ax.plot(epochs, self.test_loss_list, label='Test Loss')

        if self.best_epoch is not None and self.best_loss < float('inf'):
            ax.axvline(x=self.best_epoch, color='green', linestyle='--', alpha=0.6, label=f'Best epoch ({self.best_epoch})')

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Test Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        save_path = self.output_folder / 'loss_curve.png'
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def close(self):
        self.writer.close()
        self.ckpt_manager.wait_until_finished()
