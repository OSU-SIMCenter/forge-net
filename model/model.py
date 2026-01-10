import torch
import torch.nn as nn
import torch.nn.functional as F


class PCTransitionModel(nn.Module):
    def __init__(self, point_size, latent_size, act_dim):
        super(PCTransitionModel, self).__init__()
        
        self.latent_size = int(latent_size / 2)
        self.point_size = point_size
        
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, self.latent_size, 1)

        self.conv4 = torch.nn.Conv1d(3, 64, 1)
        self.conv5 = torch.nn.Conv1d(64, 128, 1)
        self.conv6 = torch.nn.Conv1d(128, self.latent_size, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(self.latent_size)

        # self.bn4 = nn.BatchNorm1d(64)
        # self.bn5 = nn.BatchNorm1d(128)
        # self.bn6 = nn.BatchNorm1d(self.latent_size)

        self.act_fc1 = nn.Linear(act_dim, 64)
        self.act_fc2 = nn.Linear(64, 128)
        self.act_fc3 = nn.Linear(128, self.latent_size)
        
        self.dec1 = nn.Linear(int(self.latent_size*2),256)
        self.dec2 = nn.Linear(256,256)
        self.dec3 = nn.Linear(256,self.point_size*3)

    def state_encoder(self, x): 
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.latent_size)
        return x

    def action_encoder(self, a):
        a = F.relu(self.act_fc1(a))
        a = F.relu(self.act_fc2(a))
        a = self.act_fc3(a)
        return a
        
    # def action_encoder(self, x):
    #     x = F.relu(self.bn4(self.conv4(x)))
    #     x = F.relu(self.bn5(self.conv5(x)))
    #     x = self.bn6(self.conv6(x))
    #     x = torch.max(x, 2, keepdim=True)[0]
    #     x = x.view(-1, self.latent_size)
    #     return x
        
    def decoder(self, x):
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = self.dec3(x)
        return x.view(-1, self.point_size, 3)
    
    def forward(self, x_t, a_t):
        x_l = self.state_encoder(x_t)
        a_l = self.action_encoder(a_t)
        l = torch.cat([x_l, a_l], dim=1)
        delta = self.decoder(l)
        return delta


class ResPCTransitionModel(nn.Module):
    """
   Point Cloud Transition Model with:
    - Deeper architecture (5-layer encoder, 4-layer action encoder, 4-layer decoder)
    - Residual connections throughout
    - Batch normalization on all layers
    - Strategic dropout for regularization
    - Proper weight initialization
    """
    def __init__(self, point_size, latent_size, act_dim, dropout=0.3):
        super(ResPCTransitionModel, self).__init__()
        self.latent_size = int(latent_size / 2)
        self.point_size = point_size
        self.dropout = dropout
        
        # ====================================================================
        # STATE ENCODER (5-layer PointNet with Residuals)
        # ====================================================================
        # Layer 1: 3 -> 64
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        
        # Residual Block 1: 64 -> 64
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.bn2 = nn.BatchNorm1d(64)
        
        # Residual Block 2: 64 -> 128 (dimension change)
        self.conv3 = nn.Conv1d(64, 128, 1)
        self.bn3 = nn.BatchNorm1d(128)
        self.proj1 = nn.Conv1d(64, 128, 1)  # Projection for skip connection
        self.bn_proj1 = nn.BatchNorm1d(128)
        
        # Residual Block 3: 128 -> 256 (dimension change)
        self.conv4 = nn.Conv1d(128, 256, 1)
        self.bn4 = nn.BatchNorm1d(256)
        self.proj2 = nn.Conv1d(128, 256, 1)  # Projection for skip connection
        self.bn_proj2 = nn.BatchNorm1d(256)
        
        # Final encoding: 256 -> latent_size
        self.conv5 = nn.Conv1d(256, self.latent_size, 1)
        self.bn5 = nn.BatchNorm1d(self.latent_size)
        
        # Dropout after max pooling
        self.dropout_state = nn.Dropout(dropout)
        
        # ====================================================================
        # ACTION ENCODER (4-layer with Residuals)
        # ====================================================================
        # Layer 1: 1 -> 64
        self.act_fc1 = nn.Linear(act_dim, 64)
        self.act_bn1 = nn.BatchNorm1d(64)
        
        # Residual Block 1: 64 -> 64
        self.act_fc2 = nn.Linear(64, 64)
        self.act_bn2 = nn.BatchNorm1d(64)
        
        # Residual Block 2: 64 -> 128 (dimension change)
        self.act_fc3 = nn.Linear(64, 128)
        self.act_bn3 = nn.BatchNorm1d(128)
        self.act_proj1 = nn.Linear(64, 128)  # Projection for skip connection
        self.act_bn_proj1 = nn.BatchNorm1d(128)
        
        # Residual Block 3: 128 -> 256 (dimension change)
        self.act_fc4 = nn.Linear(128, 256)
        self.act_bn4 = nn.BatchNorm1d(256)
        self.act_proj2 = nn.Linear(128, 256)  # Projection for skip connection
        self.act_bn_proj2 = nn.BatchNorm1d(256)
        
        # Final layer: 256 -> latent_size
        self.act_fc5 = nn.Linear(256, self.latent_size)
        
        # Dropout
        self.dropout_action = nn.Dropout(dropout)
        
        # ====================================================================
        # DECODER (4-layer with Residuals)
        # ====================================================================
        # Layer 1: latent_size*2 -> 512
        self.dec1 = nn.Linear(self.latent_size * 2, 512)
        self.dec_bn1 = nn.BatchNorm1d(512)
        
        # Residual Block 1: 512 -> 512
        self.dec2 = nn.Linear(512, 512)
        self.dec_bn2 = nn.BatchNorm1d(512)
        
        # Residual Block 2: 512 -> 256 (dimension change)
        self.dec3 = nn.Linear(512, 256)
        self.dec_bn3 = nn.BatchNorm1d(256)
        self.dec_proj1 = nn.Linear(512, 256)  # Projection for skip connection
        self.dec_bn_proj1 = nn.BatchNorm1d(256)
        
        # Final layer: 256 -> point_size*3
        self.dec4 = nn.Linear(256, self.point_size * 3)
        
        # Dropout (lighter in decoder)
        self.dropout_dec = nn.Dropout(dropout * 0.5)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier/He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def state_encoder(self, x):
        """
        5-layer encoder with residual connections
        Input: (B, 3, N) where N = num_points
        Output: (B, latent_size)
        """
        # Layer 1: 3 -> 64
        x = F.relu(self.bn1(self.conv1(x)))  # (B, 64, N)
        
        # Residual Block 1: 64 -> 64 (same dimension)
        identity = x
        out = F.relu(self.bn2(self.conv2(x)))
        x = out + identity  # Residual connection
        
        # Residual Block 2: 64 -> 128 (dimension change)
        identity = self.bn_proj1(self.proj1(x))  # Project identity to match dimensions
        out = F.relu(self.bn3(self.conv3(x)))
        x = out + identity  # Residual connection
        
        # Residual Block 3: 128 -> 256 (dimension change)
        identity = self.bn_proj2(self.proj2(x))  # Project identity to match dimensions
        out = F.relu(self.bn4(self.conv4(x)))
        x = out + identity  # Residual connection
        
        # Final encoding: 256 -> latent_size
        x = self.bn5(self.conv5(x))  # (B, latent_size, N)
        
        # Global max pooling
        x = torch.max(x, 2, keepdim=False)[0]  # (B, latent_size)
        
        # Dropout
        x = self.dropout_state(x)
        
        return x
    
    def action_encoder(self, a):
        """
        4-layer action encoder with residual connections
        Input: (B, 1) or (B,) action vector
        Output: (B, latent_size)
        """
        # Ensure proper shape
        if len(a.shape) == 1:
            a = a.unsqueeze(1)
        
        # Layer 1: 1 -> 64
        a = F.relu(self.act_bn1(self.act_fc1(a)))  # (B, 64)
        
        # Residual Block 1: 64 -> 64 (same dimension)
        identity = a
        out = F.relu(self.act_bn2(self.act_fc2(a)))
        a = out + identity  # Residual connection
        
        a = self.dropout_action(a)
        
        # Residual Block 2: 64 -> 128 (dimension change)
        identity = self.act_bn_proj1(self.act_proj1(a))  # Project identity
        out = F.relu(self.act_bn3(self.act_fc3(a)))
        a = out + identity  # Residual connection
        
        a = self.dropout_action(a)
        
        # Residual Block 3: 128 -> 256 (dimension change)
        identity = self.act_bn_proj2(self.act_proj2(a))  # Project identity
        out = F.relu(self.act_bn4(self.act_fc4(a)))
        a = out + identity  # Residual connection
        
        a = self.dropout_action(a)
        
        # Final layer: 256 -> latent_size
        a = self.act_fc5(a)  # (B, latent_size)
        
        return a
    
    def decoder(self, x):
        """
        4-layer decoder with residual connections
        Input: (B, latent_size * 2)
        Output: (B, point_size, 3)
        """
        # Layer 1: latent_size*2 -> 512
        x = F.relu(self.dec_bn1(self.dec1(x)))  # (B, 512)
        x = self.dropout_dec(x)
        
        # Residual Block 1: 512 -> 512 (same dimension)
        identity = x
        out = F.relu(self.dec_bn2(self.dec2(x)))
        x = out + identity  # Residual connection
        
        x = self.dropout_dec(x)
        
        # Residual Block 2: 512 -> 256 (dimension change)
        identity = self.dec_bn_proj1(self.dec_proj1(x))  # Project identity
        out = F.relu(self.dec_bn3(self.dec3(x)))
        x = out + identity  # Residual connection
        
        x = self.dropout_dec(x)
        
        # Final layer: 256 -> point_size*3
        x = self.dec4(x)  # (B, point_size*3)
        
        # Reshape to point cloud
        return x.view(-1, self.point_size, 3)
    
    def forward(self, x_t, a_t):
        """
        Forward pass
        Args:
            x_t: (B, 3, N) current point cloud state
            a_t: (B, 1) or (B,) action
        Returns:
            delta: (B, N, 3) predicted next point cloud state
        """
        # Encode state and action
        x_l = self.state_encoder(x_t)      # (B, latent_size)
        a_l = self.action_encoder(a_t)     # (B, latent_size)
        
        # Concatenate latent representations
        l = torch.cat([x_l, a_l], dim=1)   # (B, latent_size*2)
        
        # Decode to next state
        delta = self.decoder(l)            # (B, N, 3)
        
        return delta
