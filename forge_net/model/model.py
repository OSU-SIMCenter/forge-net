import torch
import torch.nn as nn
import torch.nn.functional as F

class ForgeNet(nn.Module):
    def __init__(self, point_size, latent_size, action_dims, dropout=0.3, use_res=True):
        super(ForgeNet, self).__init__()
        self.latent_size = int(latent_size / 2)
        self.point_size = point_size
        self.dropout = dropout
        self.use_res = use_res
        
        # ====================================================================
        # STATE ENCODER
        # ====================================================================
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.conv3 = nn.Conv1d(64, 128, 1)
        self.bn3 = nn.BatchNorm1d(128)
        self.proj1 = nn.Conv1d(64, 128, 1)
        self.bn_proj1 = nn.BatchNorm1d(128)
        
        self.conv4 = nn.Conv1d(128, 256, 1)
        self.bn4 = nn.BatchNorm1d(256)
        self.proj2 = nn.Conv1d(128, 256, 1)
        self.bn_proj2 = nn.BatchNorm1d(256)
        
        self.conv5 = nn.Conv1d(256, self.latent_size, 1)
        self.bn5 = nn.BatchNorm1d(self.latent_size)
        self.dropout_state = nn.Dropout(dropout)
        
        # ====================================================================
        # ACTION ENCODER
        # ====================================================================
        self.act_fc1 = nn.Linear(action_dims, 64)
        self.act_bn1 = nn.BatchNorm1d(64)
        self.act_fc2 = nn.Linear(64, 64)
        self.act_bn2 = nn.BatchNorm1d(64)
        self.act_fc3 = nn.Linear(64, 128)
        self.act_bn3 = nn.BatchNorm1d(128)
        self.act_proj1 = nn.Linear(64, 128)
        self.act_bn_proj1 = nn.BatchNorm1d(128)
        self.act_fc4 = nn.Linear(128, 256)
        self.act_bn4 = nn.BatchNorm1d(256)
        self.act_proj2 = nn.Linear(128, 256)
        self.act_bn_proj2 = nn.BatchNorm1d(256)
        self.act_fc5 = nn.Linear(256, self.latent_size)
        self.dropout_action = nn.Dropout(dropout)
        
        # ====================================================================
        # POINT-WISE DECODER
        # Uses Conv1d (kernel=1) to process points independently
        # Input size: (State Latent + Action Latent + 3 Original XYZ Coords)
        # ====================================================================
        decoder_in_dim = (self.latent_size * 2) + 3
        
        self.dec_conv1 = nn.Conv1d(decoder_in_dim, 512, 1)
        self.dec_bn1 = nn.BatchNorm1d(512)
        
        self.dec_conv2 = nn.Conv1d(512, 256, 1)
        self.dec_bn2 = nn.BatchNorm1d(256)
        
        self.dec_conv3 = nn.Conv1d(256, 128, 1)
        self.dec_bn3 = nn.BatchNorm1d(128)
        self.dec_proj1 = nn.Conv1d(512, 128, 1) # Projection for residual
        self.dec_bn_proj1 = nn.BatchNorm1d(128)
        
        self.dec_conv4 = nn.Conv1d(128, 3, 1) # Final Output: (dx, dy, dz)
        self.dropout_dec = nn.Dropout(dropout * 0.5)
        
        self._build_codecs()
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _build_codecs(self):
        if self.use_res:
            def state_encoder(x):
                x = F.relu(self.bn1(self.conv1(x)))
                identity = x
                x = F.relu(self.bn2(self.conv2(x))) + identity
                identity = self.bn_proj1(self.proj1(x))
                x = F.relu(self.bn3(self.conv3(x))) + identity
                identity = self.bn_proj2(self.proj2(x))
                x = F.relu(self.bn4(self.conv4(x))) + identity
                x = self.bn5(self.conv5(x))
                x = torch.max(x, 2, keepdim=False)[0] 
                return self.dropout_state(x)
            
            def action_encoder(a):
                if len(a.shape) == 1: a = a.unsqueeze(1)
                a = F.relu(self.act_bn1(self.act_fc1(a)))
                identity = a
                a = F.relu(self.act_bn2(self.act_fc2(a))) + identity
                identity = self.act_bn_proj1(self.act_proj1(a))
                a = F.relu(self.act_bn3(self.act_fc3(a))) + identity
                identity = self.act_bn_proj2(self.act_proj2(a))
                a = F.relu(self.act_bn4(self.act_fc4(a))) + identity
                return self.act_fc5(self.dropout_action(a))

            def decoder(combined_features):
                """
                Processes each point individually using shared weights.
                combined_features: (B, Latent*2 + 3, N)
                """
                x = F.relu(self.dec_bn1(self.dec_conv1(combined_features)))
                x = self.dropout_dec(x)
                identity = self.dec_bn_proj1(self.dec_proj1(x))
                x = F.relu(self.dec_bn2(self.dec_conv2(x)))
                x = F.relu(self.dec_bn3(self.dec_conv3(x))) + identity
                delta = self.dec_conv4(x) # (B, 3, N)
                return delta.transpose(1, 2) # (B, N, 3)
        else:
            def state_encoder(x):
                x = F.relu(self.bn1(self.conv1(x)))
                x = F.relu(self.bn2(self.conv2(x)))
                x = F.relu(self.bn3(self.conv3(x)))
                x = F.relu(self.bn4(self.conv4(x)))
                x = self.bn5(self.conv5(x))
                x = torch.max(x, 2, keepdim=False)[0] 
                return self.dropout_state(x)
            
            def action_encoder(a):
                if len(a.shape) == 1: a = a.unsqueeze(1)
                a = F.relu(self.act_bn1(self.act_fc1(a)))
                a = F.relu(self.act_bn2(self.act_fc2(a))) 
                a = F.relu(self.act_bn3(self.act_fc3(a)))
                a = F.relu(self.act_bn4(self.act_fc4(a)))
                return self.act_fc5(self.dropout_action(a))

            def decoder(combined_features):
                """
                Processes each point individually using shared weights.
                combined_features: (B, Latent*2 + 3, N)
                """
                x = F.relu(self.dec_bn1(self.dec_conv1(combined_features)))
                x = self.dropout_dec(x)
                x = F.relu(self.dec_bn2(self.dec_conv2(x)))
                x = F.relu(self.dec_bn3(self.dec_conv3(x)))
                delta = self.dec_conv4(x) # (B, 3, N)
                return delta.transpose(1, 2) # (B, N, 3)
            
        self.state_encoder = state_encoder
        self.action_encoder = action_encoder
        self.decoder = decoder
        
    def forward(self, x_t, a_t):
        B, C, N = x_t.shape
        
        # Encode Global Context
        x_l = self.state_encoder(x_t)      # (B, latent_size)
        a_l = self.action_encoder(a_t)     # (B, latent_size)
        global_latent = torch.cat([x_l, a_l], dim=1) # (B, latent_size*2)
        
        # Expand Global Context to every point
        global_expanded = global_latent.unsqueeze(2).expand(-1, -1, N) # (B, latent_size*2, N)
        
        # Concatenate Global Context with Point Positions
        # tells the decoder where each point is in space.
        combined_features = torch.cat([global_expanded, x_t], dim=1) # (B, latent_size*2 + 3, N)
        
        # Predict Point-wise Deltas
        delta = self.decoder(combined_features) 
        return delta