import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict

from . import EncoderConv2D, DecoderConv2D

class InertialAutoencoder(nn.Module):
    def __init__(self, config):
        super(InertialAutoencoder, self).__init__()
        
        # Encoders and decoder
        self.what_encoder = EncoderConv2D(config['what_latent_dim']).to(config['device'])
        self.where_encoder = EncoderConv2D(config['where_latent_dim']).to(config['device'])
        self.decoder = DecoderConv2D(config['what_latent_dim'] + config['where_latent_dim']).to(config['device'])

        # Reconstruction loss function
        self.mse_loss = nn.MSELoss()

        # The optimizer will optimize for the reconstruction loss and the Barlow Twins loss
        self.optimizer = optim.Adam(self.parameters(), lr=config['learning_rate'])
        
    def forward(self, data, config):
        frame1, frame2 = data

        # Calculate identity encoding for frame pair
        z_id1 = self.what_encoder(frame1)
        z_id2 = self.what_encoder(frame2)

        # Caculate the position encoding
        z_pos = self.where_encoder(frame1)

        # Concatenate identity and position as input to the decoder
        x = self.decoder(torch.cat((z_id1, z_pos), dim=1))
        
        return x, z_id1, z_id2, z_pos
        
    def loss(self, x, z_id1, z_id2, data, config):
        # Ignore the blurred frame - we use the original frame for the reconstruction loss term
        frame1, frame2 = data
        
        # MSE loss on both the z encodings and the pixel values:
        inertial_loss = self.mse_loss(z_id1, z_id2)
        reconstruction_loss = self.mse_loss(x, frame1)
        
        return inertial_loss, reconstruction_loss

    def save(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Trains for a single epoch. Returns a dictionary representing the loss for that epoch
    def fit(self, training_data, validation_data, config):

        audit_log = {
            "epoch_reconstruction_loss": 0.0,
            "epoch_inertial_loss": 0.0,
            "validation_reconstruction_loss": 0.0,
            "validation_inertial_loss": 0.0
        }
        
        # --- training ---
        self.train()
        for data in training_data:

            # The main optimizer calculation
            self.optimizer.zero_grad()
            x, z_id1, z_id2, z_pos = self(data, config)

            inertial_loss, reconstruction_loss = self.loss(x, z_id1, z_id2, data, config)
            total_loss = reconstruction_loss + (inertial_loss * config['inertial_lambda'])

            # Grind the grads!
            total_loss.backward()
            self.optimizer.step()
          
            audit_log['epoch_reconstruction_loss'] += float(reconstruction_loss)
            audit_log['epoch_inertial_loss'] += float(inertial_loss)

        # --- validation ---
        self.eval()
        with torch.no_grad():
            for data in validation_data:
                x, z_id1, z_id2, z_pos = self(data, config)

                inertial_loss, reconstruction_loss = self.loss(x, z_id1, z_id2, data, config)
                
                audit_log['validation_reconstruction_loss'] += float(reconstruction_loss)
                audit_log['validation_inertial_loss'] += float(inertial_loss)

        # --- auditing ---
        audit_log['epoch_reconstruction_loss'] /= len(training_data)
        audit_log['epoch_inertial_loss'] /= len(training_data)
        audit_log['validation_reconstruction_loss'] /= len(validation_data)
        audit_log['validation_inertial_loss'] /= len(validation_data)
        
        return audit_log