import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict

from . import EncoderConv2D, DecoderConv2D

class BarlowTwinsAutoencoder(nn.Module):
    def __init__(self, config):
        super(BarlowTwinsAutoencoder, self).__init__()
        
        # Encoders and decoder
        # We want true batch norm (no affine parameters) for the 'what' encoder
        self.what_encoder = EncoderConv2D(config['what_latent_dim'], affine=False).to(config['device'])
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

        # Caculate the position encoding for frame1
        z_pos = self.where_encoder(frame1)

        # Concatenate identity and position as input to the decoder
        x = self.decoder(torch.cat((z_id1, z_pos), dim=1))
        
        return x, z_id1, z_id2, z_pos
        
    def loss(self, x, z_id1, z_id2, data, config):
        frame1, frame2 = data
        
        # Barlow Twins loss consists of two parts:
        # an invariance term and a redundancy reduction term.
        # Both parts are functions of the elements of the cross-correlation matrix.
        
        # Empirical cross-correlation matrix
        cross_correlation = z_id1.T @ z_id2
        n, m = cross_correlation.shape
        assert n == m
        
        # Normalize by batch size
        cross_correlation = cross_correlation.div(config['batch_size'])

        # Invariance term
        on_diagonal_loss = torch.diagonal(cross_correlation).add(-1).pow(2).sum()
        
        # Redundancy reduction term
        off_diagonal_loss = cross_correlation.flatten()[:-1].view(n-1, n+1)[:,1:].flatten()
        off_diagonal_loss = off_diagonal_loss.pow(2).sum()

        # We also have a reconstruction loss term,
        # between the decoder reconstruction and the original frame
        reconstruction_loss = self.mse_loss(x, frame1)

        return on_diagonal_loss, off_diagonal_loss, reconstruction_loss

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
            "epoch_on_diagonal_loss": 0.0,
            "epoch_off_diagonal_loss": 0.0,
            "validation_reconstruction_loss": 0.0,
            "validation_on_diagonal_loss": 0.0,
            "validation_off_diagonal_loss": 0.0
        }
        
        # --- training ---
        self.train()
        for data in training_data:

            # The main optimizer calculation
            self.optimizer.zero_grad()
            x, z_id1, z_id2, z_pos = self(data, config)

            on_diagonal_loss, off_diagonal_loss, reconstruction_loss = self.loss(x, z_id1, z_id2, data, config)
            
            # Add all the losses with appropriate weights
            barlow_loss = (on_diagonal_loss * (1.0 - config['off_diagonal_ratio'])) + (off_diagonal_loss * config['off_diagonal_ratio'])
            total_loss = reconstruction_loss + (barlow_loss * config['barlow_lambda'])

            # Grind the grads!
            total_loss.backward()
            self.optimizer.step()
          
            audit_log['epoch_reconstruction_loss'] += float(reconstruction_loss)
            audit_log['epoch_on_diagonal_loss'] += float(on_diagonal_loss)
            audit_log['epoch_off_diagonal_loss'] += float(off_diagonal_loss)

        # --- validation ---
        self.eval()
        with torch.no_grad():
            for data in validation_data:
                x, z_id1, z_id2, z_pos = self(data, config)

                on_diagonal_loss, off_diagonal_loss, reconstruction_loss = self.loss(x, z_id1, z_id2, data, config)
                
                audit_log['validation_reconstruction_loss'] += float(reconstruction_loss)
                audit_log['validation_on_diagonal_loss'] += float(on_diagonal_loss)
                audit_log['validation_off_diagonal_loss'] += float(off_diagonal_loss)

        # --- auditing ---
        audit_log['epoch_reconstruction_loss'] /= len(training_data)
        audit_log['epoch_on_diagonal_loss'] /= len(training_data)
        audit_log['epoch_off_diagonal_loss'] /= len(training_data)
        audit_log['validation_reconstruction_loss'] /= len(validation_data)
        audit_log['validation_on_diagonal_loss'] /= len(validation_data)
        audit_log['validation_off_diagonal_loss'] /= len(validation_data)
        
        return audit_log