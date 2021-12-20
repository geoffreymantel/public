import torch
import torch.nn as nn
import torch.optim as optim

from collections import OrderedDict

from . import EncoderConv2D

class MNISTClassifier(nn.Module):
    def __init__(self, config):
        super(MNISTClassifier, self).__init__()
        
        # Encoder and classifier
        self.encoder = EncoderConv2D(config['classifier_latent_dim']).to(config['device'])
        self.classifier = nn.Linear(in_features=config['classifier_latent_dim'], out_features=10).to(config['device'])
        
        # Loss function - good for classification problems
        self.loss = nn.CrossEntropyLoss()

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=config['learning_rate'])

    def forward(self, x):
        z = self.encoder(x)
        labels = self.classifier(z)
        return labels

    # Trains for a single epoch. Returns a dictionary representing the loss for that epoch
    def fit(self, training_data, validation_data, config):

        audit_log = {
            "epoch_cross_entropy": 0.0,
            "validation_cross_entropy": 0.0
        }

        # --- training ---
        self.train()
        for data, labels in training_data:
            
            # The main optimizer calculation
            self.optimizer.zero_grad()
            predictions = self(data)
            loss = self.loss(predictions, labels)
            loss.backward()
            self.optimizer.step()

            audit_log['epoch_cross_entropy'] += float(loss)

        # --- validation ---
        self.eval()
        with torch.no_grad():
            for data, labels in validation_data:

                # Compute the testing loss
                predictions = self(data)
                loss = self.loss(predictions, labels)
                audit_log['validation_cross_entropy'] += float(loss)

                # TODO: Compute the accuracy

        # --- auditing ---
        audit_log['epoch_cross_entropy'] /= len(training_data)
        audit_log['validation_cross_entropy'] /= len(validation_data)
        
        return audit_log    
    
    def save(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])