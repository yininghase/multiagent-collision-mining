import numpy as np
from numpy import pi
import torch.nn as nn
import torch

class WeightedMeanSquaredLoss(nn.Module):
    """Weighted mean squared error loss with time-step weighting and optional static obstacle loss."""
    def __init__(self, horizon = 20, device = 'cpu', no_weight=False):
        """Initialize the loss function.
        
        Args:
            horizon (int): Prediction horizon length (default: 20).
            device (str): Device for computation (default: "cpu").
            no_weight (bool): If True, use uniform weights; otherwise, weight early steps more (default: False).
        """
        super().__init__()
        
        self.device = device
        
        if no_weight:
            self.weights = np.ones(horizon)
        elif horizon <= 10: 
            self.weights = np.flip(np.array(range(horizon))**3)
        else:
            self.weights = np.flip(np.array(range(10))**3)
            self.weights = np.append(self.weights, [1]*(horizon-10))
  
        self.weights = torch.from_numpy(self.weights / sum(self.weights)).type(torch.float32)
        self.weights = self.weights.repeat_interleave(2).to(self.device)
    
    def forward(self, preds, targets, be_static=None):
        """Compute weighted MSE loss.
        
        Args:
            preds (Tensor): Predicted controls of shape (..., 2*horizon).
            targets (Tensor): Ground truth controls of shape (..., 2*horizon).
            be_static (Tensor, optional): Static obstacle predictions for extra loss (default: None).
        
        Returns:
            Tensor: Scalar loss value.
        """
        loss_1 = (preds - targets)**2
        weighted_loss_1 = loss_1 @ self.weights
        weighted_mean_loss = torch.mean(weighted_loss_1)
        
        if be_static is not None and len(be_static)>0:
            loss_2 = (be_static)**2
            weighted_loss_2 = loss_2 @ self.weights
            weighted_loss_mean_loss_2 =torch.mean(weighted_loss_2)*0.1
            weighted_mean_loss = weighted_mean_loss + weighted_loss_mean_loss_2
            
        
        return weighted_mean_loss
       
    
