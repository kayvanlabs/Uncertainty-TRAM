
import torch.nn as nn

class TramLoss(nn.Module):
    """
    Transfer and Marginalize (TRAM) loss module for training deep learning models.

    Args:
        additon_loss_factor (float): The scaling factor applied to the privileged head loss. Default is 1.
        loss_type (str): The type of loss function to use. Currently, only 'ce' (CrossEntropyLoss) is supported.

    Attributes:
        addition_loss_factor (float): The scaling factor applied to the privileged head loss.
        cross_entropy (nn.CrossEntropyLoss): The cross-entropy loss function.
        base_head_loss (float): The base head loss calculated during forward pass.
        add_head_loss (float): The privileged head loss calculated during forward pass.

    """

    def __init__(self, additon_loss_factor=1, loss_type='ce'):
        super(TramLoss, self).__init__()
        self.addition_loss_factor = additon_loss_factor
        if loss_type == 'ce':
            self.cross_entropy = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError(f'Loss type {loss_type} not implemented')
        
        self.base_head_loss, self.add_head_loss = 0, 0

    def forward(self, outputs, targets, add_output=None, train=False):
        """
        Forward pass of the TRAM loss module.
        Args:
            outputs (Tensor): The output tensor from the base head.
            targets (Tensor): The target tensor.
            add_output (Tensor): The output tensor from the privileged head. Optional, used during training.
            train (bool): Flag indicating if the model is in training mode. Default is False.

        Returns:
            Tensor: The total loss, which is the sum of the base head loss and the scaled privileged head loss.
        
        """
        if train:
            self.base_head_loss = self.cross_entropy(outputs, targets)
            self.add_head_loss = self.cross_entropy(add_output, targets)
            return self.base_head_loss + self.addition_loss_factor * self.add_head_loss
        else:
            return self.cross_entropy(outputs, targets)
    
    def ordinal_forward(self, outputs, targets, ordinal_loss):
        """
        Forward pass for ordinal regression on the privileged branch.

        Args:
            outputs (Tensor): The output tensor from the base head.
            targets (Tensor): The target tensor.
            ordinal_loss (float): The ordinal loss value.

        Returns:
            Tensor: The total loss, which is the sum of the base head loss and the scaled privileged head loss.
        
        """
        self.base_head_loss = self.cross_entropy(outputs, targets)
        self.add_head_loss = ordinal_loss
        return self.base_head_loss + self.addition_loss_factor * self.add_head_loss
        
    @property
    def base_loss(self):
        """
        Property to access the base head loss.

        Returns:
            float: The base head loss value.
        
        """
        return self.base_head_loss
    
    @property
    def pi_loss(self):
        """
        Property to access the privileged head loss.

        Returns:
            float: The privileged head loss value.
        
        """
        return self.add_head_loss