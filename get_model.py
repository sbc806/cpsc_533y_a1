import torch
import torch.nn as nn
import torch.nn.functional as F


def get_model(config):
    """Get a network."""
    inc = config.num_pts * 2
    return FcNet(inc, config)


# Define a the model below.
class FcNet(nn.Module):
    """Simple network consisting of FC layers"""

    def __init__(self, inc, config):
        """Initialize network with hyperparameters.

        Args:
            inc (int): number of channels in the input.
            config (ml_collections.dict): configuration hyperparameters.
        """
        super(FcNet, self).__init__()
        self.config = config
        num_classes = config.num_classes
        outc_list = config.outc_list

        # TODO: Compose the model according to configuration specs
        #
        # Hint: nn.Sequential().

        # init weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            # apply a uniform distribution to the weights and bias
            print('initializing weights in {}'.format(module.__class__.__name__))
            module.weight.data.uniform_(-1, 1)
            module.bias.data.uniform_(-1, 1)

    def forward(self, x):
        """Forward pass.

        Args:
            x (array): BxNx2, input tensor.
        """
        if self.config.order_pts:
            # TODO: Order point clouds according to its x coordinates.
            #
            # Hint: Use `torch.sort` to get the ordered index
            #      and then use `torch.gather` to get the ordered points.

        # TODO: Define the forward  pass and get the logits for classification.

        return F.log_softmax(logits, dim=1)

    def get_loss(self, pred, label):
        """Compute loss by comparing prediction and labels.

        Args:
            pred (array): BxD, probability distribution over D classes.
            label (array): B, category label.
        Returns:
            loss (tensor): scalar, cross entropy loss for classfication.
        """
        loss = F.nll_loss(pred, label)
        return loss

    def get_acc(self, pred, label):
        """Compute the acccuracy."""
        pred_choice = pred.max(dim=1)[1]
        acc = (pred_choice == label).float().mean()
        return acc
