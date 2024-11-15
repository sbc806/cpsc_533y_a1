import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


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
        
        if "use_batch_normalization" in config:
            use_batch_normalization = config.use_batch_normalization
        else:
            use_batch_normalization = False
        if "activation" in config:
            self.activation = config.activation
        else:
            self.activation = 'relu'
        
        model_layers = OrderedDict()
        for i in range(0, len(outc_list)):
            if i == 0:
                model_layers[f'Linear-{i}'] = nn.Linear(inc, outc_list[i], bias=True)
            else:
                model_layers[f'Linear-{i}'] = nn.Linear(outc_list[i-1], outc_list[i], bias=True)
            if use_batch_normalization:
                model_layers[f'BatchNorm1d-{i}'] = nn.BatchNorm1d(outc_list[i])
            if self.activation == 'relu':
                model_layers[f'ReLU-{i}'] = nn.ReLU(inplace=True)
            elif self.activation == "elu":
                model_layers[f'ELU-{i}'] = nn.ELU(inplace=True)
            elif self.activation == "leaky_relu":
                model_layers[f'LeakyReLU'] = nn.LeakyReLU(inplace=True)
            elif self.activation == 'tanh':
                model_layers[f'Tanh-{i}'] = nn.Tanh()

        model_layers['output'] = nn.Linear(outc_list[-1], num_classes, bias=True)

        self.net = nn.Sequential(model_layers)

        # init weights
        if "init" not in config:
            config.init = "uniform"
        if config.init == 'uniform':
            self.apply(self._init_weights)
        elif config.init == "normal":
            self.apply(self._init_weights_normal)
        elif config.init == "xavier":
            self.apply(self._init_weights_xavier)
        elif config.init == "xavier_1":
            self.apply(self._init_weights_xavier_1)
        elif config.init == 'he':
            self.apply(self._init_weights_he)
        elif config.init == "he_1":
            self.apply(self._init_weights_he_1)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            # apply a uniform distribution to the weights and bias
            print('initializing weights in {}'.format(module.__class__.__name__))
            module.weight.data.uniform_(-1, 1)
            module.bias.data.uniform_(-1, 1)

    def _init_weights_normal(self,module):
        if isinstance(module, torch.nn.Linear):
            print('initializing normal weights in {}'.format(module.__class__.__name__))
            nn.init.normal_(module.weight)
            nn.init.normal_(module.bias)

    def _init_weights_xavier(self, module):
        if isinstance(module, torch.nn.Linear):
            print('initializing Xavier weights in {}'.format(module.__class__.__name__))
            nn.init.xavier_uniform_(module.weight)
            module.bias.data.uniform_(-0.1,0.1)

    def _init_weights_xavier_1(self, module):
        if isinstance(module, torch.nn.Linear):
            print('initializing Xavier_1 weights in {}'.format(module.__class__.__name__))
            nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)

    def _init_weights_he(self, module):
        if isinstance(module, torch.nn.Linear):
            print('initializing He weights in {}'.format(module.__class__.__name__))
            nn.init.kaiming_normal_(module.weight, nonlinearity=self.activation)
            module.bias.data.fill_(0.0)

    def _init_weights_he_1(self, module):
        if isinstance(module, torch.nn.Linear):
            print('initializing He_1 weights in {}'.format(module.__class__.__name__))
            nn.init.kaiming_normal_(module.weight, nonlinearity=self.activation)
            module.bias.data.fill_(0.01)

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
            sorted_tensor, indices = torch.sort(x[:,:,0:1], dim=1)
            sorted_indices = torch.cat((indices, indices), dim=-1)
            x = torch.gather(x, dim=1, index=sorted_indices)
        # TODO: Define the forward  pass and get the logits for classification.
        flattened_x = torch.flatten(x, start_dim=1)
        logits = self.net(flattened_x)
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
