import os

import matplotlib.pyplot as plt
import torch


class Network(object):
    """Wrapper for training and testing pipelines."""

    def __init__(self, model, config):
        """Initialize configuration."""
        self.config = config
        self.model = model
        
        if "weight_decay" in config:
            optimizer_weight_decay = config.weight_decay
        else:
            optimizer_weight_decay = 0
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=config.lr, weight_decay=optimizer_weight_decay
        )
        if self.config.use_cuda:
            self.model.cuda()

        # init auxiliary stuff such as log_func
        self._init_aux()

    def _init_aux(self):
        """Intialize aux functions, features."""
        # Define func for logging.
        self.log_func = print

        # Define directory wghere we save states such as trained model.
        self.log_dir = self.config.log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # File that we save trained model into.
        self.checkpts_file = os.path.join(self.log_dir, "checkpoint.pth")

        # We save model that achieves the best performance: early stoppoing strategy.
        self.bestmodel_file = os.path.join(self.log_dir, "best_model.pth")

        # Recording the training losses and validation performance.
        self.train_losses = []
        self.valid_oas = []
        self.idx_steps = []

    def plot_log(self):
        """Plot training logs (better with tensorboard, but we will try matplotlib this time!)."""

        # TODO: Draw plots for the training and validation results, as shown in
        # the example results. Use matplotlib's subplots.
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)

        ax1.plot(self.idx_steps, self.train_losses)
        ax1.set_title('Training loss curve')
        ax2.plot(self.idx_steps, self.valid_oas)
        ax2.set_title('Validation accuracy curve')

        fig.suptitle('Visualization of training logs')
        plt.tight_layout()
        plt.show()

    def _save(self, pt_file):
        """Saving trained model."""

        # Save the trained model.
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            pt_file,
        )

    def _restore(self, pt_file):
        """Restoring trained model."""
        print(f"restoring {pt_file}")

        # Read checkpoint file.
        load_res = torch.load(pt_file)
        # Loading model.
        self.model.load_state_dict(load_res["model"])
        # Loading optimizer.
        self.optimizer.load_state_dict(load_res["optimizer"])

    def train(self, loader_tr, loader_va=None):
        """Training pipeline."""
        # Switch model into train mode.
        self.model.train()
        best_va_acc = 0  # Record the best validation accuracy.

        for epoch in range(self.config.num_epochs):
            losses = []
            for data in loader_tr:
                # Transfer data from CPU to GPU.
                if self.config.use_cuda:
                    for key in data.keys():
                        data[key] = data[key].cuda()

                pred = self.model(data["pc"])
                loss = self.model.get_loss(pred, data["label"])
                losses += [loss]

                # Calculate the gradient.
                loss.backward()
                # Update the parameters according to the gradient.
                self.optimizer.step()
                # Zero the parameter gradients in the optimizer
                self.optimizer.zero_grad()

            loss_avg = torch.mean(torch.stack(losses)).item()

            # Save model every epoch.
            self._save(self.checkpts_file)

            # evaluate on validation set
            if loader_va:
                acc = self.test(loader_va, mode="valid")

                # Early stopping strategy.
                if acc > best_va_acc:
                    # Save model with the best accuracy on validation set.
                    best_va_acc = acc
                    self._save(self.bestmodel_file)
                self.log_func(
                    "Epoch: %3d, loss_avg: %.5f, val OA: %.5f, best val OA: %.5f"
                    % (epoch, loss_avg, acc, best_va_acc)
                )
            else:
                self.log_func( "Epoch: %3d, loss_avg: %.5f" % (epoch, loss_avg))

            # Recording training losses and validation performance.
            self.train_losses += [loss_avg] 
            self.idx_steps += [epoch]
            if loader_va:
                self.valid_oas += [acc]


    def test(self, loader_te, mode="test"):
        """Estimating the performance of model on the given dataset."""
        # Choose which model to evaluate.
        if mode == "test":
            self._restore(self.bestmodel_file)
        # Switch the model into eval mode.
        self.model.eval()

        accs = []
        num_samples = 0
        for data in loader_te:
            if self.config.use_cuda:
                for key in data.keys():
                    data[key] = data[key].cuda()
            batch_size = len(data["pc"])
            pred = self.model(data["pc"])
            acc = self.model.get_acc(pred, data["label"])
            accs += [acc * batch_size]
            num_samples += batch_size

        avg_acc = torch.stack(accs).sum() / num_samples

        # Switch the model into training mode
        self.model.train()
        return avg_acc

if __name__ == "__main__":
    """Main for mock testing."""
    from get_config import get_config
    from get_dataloader import get_dataloader
    from get_model import get_model

    config = get_config()
    model = get_model(config)
    net = Network(model, config)
    dataloader_tr, dataloader_va = get_dataloader(config, mode="train")
    dataloader_te = get_dataloader(config, mode="test")[0]

    net.train(dataloader_tr, dataloader_va)
    net.test(dataloader_te)
