import ml_collections

def get_config():
    """Get the hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    config.mode = "train"
    
    # Hyperparameters for dataset. 
    config.ratio_tr_data = 0.8 
    config.num_workers = 8 
    config.data_mnistpts_dir = "data_dump"
    config.num_pts = 256 
    config.order_pts = False 
    config.num_classes = 10
    config.random_sample = True

    # Hyperparameters for models.
    config.model = "FcNet"
    config.outc_list = [128, 128] # number of channels of intermediate FC layers.

    # Hyperparameters for training.
    config.log_dir = "logs"
    config.use_cuda = False 
    config.batch_size = 256
    config.num_epochs = 50
    config.lr = 1e-2
 
    return config