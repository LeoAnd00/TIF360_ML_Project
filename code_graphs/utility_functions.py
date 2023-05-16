import numpy as np

def get_num_parameters(model):
    
    trainable_parameters = sum([p.numel() for p in model.parameters() if p.requires_grad])
    total_parameters = sum([p.numel() for p in model.parameters()])
    
    return trainable_parameters, total_parameters

def get_data_split_indices(num_samples, val_share, test_share):
    
    train_share = 1 - val_share - test_share
    
    indices = np.arange(num_samples)
    np.random.seed(42)
    np.random.shuffle(indices)
    
    train_indices = indices[:int(train_share * num_samples)]
    val_indices = indices[int(train_share * num_samples):int((train_share + val_share) * num_samples)]
    test_indices = indices[int((train_share + val_share) * num_samples):]
    
    return train_indices, val_indices, test_indices
    