import numpy as np

def get_num_parameters(model):
    
    trainable_parameters = sum([p.numel() for p in model.parameters() if p.requires_grad])
    total_parameters = sum([p.numel() for p in model.parameters()])
    
    return trainable_parameters, total_parameters

def get_data_split_indices(num_samples, val_share, test_share):
    
    train_share = 1 - val_share - test_share
    
    indices = np.arange(num_samples)
    # np.random.seed(2)
    np.random.seed(12345)
    np.random.shuffle(indices)
    
    train_indices = indices[:int(train_share * num_samples)]
    val_indices = indices[int(train_share * num_samples):int((train_share + val_share) * num_samples)]
    test_indices = indices[int((train_share + val_share) * num_samples):]
    
    return train_indices, val_indices, test_indices

def scale_targets(train_targets:np.ndarray, val_targets:np.ndarray, test_targets:np.ndarray):
    
    mean = np.mean(train_targets, axis=0)
    std = np.std(train_targets, axis=0)
    
    scaler_targets = [mean, std]
    
    for col in range(train_targets.shape[1]):
        train_targets[:,col] = (train_targets[:,col] - mean[col]) / std[col]
        val_targets[:,col] = (val_targets[:,col] - mean[col]) / std[col]
        test_targets[:,col] = (test_targets[:,col] - mean[col]) / std[col]
    
    return train_targets, val_targets, test_targets, scaler_targets

def load_molecular_features(mode):
    
    rdkit_descriptors = np.load('../data/mol_descriptors.npy')
    morgan_fingerprints = np.load('../data/mol_morgan_fingerprints.npy')
    mordred_descriptors = np.load('../data/Mordred_mol_descriptors.npy')
    
    
    return_len = 0
    if "rdkit" in mode and "mordred" in mode:
        descriptors = np.concatenate((rdkit_descriptors, mordred_descriptors), axis=1)
        return_len += 1
    elif "mordred" in mode:
        descriptors = mordred_descriptors
        return_len += 1
    elif "rdkit" in mode:
        descriptors = rdkit_descriptors
        return_len += 1
    
    if "morgan" in mode:
        fingerprints = morgan_fingerprints
        return_len += 1
        
    if return_len == 1:
        if "morgan" in mode:
            features = fingerprints
            return features
        else:
            features = descriptors
            return features
    elif return_len == 2:
        features = np.concatenate((descriptors, fingerprints), axis=1)
        return features
    
        
