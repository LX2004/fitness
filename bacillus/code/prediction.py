import hyperopt
from hyperopt import fmin, tpe, hp, Trials
import torch
from utils import *
from net import predict_transformerv2
from initialize import initialize_weights
from torch.utils.data import DataLoader,Dataset

import numpy as np
import pdb
import pickle
import os
from sklearn.model_selection import KFold
# Define a custom sample class for sequence data
class Seq_bio_sample():
    def __init__(self, label, feature, light, ori):

        self.label = label
        self.feature = feature

        self.light = light
        self.ori = ori


def make_dataset_sequences_bio(guides, fit18s, essentials):

    features_array = []
    bios_array = []
    labels_array = []

    fit18s = np.array(fit18s)
    max_reads = np.max(fit18s) 
    min_reads = np.min(fit18s)

    print('max_reads = ',max_reads)
    print('min_reads = ',min_reads)

    pdb.set_trace()
    
    number = 0

    for sequence, score, essential in zip(guides, fit18s, essentials):

        if len(sequence) < 20:

            print('length = ', len(sequence))
            print('sequence = ',sequence)
            continue
        
        # One-hot encoding for essential labels
        essential = str(essential)

        if essential == 'True':
            ori = np.array([1,0])
        
        elif essential == 'False':
            ori = np.array([0,1])
            
        else:
            print(f"Input essential = {essential} is not in the predefined list, cannot one-hot encode.")
            continue

        feature = Dimer_split_seqs(sequence)  # Sequence as input
        feature = np.array(feature)
        feature = feature.astype(int)

        label = (score - min_reads)/(max_reads -  min_reads)

        features_array.append(feature)
        bios_array.append(ori)
        labels_array.append(label)

        number += 1
    print('number = ',number)
    
    return np.array(features_array), np.array(labels_array), np.array(bios_array)


def read_data(filename):

    import math

    guides = []
    fit18s = []

    # Biological information
    essentials = []
    df = pd.read_csv(filename)

    number = 0
    for variant_guide, essential, fitness in zip(df['variant_guide'], df['essential'], df['fitness']):
        
        fitness = float(fitness)

        if math.isnan(fitness):
            print(f'fitness is {fitness}!!!')
            continue

        guides.append(variant_guide.upper())
        fit18s.append(fitness)
        essentials.append(essential)

        number += 1

    return guides, fit18s, essentials 


# Define a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, features, biofeatures, labels):

        self.features = features
        self.labels = labels
        self.biofeatures = biofeatures
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):

        feature = self.features[idx]
        label = self.labels[idx]
        biofeature = self.biofeatures[idx]

        return feature, biofeature, label


def train(params, features_array, bios_array, labels_array):

    patience = 50
    
    print('params = ',params)

    # Store Pearson correlation for k-fold cross validation
    test_pearson_kfold = []

    for fold, (train_indices, val_indices) in enumerate(kf.split(features_array)):

        best_val_loss = float('inf')
        no_improve_epochs = 0

        print(f"Fold {fold + 1}/{k_folds}")
        print('size of train datset is: ', len(train_indices))
        print('size of test datset is: ', len(val_indices))

        # Create custom dataset and dataloaders
        train_dataset = CustomDataset(features_array[train_indices], bios_array[train_indices], labels_array[train_indices])
        test_dataset = CustomDataset(features_array[val_indices], bios_array[val_indices], labels_array[val_indices])

        train_loader = DataLoader(train_dataset, batch_size=params['train_batch_size'], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=params['train_batch_size'], shuffle=False)

        print('Train set size = ',len(train_loader))
        print('Test set size = ',len(test_loader))

        print('start compose simple gan model')
        gen = predict_transformerv2.Predict_transformer_bacillus(params=params).to(device)
        initialize_weights(gen)
        print('successful compose simple gan model')

        # Optimizer and loss functions
        opt_gen = torch.optim.Adam(gen.parameters(), lr=params['train_base_learning_rate'], weight_decay=params['l2_regularization'])
        loss_fc = torch.nn.MSELoss()

        loss_train =[]
        loss_test = []
        metric = []

        '''Start training'''
        for epoch in range(params['train_epochs_num']):
            
            # Learning rate adjustment
            if epoch > 0 and epoch % 100 == 0:
                for param_group in opt_gen.param_groups:
                    print('Adjust learning rate')
                    param_group['lr'] = param_group['lr'] / 2.0

            loss_train_one_epoch = 0
            loss_test_one_epoch = 0
            loss_mse = 0
            loss_pier = 0
            
            gen.train()
            for data, bio, target in train_loader:
                
                data = data.to(device)
                target = target.to(device)
                bio = bio.to(device)

                output = gen(data,bio)
                output = torch.squeeze(output, dim=1)

                loss_gen = loss_fc(target.float(), output.float())
                loss_pi = loss_pierxun(target=target.float(),output=output.float())

                loss_gen = loss_gen.float()
                loss_pi = loss_pi.float()

                if loss_kind == 'pearson':
                    loss_all = -loss_pi
                elif loss_kind == 'pearson_mse':
                    loss_all = -loss_pi + loss_gen
                elif loss_kind == 'mse':
                    loss_all = loss_gen
                else:
                    print('Invalid loss type, please check!')

                opt_gen.zero_grad()
                loss_all.backward()
                opt_gen.step()

                loss_train_one_epoch += loss_all.item()
                loss_mse += loss_gen.item()
                loss_pier += loss_pi.item()
                
            loss_train.append(loss_train_one_epoch/len(train_loader))

            if epoch % 10 == 0:
                print(
                    f"Epoch[{epoch}/{params['train_epochs_num']}] ****Train loss: {loss_train_one_epoch/len(train_loader):.6f}****MSE loss: {loss_mse/len(train_loader):.6f}****Pierxun loss: {loss_pier/len(train_loader):.6f}"
                )
            
            # Evaluation
            gen.eval()
            targets = []
            outputs = []

            for data, bio, target in test_loader:
                
                data = data.to(device)
                target = target.to(device)
                bio = bio.to(device)

                output = gen(data, bio)
                output = torch.squeeze(output, dim=1)
                loss_gen = loss_fc(target, output)

                targets.append(target)
                outputs.append(output)

                loss_test_one_epoch += loss_gen.detach().cpu().numpy() 
            
            correlation_coefficient = compute_correlation_coefficient(torch.cat(targets, dim=0), torch.cat(outputs, dim=0))
            loss_test.append(loss_test_one_epoch/len(test_loader))

            # Early stopping logic
            if loss_test_one_epoch/len(test_loader) < best_val_loss:
                best_val_loss = loss_test_one_epoch/len(test_loader)
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            if epoch % 10 == 0:
                print(
                    f"Epoch[{epoch}/{params['train_epochs_num']}] ****Test loss: {loss_test_one_epoch/len(test_loader):.6f}********test correlation_coefficient:{correlation_coefficient}"
                )
            
            metric.append(correlation_coefficient)

            # Save model if performance is good
            if correlation_coefficient > 0.7:
                if loss_kind == 'pearson':
                    torch.save(gen,'../models/bio_pearson_predict_{0}_mertric={1}.pth'.format(epoch, correlation_coefficient))
                elif loss_kind == 'pearson_mse':
                    torch.save(gen,'../models/bio_pearson_mse_predict_{0}_mertric={1}.pth'.format(epoch, correlation_coefficient))
                elif loss_kind == 'mse':
                    torch.save(gen,'../models/bio_mse_predict_{0}_mertric={1}.pth'.format(epoch, correlation_coefficient))
                else:
                    print('Loss type error, please check!')

            # Learning rate decay logic
            if no_improve_epochs > 0 and no_improve_epochs % 10:
                for param_group in opt_gen.param_groups:
                    param_group['lr'] = param_group['lr']*0.85

            # Early stopping if no improvement
            if no_improve_epochs >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        # Save metrics for this fold
        dict2 = {'correlation_coefficient':max(metric),'min_train_loss':min(loss_train),'min_test_loss':min(loss_test),'k_fold':fold+1}
        
        if loss_kind == 'pearson':
            write_good_record(dict1=params,dict2=dict2,file_path='good_record_metric_pearson.txt')
        elif loss_kind == 'pearson_mse':
            write_good_record(dict1=params,dict2=dict2,file_path='good_record_metric_pearson_mse.txt')
        elif loss_kind == 'mse':
            write_good_record(dict1=params,dict2=dict2,file_path='good_record_metric_mse.txt')
        else:
            print('Loss type error, please check!')
        
        test_pearson_kfold.append(max(metric))

    return -max(test_pearson_kfold)


# train(params,train_dataset,test_dataset)
if __name__ == '__main__':

    # Load and process data
    filename = '../data/Bacillus_subtilis.csv'
    guides, fit18s, essentials = read_data(filename=filename)

    # Create a DataFrame and export to CSV
    df = pd.DataFrame({
        'guide_rna': guides,
        'fitness': fit18s,
        'essential': essentials
    })
    df.to_csv('../data/Bacillus_sample.csv', index=False)

    # Convert sequences to model input format
    features_array, labels_array, biofeatures_array = make_dataset_sequences_bio(guides, fit18s, essentials)

    # Define K-fold cross-validation
    k_folds = 5
    kf = KFold(n_splits=k_folds, shuffle=True)

    # Set up training device
    params = {
        'device_num': 2,
        'dropout_rate1': 0.4320438222805047,
        'dropout_rate2': 0.12133028008904646,
        'dropout_rate_fc': 0.49166131520310447,
        'embedding_dim1': 64,
        'embedding_dim2': 128,
        'fc_hidden1': 87,
        'fc_hidden2': 8,
        'hidden_dim1': 512,
        'hidden_dim2': 1024,
        'l2_regularization': 2e-05,
        'latent_dim1': 256,
        'latent_dim2': 64,
        'num_head1': 4,
        'num_head2': 8,
        'seq_len': 20,
        'train_base_learning_rate': 0.0009764177415433284,
        'train_batch_size': 1024,
        'train_epochs_num': 500,
        'transformer_num_layers1': 3,
        'transformer_num_layers2': 11
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(params['device_num'])
    print('device =', device)

    # Define the loss type
    # Options: ['pearson', 'pearson_mse', 'mse']
    loss_kind = 'pearson_mse'

    # Run one trial training for validation
    train(params, features_array=features_array, bios_array=biofeatures_array, labels_array=labels_array)

    '''Start hyperparameter optimization'''
    # Define hyperparameter search space
    space = {
        'train_batch_size': hp.choice('train_batch_size', [1024]),
        'seq_len': hp.choice('seq_len', [20]),
        'device_num': hp.choice('device_num', [2]),
        'train_epochs_num': hp.choice('train_epochs_num', [500]),

        'train_base_learning_rate': hp.loguniform('train_base_learning_rate', -7, -4),

        'dropout_rate1': hp.uniform('dropout_rate1', 0.1, 0.5),
        'dropout_rate2': hp.uniform('dropout_rate2', 0.1, 0.5),
        'dropout_rate_fc': hp.uniform('dropout_rate_fc', 0.1, 0.5),

        'transformer_num_layers1': hp.randint('transformer_num_layers1', 1, 12),
        'transformer_num_layers2': hp.randint('transformer_num_layers2', 1, 12),
        
        # 'l2_regularization': hp.loguniform('l2_regularization', -8, -2),
        'l2_regularization': hp.choice('l2_regularization', [5e-5, 2e-5, 5e-6]),

        'num_head1': hp.choice('num_head1', [2, 4, 8, 16]),
        'num_head2': hp.choice('num_head2', [2, 4, 8, 16]),

        'hidden_dim1': hp.choice('hidden_dim1', [64, 128, 256, 512, 1024]),
        'latent_dim1': hp.choice('latent_dim1', [64, 128, 256, 512]),
        'embedding_dim1': hp.choice('embedding_dim1', [64, 128, 256, 512]),

        'hidden_dim2': hp.choice('hidden_dim2', [128, 256, 512, 1024]),
        'latent_dim2': hp.choice('latent_dim2', [64, 128, 256, 512]),
        'embedding_dim2': hp.choice('embedding_dim2', [64, 128, 256, 512]),

        'fc_hidden1': hp.randint('fc_hidden1', 64, 256),
        'fc_hidden2': hp.randint('fc_hidden2', 8, 64)
    }

    # Create a Trials object to log the search process
    trials = Trials()

    # Wrap training function as a hyperopt objective
    objective = lambda params: train(params, features_array=features_array, bios_array=biofeatures_array, labels_array=labels_array)

    # Run hyperparameter search
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=1000, trials=trials)

    # Print the best parameters
    print('Best parameters:', best)
