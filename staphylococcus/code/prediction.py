import hyperopt
from hyperopt import fmin, tpe, hp, Trials
import torch
from utils import *
from net import predict_transformerv2
from initialize import initialize_weights
from torch.utils.data import DataLoader,Dataset

import numpy as np
import pdb

from sklearn.model_selection import KFold


# Define a custom sample class
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

    print('max_reads = ', max_reads)
    print('min_reads = ', min_reads)

    pdb.set_trace()
    number = 0

    for sequence, score, essential in zip(guides, fit18s, essentials):

        if len(sequence) < 20:
            print('length = ', len(sequence))
            print('sequence = ', sequence)
            continue
        
        # One-hot encode essentiality
        essential = str(essential)

        if essential == 'essential':
            ori = np.array([1,0])
        elif essential == 'neutral':
            ori = np.array([0,1])
        else:
            print(f"Input Essential = {essential} is not in the allowed list, cannot one-hot encode.")
            continue

        feature = Dimer_split_seqs(sequence)  # Encode entire sequence
        feature = np.array(feature).astype(int)

        label = (score - min_reads)/(max_reads - min_reads)

        features_array.append(feature)
        bios_array.append(ori)
        labels_array.append(label)

        number += 1
        print('number = ', number)
    
    return np.array(features_array), np.array(labels_array), np.array(bios_array)


def read_data(filename):

    import math
    guides = []
    fit18s = []
    essentials = []

    df = pd.read_csv(filename)

    # Use fitness from Newman strain (avoid Dalbavancin)
    number = 0
    for variant_guide, essential, fitness in zip(df['sgRNA sequence'], df['essential(Newman)'], df['fitness(Newman)']):
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
    print('params = ', params)

    # Store Pearson correlation from cross-validation
    test_pearson_kfold = []

    # Cross-validation folds
    for fold, (train_indices, val_indices) in enumerate(kf.split(features_array)):

        best_val_loss = float('inf')  # Best validation loss
        no_improve_epochs = 0  # Counter for early stopping

        print(f"Fold {fold + 1}/{k_folds}")
        print('Train dataset size: ', len(train_indices))
        print('Test dataset size: ', len(val_indices))

        # Create custom dataset
        train_dataset = CustomDataset(features_array[train_indices], bios_array[train_indices], labels_array[train_indices])
        test_dataset = CustomDataset(features_array[val_indices], bios_array[val_indices], labels_array[val_indices])

        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=params['train_batch_size'], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=params['train_batch_size'], shuffle=False)

        print('Train loader length = ', len(train_loader))
        print('Test loader length = ', len(test_loader))

        # Initialize model
        print('Start building the model')
        gen = predict_transformerv2.Predict_Staphylococcus_aureus(params=params).to(device)
        initialize_weights(gen)
        print('Model built successfully')

        # Optimizer and loss
        opt_gen = torch.optim.Adam(gen.parameters(), lr=params['train_base_learning_rate'], weight_decay=params['l2_regularization'])
        loss_fc = torch.nn.MSELoss()

        loss_train = []
        loss_test = []
        metric = []

        '''Training loop'''
        for epoch in range(params['train_epochs_num']):

            # Adjust learning rate
            if epoch > 0 and epoch % 100 == 0:
                for param_group in opt_gen.param_groups:
                    print('Adjusting learning rate')
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

                output = gen(data, bio)
                output = torch.squeeze(output, dim=1)

                loss_gen = loss_fc(target.float(), output.float())
                loss_pi = loss_pierxun(target=target.float(), output=output.float())

                loss_gen = loss_gen.float()
                loss_pi = loss_pi.float()

                if loss_kind == 'pearson':
                    loss_all = -loss_pi
                elif loss_kind == 'pearson_mse':
                    loss_all = -loss_pi + loss_gen
                elif loss_kind == 'mse':
                    loss_all = loss_gen
                else:
                    print('Invalid loss type specified!')

                opt_gen.zero_grad()
                loss_all.backward()
                opt_gen.step()

                loss_train_one_epoch += loss_all.item()
                loss_mse += loss_gen.item()
                loss_pier += loss_pi.item()
                
            loss_train.append(loss_train_one_epoch / len(train_loader))

            if epoch % 10 == 0:
                print(
                    f"Epoch[{epoch}/{params['train_epochs_num']}] ****Train loss: {loss_train_one_epoch/len(train_loader):.6f}****MSE loss: {loss_mse/len(train_loader):.6f}****Pearson loss: {loss_pier/len(train_loader):.6f}"
                )
            
            # Evaluation on test set
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
            loss_test.append(loss_test_one_epoch / len(test_loader))

            # Early stopping condition
            if loss_test_one_epoch / len(test_loader) < best_val_loss:
                best_val_loss = loss_test_one_epoch / len(test_loader)
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            if epoch % 10 == 0:
                print(f"Epoch[{epoch}/{params['train_epochs_num']}] ****Test loss: {loss_test_one_epoch/len(test_loader):.6f}****Test Pearson: {correlation_coefficient}")
            
            metric.append(correlation_coefficient)

            # Save model if performance is good
            if correlation_coefficient > 0.85:
                if loss_kind == 'pearson':
                    torch.save(gen, f'../models/bio_pearson_predict_{epoch}_mertric={correlation_coefficient}.pth')
                elif loss_kind == 'pearson_mse':
                    torch.save(gen, f'../models/bio_pearson_mse_predict_{epoch}_mertric={correlation_coefficient}.pth')
                elif loss_kind == 'mse':
                    torch.save(gen, f'../models/bio_mse_predict_{epoch}_mertric={correlation_coefficient}.pth')
                else:
                    print('Loss type error, please check!')

            # Learning rate decay logic
            if no_improve_epochs > 0 and no_improve_epochs % 10 == 0:
                for param_group in opt_gen.param_groups:
                    param_group['lr'] *= 0.85

            # Early stopping logic
            if no_improve_epochs >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        # Save metric summary
        dict2 = {
            'correlation_coefficient': max(metric),
            'min_train_loss': min(loss_train),
            'min_test_loss': min(loss_test),
            'k_fold': fold + 1
        }

        if loss_kind == 'pearson':
            write_good_record(dict1=params, dict2=dict2, file_path='good_record_metric_pearson.txt')
        elif loss_kind == 'pearson_mse':
            write_good_record(dict1=params, dict2=dict2, file_path='good_record_metric_pearson_mse.txt')
        elif loss_kind == 'mse':
            write_good_record(dict1=params, dict2=dict2, file_path='good_record_metric_mse.txt')
        else:
            print('Loss type error, please check!')

        test_pearson_kfold.append(max(metric))

    return -max(test_pearson_kfold)
