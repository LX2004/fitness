import hyperopt
from hyperopt import fmin, tpe, hp, Trials
import torch
from utils import *
from net import predict_transformerv2
from initialize import initialize_weights
from torch.utils.data import DataLoader, Dataset

import numpy as np
import pdb
import pickle
import os
from sklearn.model_selection import KFold


def make_dataset_sequences_bio(guides, fit18s, conditions, oris, codings):

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
    base_conditions = ['GP', 'CP', 'SynP']

    for sequence, score, condition, ori, coding in zip(guides, fit18s, conditions, oris, codings):

        if len(sequence) < 20:
            print('length = ', len(sequence))
            print('sequence = ', sequence)
            continue

        # One-hot style encoding for coding (True/False)
        coding = str(coding)
        if coding == 'True':
            bio = np.array([1, 0])
        elif coding == 'False':
            bio = np.array([0, 1])
        else:
            print(f"Input coding = {coding} is not in the allowed set; cannot perform one-hot encoding.")
            continue

        # One-hot style encoding for ori (+/-)
        ori = str(ori)
        if ori == '+':
            bio = np.concatenate([bio, np.array([1, 0])])
        elif ori == '-':
            bio = np.concatenate([bio, np.array([0, 1])])
        else:
            print(f"Input ori = {ori} is not in the allowed set; cannot perform one-hot encoding.")
            continue

        # One-hot style encoding for condition (GP/CP/SynP)
        if condition in base_conditions:
            if condition == 'GP':
                bio = np.concatenate([bio, np.array([1, 0, 0])])
            elif condition == 'CP':
                bio = np.concatenate([bio, np.array([0, 1, 0])])
            else:
                bio = np.concatenate([bio, np.array([0, 0, 1])])
        else:
            print("Input condition is not in the allowed set; cannot perform one-hot encoding.")
            continue

        # Use the last 20-nt of the sequence as input features
        feature = Dimer_split_seqs(sequence[-20:])
        feature = np.array(feature).astype(int)

        # Min–max normalize the label within this set
        label = (score - min_reads) / (max_reads - min_reads)

        features_array.append(feature)
        bios_array.append(bio)
        labels_array.append(label)

        number += 1
        print('number = ', number)

    return np.array(features_array), np.array(labels_array), np.array(bios_array)


def read_data(filename):

    guides = []
    fit18s = []
    oris = []
    codings = []

    # Bio annotations
    conditions = []
    df = pd.read_csv(filename)

    # Condition 1: GP (heterotrophic growth)
    number = 0

    for grna, fitness1, fitness2, fitness3, ori, coding in zip(
        df['sgRNA'], df['GP3-1'], df['GP3-2'], df['GP3-3'], df['ori'], df['coding']
    ):
        # Cast to float
        fitness1 = float(fitness1)
        fitness2 = float(fitness2)
        fitness3 = float(fitness3)

        # Exclude triplicates with high dispersion
        if max(fitness1, fitness2, fitness3) - min(fitness1, fitness2, fitness3) >= 0.5:
            continue

        guides.append(grna.upper())
        fit18s.append((fitness1 + fitness2 + fitness3) / 3)
        number += 1

        codings.append(coding)
        oris.append(ori)

    conditions += number * ['GP']

    # Condition 2: CP (autotrophic growth)
    number = 0
    for grna, fitness1, fitness2, fitness3, ori, coding in zip(
        df['sgRNA'], df['GP3-1'], df['GP3-2'], df['GP3-3'], df['ori'], df['coding']
    ):
        fitness1 = float(fitness1)
        fitness2 = float(fitness2)
        fitness3 = float(fitness3)

        if max(fitness1, fitness2, fitness3) - min(fitness1, fitness2, fitness3) >= 0.5:
            continue

        guides.append(grna.upper())
        fit18s.append((fitness1 + fitness2 + fitness3) / 3)

        codings.append(coding)
        oris.append(ori)

        number += 1

    conditions += number * ['CP']

    # Condition 3: SynP (syngas)
    number = 0
    for grna, fitness1, fitness2, fitness3, ori, coding in zip(
        df['sgRNA'], df['GP3-1'], df['GP3-2'], df['GP3-3'], df['ori'], df['coding']
    ):
        fitness1 = float(fitness1)
        fitness2 = float(fitness2)
        fitness3 = float(fitness3)

        if max(fitness1, fitness2, fitness3) - min(fitness1, fitness2, fitness3) >= 0.5:
            continue

        guides.append(grna.upper())
        fit18s.append((fitness1 + fitness2 + fitness3) / 3)

        codings.append(coding)
        oris.append(ori)

        number += 1

    conditions += number * ['SynP']
    number = 0

    return guides, fit18s, conditions, oris, codings


def read_data(filename, choose=0):

    guides = []
    fit18s = []

    # Bio annotations
    conditions = []
    oris = []
    codings = []

    df = pd.read_csv(filename)

    # Condition 1: GP (heterotrophic growth)
    number = 0

    for grna, fitness1, fitness2, fitness3, ori, coding in zip(
        df['sgRNA'], df['GP3-1'], df['GP3-2'], df['GP3-3'], df['ori'], df['coding']
    ):
        # Cast to float
        fitness1 = float(fitness1)
        fitness2 = float(fitness2)
        fitness3 = float(fitness3)

        # Exclude triplicates with high dispersion
        if max(fitness1, fitness2, fitness3) - min(fitness1, fitness2, fitness3) >= 0.5:
            continue

        guides.append(grna.upper())

        if choose == 1:
            fit18s.append(fitness1)
        elif choose == 2:
            fit18s.append(fitness2)
        elif choose == 3:
            fit18s.append(fitness3)
        else:
            fit18s.append((fitness1 + fitness2 + fitness3) / 3)

        oris.append(ori)
        codings.append(coding)

        number += 1

    conditions += number * ['GP']

    # Condition 2: CP (autotrophic growth)
    number = 0
    for grna, fitness1, fitness2, fitness3, ori, coding in zip(
        df['sgRNA'], df['GP3-1'], df['GP3-2'], df['GP3-3'], df['ori'], df['coding']
    ):
        fitness1 = float(fitness1)
        fitness2 = float(fitness2)
        fitness3 = float(fitness3)

        if max(fitness1, fitness2, fitness3) - min(fitness1, fitness2, fitness3) >= 0.5:
            continue

        guides.append(grna.upper())

        if choose == 1:
            fit18s.append(fitness1)
        elif choose == 2:
            fit18s.append(fitness2)
        elif choose == 3:
            fit18s.append(fitness3)
        else:
            fit18s.append((fitness1 + fitness2 + fitness3) / 3)

        oris.append(ori)
        codings.append(coding)
        number += 1

    conditions += number * ['CP']

    # Condition 3: SynP (syngas)
    number = 0
    for grna, fitness1, fitness2, fitness3, ori, coding in zip(
        df['sgRNA'], df['GP3-1'], df['GP3-2'], df['GP3-3'], df['ori'], df['coding']
    ):
        # Cast to float
        fitness1 = float(fitness1)
        fitness2 = float(fitness2)
        fitness3 = float(fitness3)

        # Exclude triplicates with high dispersion
        if max(fitness1, fitness2, fitness3) - min(fitness1, fitness2, fitness3) >= 0.5:
            continue

        guides.append(grna.upper())

        if choose == 1:
            fit18s.append(fitness1)
        elif choose == 2:
            fit18s.append(fitness2)
        elif choose == 3:
            fit18s.append(fitness3)
        else:
            fit18s.append((fitness1 + fitness2 + fitness3) / 3)

        number += 1

        oris.append(ori)
        codings.append(coding)

    conditions += number * ['SynP']
    number = 0

    return guides, fit18s, conditions, oris, codings


# Custom dataset
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

    # Store Pearson coefficients from k-folds
    test_pearson_kfold = []

    # Iterate over k-folds
    for fold, (train_indices, val_indices) in enumerate(kf.split(features_array)):

        best_val_loss = float('inf')  # Best validation loss tracker
        no_improve_epochs = 0         # Epochs without improvement

        print(f"Fold {fold + 1}/{k_folds}")

        print('size of train dataset is: ', len(train_indices))
        print('size of test dataset is: ', len(val_indices))

        # Build datasets
        train_dataset = CustomDataset(
            features_array[train_indices], bios_array[train_indices], labels_array[train_indices]
        )
        test_dataset = CustomDataset(
            features_array[val_indices], bios_array[val_indices], labels_array[val_indices]
        )

        # DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=params['train_batch_size'], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=params['train_batch_size'], shuffle=False)

        # Inspect loader lengths
        print('Train loader length = ', len(train_loader))
        print('Test loader length = ', len(test_loader))

        # Instantiate model
        print('start compose simple gan model')
        gen = predict_transformerv2.Predict_E_lim_ori_coding(params=params).to(device)

        initialize_weights(gen)
        print('successful compose simple gan model')

        # Optimizer & loss
        opt_gen = torch.optim.Adam(gen.parameters(), lr=params['train_base_learning_rate'],
                                   weight_decay=params['l2_regularization'])
        loss_fc = torch.nn.MSELoss()

        loss_train = []
        loss_test = []

        metric = []

        # ===== Training loop =====
        for epoch in range(params['train_epochs_num']):

            # Adjust learning rate every 100 epochs
            if epoch > 0 and epoch % 100 == 0:
                for param_group in opt_gen.param_groups:
                    print('Adjust learning rate')
                    param_group['lr'] = param_group['lr'] / 2.0

            loss_train_one_epoch = 0
            loss_test_one_epoch = 0

            loss_mse = 0
            loss_pier = 0

            # Train
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
                    print('Invalid loss function type; please check.')
                    loss_all = loss_gen  # fallback

                opt_gen.zero_grad()
                loss_all.backward()
                opt_gen.step()

                loss_train_one_epoch += loss_all.item()
                loss_mse += loss_gen.item()
                loss_pier += loss_pi.item()

            loss_train.append(loss_train_one_epoch / len(train_loader))

            if epoch % 10 == 0:
                print(
                    f"Epoch[{epoch}/{params['train_epochs_num']}] ****Train loss: {loss_train_one_epoch/len(train_loader):.6f}****MSE loss: {loss_mse/len(train_loader):.6f}****Pierxun loss: {loss_pier/len(train_loader):.6f}"
                )

            # Evaluate on validation set
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

            correlation_coefficient, spearman = compute_correlation_coefficient(
                torch.cat(targets, dim=0), torch.cat(outputs, dim=0)
            )

            loss_test.append(loss_test_one_epoch / len(test_loader))

            # Track best validation loss
            if loss_test_one_epoch / len(test_loader) < best_val_loss:
                best_val_loss = loss_test_one_epoch / len(test_loader)
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            if epoch % 10 == 0:
                print(
                    f"Epoch[{epoch}/{params['train_epochs_num']}] ****Test loss: {loss_test_one_epoch/len(test_loader):.6f}********Test PCC:{correlation_coefficient}********Test spearman:{spearman}"
                )

            metric.append(correlation_coefficient)

            # Save model if metric improves beyond threshold
            if correlation_coefficient > 0.446:

                if loss_kind == 'pearson':
                    torch.save(gen, '../models/ori_coding_pearson_PCC=_{0}_spearman={1}.pth'.format(correlation_coefficient, spearman))

                elif loss_kind == 'pearson_mse':
                    torch.save(gen, '../models/ori_coding_pearson_mse_PCC=_{0}_spearman={1}.pth'.format(correlation_coefficient, spearman))

                elif loss_kind == 'mse':
                    torch.save(gen, '../models/ori_coding_mse_PCC=_{0}_spearman={1}.pth'.format(correlation_coefficient, spearman))

                else:
                    print('Loss function type error; please check.')

            # Learning rate decay when no improvement
            if no_improve_epochs > 0 and no_improve_epochs % 10:
                for param_group in opt_gen.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.85

            # Early stopping
            if no_improve_epochs >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        # Log metrics for this fold
        dict2 = {
            'correlation_coefficient': max(metric),
            'min_train_loss': min(loss_train),
            'min_test_loss': min(loss_test),
            'k_fold': fold + 1
        }

        if loss_kind == 'pearson':
            write_good_record(dict1=params, dict2=dict2, file_path='ori_coding_record_metric_pearson.txt')

        elif loss_kind == 'pearson_mse':
            write_good_record(dict1=params, dict2=dict2, file_path='ori_coding_record_metric_pearson_mse.txt')

        elif loss_kind == 'mse':
            write_good_record(dict1=params, dict2=dict2, file_path='ori_coding_record_metric_mse.txt')

        else:
            print('Loss function type error; please check.')

        test_pearson_kfold.append(max(metric))
        test_pearson_kfold.append(max(metric))

    return -max(test_pearson_kfold)


# Select input type: 0 = average of triplicates, 1 = first replicate, 2 = second replicate, 3 = third replicate
kind = 1

# train(params,train_dataset,test_dataset)
if __name__ == '__main__':

    # Load and sanitize raw data
    filename = '../data/E_limosum_essential_ori_coding.csv'

    df = pd.read_csv(
        filename,
        usecols=["sgRNA", "fitness", "condition", "ori", "coding"],
        encoding="utf-8-sig",
        dtype={"fitness": float}  # enforce float for fitness only
    )

    df = df.dropna(how="any").reset_index(drop=True)
    df["sgRNA"] = df["sgRNA"].str.upper()  # uppercase sgRNA

    guides = df["sgRNA"].tolist()
    fit18s = df["fitness"].tolist()
    conditions = df["condition"].tolist()
    oris = df["ori"].tolist()
    codings = df["coding"].tolist()

    # Alternative loader using replicated columns:
    # guides, fit18s, conditions, oris, codings = read_data(filename=filename, choose=kind)

    features_array, labels_array, biofeatures_array = make_dataset_sequences_bio(
        guides, fit18s, conditions, oris, codings
    )

    # K-fold cross-validation
    k_folds = 5
    kf = KFold(n_splits=k_folds, shuffle=True)

    # Device setup
    params = {
        'device_num': 1, 'dropout_rate1': 0.4590779857411303, 'dropout_rate2': 0.11504358270272816,
        'dropout_rate_fc': 0.4790614993037541, 'embedding_dim1': 256, 'embedding_dim2': 256,
        'fc_hidden1': 182, 'fc_hidden2': 11, 'hidden_dim1': 512, 'hidden_dim2': 512,
        'l2_regularization': 2e-05, 'latent_dim1': 256, 'latent_dim2': 256, 'num_head1': 16,
        'num_head2': 16, 'seq_len': 20, 'train_base_learning_rate': 0.0010120466640366524,
        'train_batch_size': 1024, 'train_epochs_num': 500, 'transformer_num_layers1': 7,
        'transformer_num_layers2': 11
    }
    # params = {'device_num': 1, 'dropout_rate1': 0.3258494549467406, 'dropout_rate2': 0.2974783660130027, 'dropout_rate_fc': 0.3134874750986153, 'embedding_dim1': 64, 'embedding_dim2': 256, 'fc_hidden1': 109, 'fc_hidden2': 56, 'hidden_dim1': 1024, 'hidden_dim2': 256, 'l2_regularization': 5e-05, 'latent_dim1': 64, 'latent_dim2': 256, 'num_head1': 8, 'num_head2': 8, 'seq_len': 20, 'train_base_learning_rate': 0.0010350836441350173, 'train_batch_size': 512, 'train_epochs_num': 500, 'transformer_num_layers1': 4, 'transformer_num_layers2': 8}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(params['device_num'])
    print('device =', device)

    # Choose the loss type
    # loss_kind = ['pearson', 'pearson_mse', 'mse']
    loss_kind = 'pearson_mse'

    # Dry run once
    train(params, features_array=features_array, bios_array=biofeatures_array, labels_array=labels_array)

    # ===== Hyperparameter search =====
    # Define the hyperparameter space
    space = {
        'train_batch_size': hp.choice('train_batch_size', [1024]),
        'seq_len': hp.choice('seq_len', [20]),
        'device_num': hp.choice('device_num', [1]),
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

    # Track optimization with Trials
    trials = Trials()

    # Wrap train() as a hyperopt objective
    objective = lambda params: train(params, features_array=features_array, bios_array=biofeatures_array, labels_array=labels_array)

    # Run optimization
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=1000, trials=trials)

    # Report best hyperparameters
    print('Best hyperparameters:', best)
