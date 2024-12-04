import hyperopt
from hyperopt import fmin, tpe, hp, Trials
import torch
from utils import *
from net import predict_transformerv2
from initialize import initialize_weights
from torch.utils.data import DataLoader,Dataset

import numpy as np
import pdb
import os
from sklearn.model_selection import KFold

def make_dataset_sequences_bio(guides, fit18s, oris, lights):

    features_array = []
    bios_array = []
    labels_array = []

    fit18s = np.array(fit18s)
    max_reads = np.max(fit18s) 
    min_reads = np.min(fit18s)

    print('max_reads = ',max_reads)
    print('min_reads = ',min_reads)

    number = 0
    base_choice_ori = ['+', '-']
    base_choice_light = ['100', '300', '0']

    for sequence, score, ori, light in zip(guides, fit18s, oris, lights):

        if len(sequence) < 20:

            print('length = ', len(sequence))
            print('sequence = ',sequence)
            continue
        
        # One-hot vector encoding
        if ori in base_choice_ori:

            if ori == '+':
                ori = np.array([1,0])
            
            else:
                ori = np.array([0,1])

        else:
            print("The input ori string is not in the given list and cannot be one-hot encoded.")
            continue

         # One-hot vector encoding
        if light in base_choice_light:

            if light == '100':
                light = np.array([1,0,0])
            
            elif light == '300':
                light = np.array([0,1,0])

            else :
                light = np.array([0,0,1])

        else:
            print("The input string light is not in the given list and cannot be one-hot encoded.")
            continue

        feature = Dimer_split_seqs(sequence[-20:])
        feature = np.array(feature)
        feature = feature.astype(int)

        label = (score - min_reads)/(max_reads -  min_reads)

        features_array.append(feature)
        bios_array.append(np.concatenate((light, ori)))
        labels_array.append(label)

        number += 1
        print('number = ',number)
    
    return np.array(features_array), np.array(labels_array), np.array(bios_array)


def read_data(filename):

    guides = []
    fit18s = []

    # bio
    lights = []
    oris = []

    with open(filename, 'r') as file:
        reader = csv.reader(file)
        header = next(reader) 
        
        guide_idx = header.index('guide') 
        fit18_idx = header.index('fitness')  

        light_idx = header.index('light|intensity') 
        ori_idx = header.index('ori')
        
        for row in reader:

            guide = row[guide_idx] 
            fit18 = row[fit18_idx] 

            light = row[light_idx] 
            ori = row[ori_idx]
        
            if isinstance(guide, str) and isinstance(fit18, str) and isinstance(light, str) and isinstance(ori, str) and guide != 'NA' and fit18 != 'NA':
            
                guides.append(guide)
                fit18s.append(float(fit18))

                lights.append(light)
                oris.append(ori)

            else:
                print("guide:", guide)
                print("fit18:", fit18)

    return guides, fit18s, lights, oris


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
    
    print('params = ',params)

    test_pearson_kfold = []

    for fold, (train_indices, val_indices) in enumerate(kf.split(features_array)):
        print(f"Fold {fold + 1}/{k_folds}")

        print('size of train datset is: ', len(train_indices))
        print('size of test datset is: ', len(val_indices))

        train_dataset = CustomDataset(features_array[train_indices], bios_array[train_indices], labels_array[train_indices])
        test_dataset = CustomDataset(features_array[val_indices], bios_array[val_indices], labels_array[val_indices])

        train_loader = DataLoader(train_dataset, batch_size=params['train_batch_size'], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=params['train_batch_size'], shuffle=False)

        print('Training set length =',len(train_loader))
        print('Test set length =',len(test_loader))

        print('start compose simple gan model')
        gen = predict_transformerv2.Predict_transformer_bio(params=params).to(device)

        initialize_weights(gen)
        print('successful compose simple gan model')

        opt_gen = torch.optim.Adam(gen.parameters(), lr=params['train_base_learning_rate'], weight_decay=params['l2_regularization'])
        loss_fc = torch.nn.MSELoss()

        loss_train =[]
        loss_test = []

        metric = []

        '''Start training'''
        for epoch in range(params['train_epochs_num']):
            
            
            # Adjusting the learning rate
            if epoch > 0 and epoch % 100 == 0:

                for param_group in opt_gen.param_groups:

                    print('Adjusting the learning rate')
                    param_group['lr'] = param_group['lr'] / 2.0

            loss_train_one_epoch = 0
            loss_test_one_epoch = 0

            loss_mse = 0
            loss_pier = 0
            
            # Start training
            gen.train()

            for data, bio, target in train_loader:
                
                data = data.to(device)
                target = target.to(device)
                bio = bio.to(device)

                output = gen(data,bio)
                output = torch.squeeze(output, dim=1)

                loss_gen = loss_fc(target.float(), output.float())
                loss_pi = loss_pierxun(target=target.float(),output=output.float())

                # print('*****loss_gen = ******',loss_gen)
                loss_gen = loss_gen.float()
                loss_pi = loss_pi.float()

                if loss_kind == 'pearson':
                    loss_all = -loss_pi

                elif loss_kind == 'pearson_mse':
                    loss_all = -loss_pi + loss_gen
                
                elif loss_kind == 'mse':
                    loss_all = loss_gen

                else:
                    print('The input loss function type is incorrect, please check!!!')

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
            
            # Test set starts testing
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
                
                # print('data.shape = ', data.shape)
                # print('target.shape = ', target.shape)
                # print('output.shape = ', output.shape)

                targets.append(target)
                outputs.append(output)

                # pdb.set_trace()
                # plot_test_prediction_result(output,target,epoch)

                loss_test_one_epoch += loss_gen.detach().cpu().numpy() 
            
            correlation_coefficient = compute_correlation_coefficient(torch.cat(targets, dim=0), torch.cat(outputs, dim=0))
            # pdb.set_trace()

            loss_test.append(loss_test_one_epoch/len(test_loader))

            if epoch % 10 == 0:
                
                print(
                        f"Epoch[{epoch}/{params['train_epochs_num']}] ****Test loss: {loss_test_one_epoch/len(test_loader):.6f}********test correlation_coefficient:{correlation_coefficient}"
                        )
            
            metric.append(correlation_coefficient)

            # Save loss and prediction model
            if correlation_coefficient > 0.7:
                
                if loss_kind == 'pearson':
                    torch.save(gen,'../model/bio_pearson_predict_{0}_mertric={1:.3f}.pth'.format(epoch, correlation_coefficient))
                
                elif loss_kind == 'pearson_mse':
                    torch.save(gen,'../model/bio_pearson_mse_predict_{0}_mertric={1:.3f}.pth'.format(epoch, correlation_coefficient))
                
                elif loss_kind == 'mse':
                    torch.save(gen,'../model/bio_mse_predict_{0}_mertric={1:.3f}.pth'.format(epoch, correlation_coefficient))
                
                else:
                    print('The loss function type is wrong, please check! ! ! !')

        # Storing metrics

        df = pd.DataFrame({
            'Epoch': range(1, len(loss_train) + 1), 
            'Loss_Train': loss_train,
            'Loss_Test': loss_test
        })

        dict2 = {'correlation_coefficient':max(metric),'min_train_loss':min(loss_train),'min_test_loss':min(loss_test),'k_fold':fold+1}
        
        if loss_kind == 'pearson':
            write_good_record(dict1=params,dict2=dict2,file_path='../record/good_record_metric_pearson.txt')# return  min(loss_test)
            file_path = f'../record/pearson_loss_data_pcc={correlation_coefficient:.3f}.csv'
            df.to_csv(file_path, index=False)
        
        elif loss_kind == 'pearson_mse':
            write_good_record(dict1=params,dict2=dict2,file_path='../record/good_record_metric_pearson_mse.txt')# return  min(loss_test)
            file_path = f'../record/pearson_mse_loss_data_pcc={correlation_coefficient:.3f}.csv'
            df.to_csv(file_path, index=False)

        elif loss_kind == 'mse':
            write_good_record(dict1=params,dict2=dict2,file_path='../record/good_record_metric_mse.txt')# return  min(loss_test)
            file_path = f'../record/mse_loss_data_pcc={correlation_coefficient:.3f}.csv'
            df.to_csv(file_path, index=False)

        else:
            print('The loss function type is wrong, please check! ! ! !')
        
        test_pearson_kfold.append(max(metric))

    return -max(test_pearson_kfold)


# train(params,train_dataset,test_dataset)
if __name__ == '__main__':

    # Processing Data
    filename = '../data/data_all_light.csv'
    guides, fit18s, lights, oris = read_data(filename=filename)
    features_array, labels_array, biofeatures_array = make_dataset_sequences_bio(guides, fit18s, oris, lights)

    # Defining K-fold cross validation
    k_folds = 5
    kf = KFold(n_splits=k_folds, shuffle=True)

    # Determine the GPU number for training
    params = {'device_num': 6, 'dropout_rate1': 0.3258494549467406, 'dropout_rate2': 0.2974783660130027, 'dropout_rate_fc': 0.3134874750986153, 'embedding_dim1': 64, 'embedding_dim2': 256, 'fc_hidden1': 109, 'fc_hidden2': 56, 'hidden_dim1': 1024, 'hidden_dim2': 256, 'l2_regularization': 5e-05, 'latent_dim1': 64, 'latent_dim2': 256, 'num_head1': 8, 'num_head2': 8, 'seq_len': 20, 'train_base_learning_rate': 0.0010350836441350173, 'train_batch_size': 512, 'train_epochs_num': 500, 'transformer_num_layers1': 4, 'transformer_num_layers2': 9}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(params['device_num'])
    print('device =',device)

    # Determine the loss type
    # loss_kind = ['pearson', 'pearson_mse', 'mse']
    loss_kind = 'pearson_mse'

    loss_kind = 'mse'
    if loss_kind == 'pearson':
        file_path = '../record/good_record_metric_pearson.txt'# return  min(loss_test)
        os.mknod(file_path)
    
    elif loss_kind == 'pearson_mse':
        file_path = '../record/good_record_metric_pearson_mse.txt'# return  min(loss_test)
        os.mknod(file_path)
    
    elif loss_kind == 'mse':
        file_path = '../record/good_record_metric_mse.txt'# return  min(loss_test)
        os.mknod(file_path)
    
    else:
        print('The input loss function type is incorrect, please check!!!')

    # Test run the main function
    train(params, features_array=features_array, bios_array=biofeatures_array, labels_array=labels_array)

    '''start search'''
    # Defining the search space for hyperparameters
    space = {

        'train_batch_size':hp.choice('train_batch_size',[1024]),
        'seq_len':hp.choice('seq_len',[20]),  
        'device_num':hp.choice('device_num',[6]),
        'train_epochs_num':hp.choice('train_epochs_num',[500]),

        'train_base_learning_rate': hp.loguniform('train_base_learning_rate', -7, -4),

        'dropout_rate1': hp.uniform('dropout_rate1', 0.1, 0.5),
        'dropout_rate2': hp.uniform('dropout_rate2', 0.1, 0.5),
        'dropout_rate_fc': hp.uniform('dropout_rate_fc', 0.1, 0.5),

        'transformer_num_layers1': hp.randint('transformer_num_layers1',1, 12),
        'transformer_num_layers2': hp.randint('transformer_num_layers2',1, 12),
        
        # 'l2_regularization': hp.loguniform('l2_regularization', -8, -2),
        'l2_regularization': hp.choice('l2_regularization', [5e-5,2e-5,5e-6]),

        'num_head1': hp.choice('num_head1', [2, 4, 8, 16]),
        'num_head2': hp.choice('num_head2', [2, 4, 8, 16]),

        'hidden_dim1': hp.choice('hidden_dim1',[64,128,256,512,1024]),
        'latent_dim1': hp.choice('latent_dim1', [64,128, 256,512]),
        'embedding_dim1': hp.choice('embedding_dim1',[64,128, 256,512]),

        'hidden_dim2': hp.choice('hidden_dim2',[128,256,512,1024]),
        'latent_dim2': hp.choice('latent_dim2', [64, 128, 256,512]),
        'embedding_dim2': hp.choice('embedding_dim2',[64, 128, 256,512]),

        'fc_hidden1': hp.randint('fc_hidden1',64, 256),
        'fc_hidden2': hp.randint('fc_hidden2',8, 64)
    }

    # Create a Trials object to track the optimization process
    trials = Trials()

    # Wrap the training function as an objective function suitable for hyperopt
    objective = lambda params: train(params, features_array=features_array, bios_array=biofeatures_array, labels_array=labels_array)
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=1000, trials=trials)

    # Printing Optimal Parameters
    print('Optimal Parameters: ', best)