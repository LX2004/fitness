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

        print('Training set length = ',len(train_loader))
        print('Test set length =',len(test_loader))


        print('start compose simple gan model')
        gen = predict_transformerv2.Predict_transformer(params=params).to(device)

        initialize_weights(gen)
        print('successful compose simple gan model')

        opt_gen = torch.optim.Adam(gen.parameters(), lr=params['train_base_learning_rate'], weight_decay=params['l2_regularization'])
        loss_fc = torch.nn.MSELoss()

        loss_train =[]
        loss_test = []

        metric = []
        for epoch in range(params['train_epochs_num']):
            

            if epoch > 0 and epoch % 100 == 0:
                for param_group in opt_gen.param_groups:

                    print('Adjusting the learning rate')
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

            gen.eval()
            targets = []
            outputs = []

            for data,bio,target in test_loader:
                
                data = data.to(device)
                target = target.to(device)
                bio = bio.to(device)

                output = gen(data,bio)
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

            if correlation_coefficient > 0.633:

                # np.save('./result/loss_train_pierxun={0}.npy'.format(correlation_coefficient),np.array(loss_train))
                # np.save('./result/loss_test_pierxun={0}.npy'.format(correlation_coefficient),np.array(loss_test))
                
                if loss_kind == 'pearson':
                    torch.save(gen,'../model/pearson_predict_{0}_mertric={1:.3f}.pth'.format(epoch,correlation_coefficient))
                
                elif loss_kind == 'pearson_mse':
                    torch.save(gen,'../model/pearson_mse_predict_{0}_mertric={1:.3f}.pth'.format(epoch,correlation_coefficient))
                
                elif loss_kind == 'mse':
                    torch.save(gen,'../model/mse_predict_{0}_mertric={1:.3f}.pth'.format(epoch,correlation_coefficient))
                
                else:
                    print('The input loss function type is incorrect, please check!!!')

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
            print('The input loss function type is incorrect, please check!!!')
        
        test_pearson_kfold.append(max(metric))

    return -max(test_pearson_kfold)


def read_data(filename):

    guides = []
    fit18s = []
    fit75s = []

    # bio
    essentials = []
    oris = []

    codings = []
    ntargets = []

    with open(filename, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)  

        guide_idx = header.index('guide')  
        fit18_idx = header.index('fit18')  
        fit75_idx = header.index('fit75')  

        essential_idx = header.index('essential')  
        ori_idx = header.index('ori')  

        coding_idx = header.index('coding') 
        ntargets_idx = header.index('ntargets')  
        
        for row in reader:

            guide = row[guide_idx]  
            fit18 = row[fit18_idx]  
            fit75 = row[fit75_idx]  

            essential = row[essential_idx]  
            ori = row[ori_idx]  

            coding = row[coding_idx] 
            ntarget = row[ntargets_idx]  

            if isinstance(guide, str) and isinstance(fit18, str) and isinstance(essential, str) and isinstance(ori, str) and isinstance(ori, str):

                guides.append(guide)
                fit18s.append(float(fit18))
                fit75s.append(float(fit75))

                essentials.append(essential)
                oris.append(ori)

                codings.append(coding)
                ntargets.append(float(ntarget))

            else:
                print("guide:", guide)
                print("fit18:", fit18)
    return guides, fit18s, essentials, oris, codings, ntargets, fit75s

def make_dataset_for_find_prometer(guides, fit18s, essentials, oris, codings, ntargets, fit75s):
    
    max_reads = np.max(np.array(fit18s)) 
    min_reads = -3.5

    ntargets_max = max(ntargets)
    ntargets_min = min(ntargets)

    print('ntargets_max = ', ntargets_max)
    print('ntargets_min = ', ntargets_min)

    print('max_reads = ',max_reads)
    print('min_reads = ',min_reads)

    features = []
    labels = []
    biofeatures = []

    number = 0

    for sequence, score, essential, ori, coding, ntarget in zip(guides, fit18s, essentials, oris, codings, ntargets):

        if len(sequence) != 20 or score < -3.5:

            print('length = ', len(sequence))
            print('sequence = ',sequence)
            continue

        feature = Dimer_split_seqs(sequence)  
        feature = np.array(feature)
        feature = feature.astype(int)

        essential_feature = encode_essential(essential)
        ori_feature = encode_ori(ori)
        coding_feature = encode_coding(coding)
        ntarget_feature = (ntarget - ntargets_min)/(ntargets_max - ntargets_min)

        label = (score - min_reads)/(max_reads -  min_reads)
        biofeature = np.concatenate((essential_feature, ori_feature, coding_feature, np.array([ntarget_feature])))

        features.append(features)
        labels.append(label)
        biofeatures.append(biofeature)

        number += 1
        print('number = ',number) 
        
    print('save np Data')
    return np.array(features), np.array(labels), np.array(biofeatures)


if __name__ == '__main__':

    k_folds = 5
    kf = KFold(n_splits=k_folds, shuffle=True)

    params = {
        'device_num': 2, 
        'dropout_rate1': 0.13798726336989411, 'dropout_rate2': 0.4517850417291053, 
        'dropout_rate_fc': 0.3385214251346743, 
        
        'embedding_dim1': 256, 'embedding_dim2': 64, 
        'fc_hidden1': 127, 'fc_hidden2': 36, 
        'hidden_dim1': 64, 'hidden_dim2': 256, 
        'l2_regularization': 5e-05, 
        
        'latent_dim1': 128, 'latent_dim2': 128, 
        
        'num_head1': 8, 'num_head2': 4, 
        'seq_len': 20, 'train_base_learning_rate': 0.00200054708177843, 
        
        'train_batch_size': 256, 'train_epochs_num': 500, 
        'transformer_num_layers1': 8, 'transformer_num_layers2': 5, 
        }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(params['device_num'])
    print('device =',device)
    
    filename = "../data/screen_data.csv"
    guides, fit18s, essentials, oris, codings, ntargets, fit75s = read_data(filename=filename)
    features_array ,labels_array, biofeatures_array = make_dataset_for_find_prometer(guides, fit18s, essentials, oris, codings, ntargets, fit75s)
    print('End of data processing')

    # loss_kind = ['pearson', 'pearson_mse', 'mse']
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

    train(params, features_array=features_array, bios_array=biofeatures_array, labels_array=labels_array)
    space = {

        'train_batch_size':hp.choice('train_batch_size',[512]),
        'seq_len':hp.choice('seq_len',[20]),  
        'device_num':hp.choice('device_num',[2]),
        'train_epochs_num':hp.choice('train_epochs_num',[400]),

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

    trials = Trials()
    objective = lambda params: train(params, features_array=features_array, bios_array=biofeatures_array, labels_array=labels_array)
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=1000, trials=trials)

    print('Optimal parameters:', best)
