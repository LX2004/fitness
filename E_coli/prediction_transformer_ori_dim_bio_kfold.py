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

class Seq_sample():
    def __init__(self, label, feature, essential, ori, coding, ntarget):
        self.label = label
        self.feature = feature
        self.essential = essential
        self.ori = ori
        self.coding = coding
        self.ntarget = ntarget

# 定义一个自定义数据集类
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

    # 存储交叉验证的pearson相关系数
    test_pearson_kfold = []

   # 循环遍历每个折叠
    for fold, (train_indices, val_indices) in enumerate(kf.split(features_array)):
        print(f"Fold {fold + 1}/{k_folds}")

        print('size of train datset is: ', len(train_indices))
        print('size of test datset is: ', len(val_indices))

        # 创建自定义数据集对象
        train_dataset = CustomDataset(features_array[train_indices], bios_array[train_indices], labels_array[train_indices])
        test_dataset = CustomDataset(features_array[val_indices], bios_array[val_indices], labels_array[val_indices])

        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=params['train_batch_size'], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=params['train_batch_size'], shuffle=False)

        # 查看数据集长度
        print('训练集长度 = ',len(train_loader))
        print('测试集长度 = ',len(test_loader))

        #实例模型
        print('start compose simple gan model')
        gen = predict_transformerv2.Predict_transformer(params=params).to(device)

        initialize_weights(gen)
        print('successful compose simple gan model')

        #定义优化器
        opt_gen = torch.optim.Adam(gen.parameters(), lr=params['train_base_learning_rate'], weight_decay=params['l2_regularization'])

        loss_fc = torch.nn.MSELoss()

        loss_train =[]
        loss_test = []

        metric = []

        '''开始训练'''
        for epoch in range(params['train_epochs_num']):
            
            
            # 调节学习速率
            if epoch > 0 and epoch % 100 == 0:
                for param_group in opt_gen.param_groups:
                    print('调节学习速率')
                    param_group['lr'] = param_group['lr'] / 2.0

            loss_train_one_epoch = 0
            loss_test_one_epoch = 0

            loss_mse = 0
            loss_pier = 0
            
            # 开始训练
            gen.train()

            for batch_idx, (data, bio, target) in enumerate(train_loader):
                
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
                    print('输入的损失函数类型有误，请检查！！！')

                opt_gen.zero_grad()
                loss_all.backward()
                opt_gen.step()

                loss_train_one_epoch += loss_all.item()
                loss_mse += loss_gen.item()
                loss_pier += loss_pi.item()
                
            loss_train.append(loss_train_one_epoch)

            if epoch % 10 == 0:
                print(
                        f"Epoch[{epoch}/{params['train_epochs_num']}] ****Train loss: {loss_train_one_epoch/len(train_loader):.6f}****MSE loss: {loss_mse/len(train_loader):.6f}****Pierxun loss: {loss_pier/len(train_loader):.6f}"
                        )
            
            # 测试集开始测试
            gen.eval()
            targets = []
            outputs = []

            for batch_idx, (data,bio,target) in enumerate(test_loader):
                
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

            loss_test.append(loss_test_one_epoch)

            if epoch % 10 == 0:
                
                print(
                        f"Epoch[{epoch}/{params['train_epochs_num']}] ****Test loss: {loss_test_one_epoch/len(test_loader):.6f}********test correlation_coefficient:{correlation_coefficient}"
                        )
            
            metric.append(correlation_coefficient)

            # 保存loss和预测模型
            if correlation_coefficient > 0.633:
                np.save('./result/loss_train_pierxun={0}.npy'.format(correlation_coefficient),np.array(loss_train))
                np.save('./result/loss_test_pierxun={0}.npy'.format(correlation_coefficient),np.array(loss_test))
                
                if loss_kind == 'pearson':
                    torch.save(gen,'./models/pearson_predict_{0}_mertric={1}.pth'.format(epoch,correlation_coefficient))
                
                elif loss_kind == 'pearson_mse':
                    torch.save(gen,'./models/pearson_mse_predict_{0}_mertric={1}.pth'.format(epoch,correlation_coefficient))
                
                elif loss_kind == 'mse':
                    torch.save(gen,'./models/mse_predict_{0}_mertric={1}.pth'.format(epoch,correlation_coefficient))
                
                else:
                    print('损失函数类型出错，请检查！！！！')
        # 存储指标
        dict2 = {'correlation_coefficient':max(metric),'min_train_loss':min(loss_train),'min_test_loss':min(loss_test),'k_fold':fold+1}
        
        if loss_kind == 'pearson':
            write_good_record(dict1=params,dict2=dict2,file_path='good_record_metric_pearson.txt')# return  min(loss_test)
        
        elif loss_kind == 'pearson_mse':
            write_good_record(dict1=params,dict2=dict2,file_path='good_record_metric_pearson_mse.txt')# return  min(loss_test)
        
        elif loss_kind == 'mse':
            write_good_record(dict1=params,dict2=dict2,file_path='good_record_metric_mse.txt')# return  min(loss_test)
        
        else:
            print('损失函数类型出错，请检查！！！！')
        
        test_pearson_kfold.append(max(metric))

    return -max(test_pearson_kfold)


# train(params,train_dataset,test_dataset)
if __name__ == '__main__':

    # 定义K折交叉验证
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

    # 确定训练的GPU编号
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(params['device_num'])
    print('device =',device)

    # 创建一个空列表来存储数据点
    data_points = []

    # 遍历/sample文件夹中的文件
    sample_folder = "./samples_more_than_3.5"  # fit18值大于-3.5

    for filename in os.listdir(sample_folder):

        if filename.endswith(".pkl"):  # 假设您保存的是pickle文件

            file_path = os.path.join(sample_folder, filename)

            with open(file_path, 'rb') as file:

                data_point = pickle.load(file)
                data_points.append(data_point)

    # 提取特征和标签
    features = []
    labels = []
    biofeatures = []

    for data_point in data_points:

        features.append(data_point.feature)
        labels.append(data_point.label)

        essential = data_point.essential
        ori = data_point.ori
        coding = data_point.coding
        ntarget = data_point.ntarget

        # print('essential.shape', essential.shape)
        # print('ori.shape', ori.shape)
        # print('coding.shape', coding.shape)

        # 将浮点数添加到堆叠后的数组中
        biofeature = np.concatenate((essential, ori, coding, np.array([ntarget])))

        # print('biofeature.shape = ', biofeature.shape)
        biofeatures.append(biofeature)


    # 将特征和标签转换为NumPy数组
    features_array = np.array(features)
    labels_array = np.array(labels)
    biofeatures_array = np.array(biofeatures)

    # 确定loss种类
    # loss_kind = ['pearson', 'pearson_mse', 'mse']
    loss_kind = 'mse'

    # 试运行一次主函数
    train(params, features_array=features_array, bios_array=biofeatures_array, labels_array=labels_array)


    '''开始搜索'''
    # 定义超参数的搜索空间
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

    # 创建Trials对象以跟踪优化过程
    trials = Trials()

    # 将训练函数包装为适用于hyperopt的目标函数
    objective = lambda params: train(params, features_array=features_array, bios_array=biofeatures_array, labels_array=labels_array)
    # 运行优化
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=1000, trials=trials)

    # 打印最佳参数
    print('最佳参数:', best)
