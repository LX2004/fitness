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
        
        # 进行独热向量编码
        essential = str(essential)

        if essential == 'essential':
        
            ori = np.array([1,0])
        
        elif essential == 'neutral':
            ori = np.array([0,1])
            
        else:
            print(f"输入的 Essential = {essential} 字符串不在给定的列表中，无法进行独热向量编码。")
            continue

        feature = Dimer_split_seqs(sequence)  # 所有序列作为输入
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

    # 生物信息
    essentials = []
    df = pd.read_csv(filename)

    # fitness(NCTC8325-4),fitness(Newman),fitness(Dalbavancin), 其中Dalbavancin尽量不使用

    number = 0
    for variant_guide, essential, fitness in zip(df['sgRNA sequence'], df['essential(Newman)'], df['fitness(Newman)']):
        
        # 转化为 float 类型
        fitness = float(fitness)

        if math.isnan(fitness):
            print(f'fitness is {fitness}!!!')
            
            continue

        guides.append(variant_guide.upper())
        fit18s.append(fitness)
        essentials.append(essential)

        number += 1
    

    return guides, fit18s, essentials 

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

    patience = 50
    
    print('params = ',params)

    # 存储交叉验证的pearson相关系数
    test_pearson_kfold = []

   # 循环遍历每个折叠
    for fold, (train_indices, val_indices) in enumerate(kf.split(features_array)):

        best_val_loss = float('inf')  # 初始最优验证损失
        no_improve_epochs = 0  # 验证损失未改进的回合数

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
        gen = predict_transformerv2.Predict_Staphylococcus_aureus(params=params).to(device)

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
                    print('输入的损失函数类型有误，请检查！！！')

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
            
            # 测试集开始测试
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

            # 检查验证损失是否改进
            if loss_test_one_epoch/len(test_loader) < best_val_loss:
                best_val_loss = loss_test_one_epoch/len(test_loader)

                no_improve_epochs = 0  # 重置未改进计数器
            else:
                no_improve_epochs += 1

            if epoch % 10 == 0:
                
                print(
                        f"Epoch[{epoch}/{params['train_epochs_num']}] ****Test loss: {loss_test_one_epoch/len(test_loader):.6f}********test correlation_coefficient:{correlation_coefficient}"
                        )
            
            metric.append(correlation_coefficient)

            # 保存loss和预测模型
            if correlation_coefficient > 0.85:
                
                # print("第一层权重:", gen.final_fc1.weight.data)
                # print("第一层偏置:", gen.final_fc1.bias.data)

                # np.save('result/loss_train_pierxun={0}.npy'.format(correlation_coefficient),np.array(loss_train))
                # np.save('result/loss_test_pierxun={0}.npy'.format(correlation_coefficient),np.array(loss_test))
                
                if loss_kind == 'pearson':
                    torch.save(gen,'../models/bio_pearson_predict_{0}_mertric={1}.pth'.format(epoch, correlation_coefficient))
                
                elif loss_kind == 'pearson_mse':
                    torch.save(gen,'../models/bio_pearson_mse_predict_{0}_mertric={1}.pth'.format(epoch, correlation_coefficient))
                
                elif loss_kind == 'mse':
                    torch.save(gen,'../models/bio_mse_predict_{0}_mertric={1}.pth'.format(epoch, correlation_coefficient))
                
                else:
                    print('损失函数类型出错，请检查！！！！')

            # 学习率衰减逻辑
            if no_improve_epochs > 0 and no_improve_epochs % 10:

                for param_group in opt_gen.param_groups:
                    param_group['lr'] = param_group['lr']*0.85


            # 提前停止逻辑
            if no_improve_epochs >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

                # pdb.set_trace()
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




if __name__ == '__main__':

    # 处理数据
    filename = '../data/Staphylococcus_aureus_essential.csv'
    guides, fit18s, essentials = read_data(filename=filename)

    csv_essentials = [True if item == "essential" else False for item in essentials]

    # df = pd.DataFrame({
    #     'guide_rna': guides,
    #     'fitness': fit18s,
    #     'essential': csv_essentials
    # })
    
    # # 将 DataFrame 写入 CSV 文件
    # df.to_csv('../data/Staphylococcus_sample.csv', index=False)
    # pdb.set_trace()

    features_array, labels_array, biofeatures_array = make_dataset_sequences_bio(guides, fit18s, essentials)

    # 定义K折交叉验证
    k_folds = 5
    kf = KFold(n_splits=k_folds, shuffle=True)

    # 确定训练的GPU编号
    params = {'device_num': 2, 'dropout_rate1': 0.3258494549467406, 'dropout_rate2': 0.2974783660130027, 'dropout_rate_fc': 0.3134874750986153, 'embedding_dim1': 64, 'embedding_dim2': 256, 'fc_hidden1': 109, 'fc_hidden2': 56, 'hidden_dim1': 1024, 'hidden_dim2': 256, 'l2_regularization': 5e-05, 'latent_dim1': 64, 'latent_dim2': 256, 'num_head1': 8, 'num_head2': 8, 'seq_len': 20, 'train_base_learning_rate': 0.0010350836441350173, 'train_batch_size': 512, 'train_epochs_num': 500, 'transformer_num_layers1': 4, 'transformer_num_layers2': 8}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(params['device_num'])
    print('device =',device)

    # 确定loss种类
    # loss_kind = ['pearson', 'pearson_mse', 'mse']
    loss_kind = 'pearson_mse'

    # 试运行一次主函数
    train(params, features_array=features_array, bios_array=biofeatures_array, labels_array=labels_array)

    '''开始搜索'''
    # 定义超参数的搜索空间
    space = {

        'train_batch_size':hp.choice('train_batch_size',[1024]),
        'seq_len':hp.choice('seq_len',[20]),  
        'device_num':hp.choice('device_num',[2]),
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

    # 创建Trials对象以跟踪优化过程
    trials = Trials()

    # 将训练函数包装为适用于hyperopt的目标函数
    objective = lambda params: train(params, features_array=features_array, bios_array=biofeatures_array, labels_array=labels_array)
    # 运行优化
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=1000, trials=trials)

    # 打印最佳参数
    print('最佳参数:', best)
