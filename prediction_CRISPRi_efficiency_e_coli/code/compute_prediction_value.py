from random import choice
import numpy as np
from utils import *
from net import predict_transformerv2
from IPython import display

def read_data(filename):

    import math

    guides = []
    fit18s = []
    genes = []

    # 生物信息
    essentials = []
    df = pd.read_csv(filename)

    number = 0
    for variant_guide, essential, fitness, gene in zip(df['variant_guide'], df['essential'], df['fitness'], df['gene']):
        
        # 转化为 float 类型
        fitness = float(fitness)

        if math.isnan(fitness):
            print(f'fitness is {fitness}!!!')
            
            continue

        guides.append(variant_guide.upper())
        fit18s.append(fitness)

        essentials.append(essential)
        genes.append(gene)

        number += 1

    return guides, fit18s, essentials, genes


def make_dataset_sequences_bio(guides, fit18s, essentials):

    features_array = []
    bios_array = []
    labels_array = []

    fit18s = np.array(fit18s)

    max_reads = np.max(fit18s) 
    min_reads = np.min(fit18s)

    print('max_reads = ',max_reads)
    print('min_reads = ',min_reads)
    # pdb.set_trace()

    number = 0
 

    for sequence, score, essential in zip(guides, fit18s, essentials):

        if len(sequence) < 20:

            print('length = ', len(sequence))
            print('sequence = ',sequence)
            continue
        
        # 进行独热向量编码
        essential = str(essential)

        if essential == 'True':
        
            ori = np.array([1,0])
        
        elif essential == 'False':
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


def compute_scaler(model_output):

    model_output = model_output.detach().cpu().numpy()

    max_reads =  1.157
    min_reads =  -0.234

    model_output = min_reads + (max_reads - min_reads) * model_output

    return model_output


if __name__=='__main__':
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(4)
    print('device =',device)

    # 从数据库中加载
    filename = '../data/Bacillus_subtilis.csv'
    guides, fit18s, essentials, genes= read_data(filename=filename)
    features_array, labels_array, biofeatures_array = make_dataset_sequences_bio(guides, fit18s, essentials)

    # 加载模型
    # model = torch.load('good_modles/predict_497_mertric=0.6346185573845174.pth')
    model = torch.load('../models/remove_bio_pearson_mse_predict_57_mertric=0.618637.pth').to(device)
    
    # 打印模型的权重参数
    for name, param in model.named_parameters():

        print(f"Parameter name: {name}, Shape: {param.shape}, Parameter: {param}")
        break

    prediction_remove_bio = compute_scaler(model(torch.tensor(features_array).to(device)))

    model = torch.load('../models/bio_pearson_mse_predict_86_mertric=0.7014662700653662.pth').to(device)
    prediction = compute_scaler(model(torch.tensor(features_array).to(device), torch.tensor(biofeatures_array).to(device)))

    # 将它们存储到一个字典中，作为 pandas DataFrame 的数据
    data = {
        'gene':genes,
        'guides': guides,
        'fit18s': fit18s,
        'essentials':essentials,

        'prediction_with_bio': np.squeeze(prediction),
        'prediction_remove_bio': np.squeeze(prediction_remove_bio)
    }

    print(prediction_remove_bio.shape)
    print(prediction.shape)

    print(prediction_remove_bio[0])
    print(prediction[0])

    # 将数据转换为 DataFrame
    df = pd.DataFrame(data)

    # 将 DataFrame 保存为 CSV 文件
    file_path = '../result/prediction_and_actual_value_bacillus.csv'
    df.to_csv(file_path, index=False)

