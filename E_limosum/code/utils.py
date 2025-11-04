import scipy as sp
import pdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import csv
import os


def encode_essential(essential):
    # Allowed categories
    base_choice = ['NA', 'FALSE', 'TRUE']

    if essential in base_choice:
        # One-hot encoding
        if essential == 'NA':
            return np.array([1, 0, 0])
        if essential == 'FALSE':
            return np.array([0, 1, 0])
        if essential == 'TRUE':
            return np.array([0, 0, 1])
        # print("One-hot vector:", encoded_vector)
    else:
        print("Input string is not in the allowed set; one-hot encoding cannot be performed.")

def encode_ori(ori):
    base_choice = ['+', '-']

    # One-hot encoding
    if ori in base_choice:
        if ori == '+':
            return np.array([1, 0])
        if ori == '-':
            return np.array([0, 1])
    else:
        print("Input string is not in the allowed set; one-hot encoding cannot be performed.")

def encode_coding(coding):
    base_choice = ['NA', 'FALSE', 'TRUE']

    if coding in base_choice:
        # One-hot encoding
        if coding == 'NA':
            return np.array([1, 0, 0])
        if coding == 'FALSE':
            return np.array([0, 1, 0])
        if coding == 'TRUE':
            return np.array([0, 0, 1])
        # print("One-hot vector:", encoded_vector)
    else:
        print("Input string is not in the allowed set; one-hot encoding cannot be performed.")

def write_good_record(dict1, dict2, file_path):
    # Merge two dictionaries
    merged_dict = {**dict1, **dict2}

    # Create file if it does not exist; otherwise append
    if not os.path.isfile(file_path):
        with open(file_path, 'w') as file:
            file.write(f"{merged_dict}\n")
    else:
        with open(file_path, 'a') as file:
            file.write(f"{merged_dict}\n")

def one_hot(sequence):
    bases = ['A', 'T', 'G', 'C']
    # Initialize an array to store one-hot encodings
    one_hot_encoded = np.zeros((len(sequence), len(bases)))

    # Fill one-hot array
    for i, base in enumerate(sequence):
        one_hot_encoded[i, bases.index(base)] = 1

    return one_hot_encoded

def loss_pierxun(output, target):
    # print('target.shape = ', target.shape)
    # print('output.shape = ', output.shape)

    target_mean = torch.mean(target)
    outpu_mean = torch.mean(output)

    target_var = torch.std(target)
    output_var = torch.std(output)

    p = torch.mean((output - outpu_mean) * (target - target_mean))

    if output_var == 0:
        p /= ((output_var + 1e-7) * target_var)
        return p

    p /= (output_var * target_var)

    # print('Pearson correlation:', p)

    return p

def text_build_vocab():
    dic = [a for a in 'ATCG']
    dic += [a + b for a in 'ATCG' for b in 'ATCG']
    dic += [a + '0' for a in 'ATCG']
    return dic

def transformer_index_to_ATCGseq(data):
    # data = torch.randn(4, 100)  # Example tensor
    # Map indices to "A", "T", "C", "G"
    max_indices = torch.argmax(data, dim=0)
    max_indices = max_indices.to('cpu').numpy()
    # print('max_indices =', max_indices)
    mapping = {0: "A", 1: "T", 2: "C", 3: "G"}
    sequence = [mapping[i] for i in max_indices]
    # Join into a string
    sequence_str = ''.join(sequence)
    return sequence_str

def trans_output_to_input(fake_im):
    # Convert generated noise output to evaluator input:
    # first back to a raw sequence, then to a NumPy array
    sample_seq = []
    for num_sample in range(fake_im.shape[0]):
        sample_one = fake_im[num_sample, 0, :, :]
        sample_seq.append(transformer_index_to_ATCGseq(sample_one))
    # print('sample_seq = ', sample_seq)
    sample_result = []
    for seq in sample_seq:
        sample_result.append(Dimer_split_seqs(seq))

    sample_result = np.array(sample_result)
    sample_result = np.expand_dims(sample_result, axis=1)
    tensor = torch.from_numpy(sample_result)
    fake_img = tensor.to('cuda')
    return fake_img



def Dimer_split_seqs(seq):
    t = text_build_vocab()
    # print('t = ', t)
    # pdb.set_trace()
    ori_result = []
    dim_result = []
    pos_result = []
    
    result = ''

    lens = len(seq)

    for i in range(lens):
        result += ' ' + seq[i].upper()
        ori_result.append(t.index(seq[i].upper()))

    # dimer_encode
    # result += ' '
    # result += 'SEP1'

    seq += '0'
    wt = 2
    for i in range(lens):
        result += ' ' + seq[i:i + wt].upper()
        dim_result.append(t.index(seq[i:i + wt].upper()))
    
    # print('result = ',result)
    
    # pdb.set_trace()

    pos_result += [i for i in range(1, lens + 1)]
    # print('ori_result = ', ori_result)
    # print('dim_result = ', dim_result)
    # print('pos_result = ', pos_result)
    if ori_result[0] < 0:
        pdb.set_trace()
        print('seq = ', seq)
    
    seq_r = []
    seq_r.append(ori_result)
    seq_r.append(dim_result)
    seq_r.append(pos_result)
    # print('ori lenth = ',len(ori_result))
    # print('dim lenth = ',len(dim_result))
    # print('pos lenth = ',len(pos_result))
    # pdb.set_trace()
    # seq = pd.concat([nuc_seq, pos_seq], axis=0, ignore_index=True)

    return seq_r
def plot_test_prediction_result(output,label,epoch):
    val_pre = output.detach().cpu().numpy()
    val_pra = label.detach().cpu().numpy()
    
    plt.close()
    plt.figure()
    plt.plot(val_pre,label = 'val_pre')
    plt.plot(val_pra,label = 'val_pra')
    plt.legend()
    plt.title('prediction value and practice value')
    plt.savefig(f'result/epoch={epoch}')
    plt.show()
    
def compute_correlation_coefficient(output, label):
    target = output.detach().cpu().numpy().astype(float).ravel()
    prediction = label.detach().cpu().numpy().astype(float).ravel()

    mask = ~(np.isnan(prediction) | np.isnan(target))
    if not mask.all():
        print("NaN detected; corresponding entries will be ignored.")
        target = target[mask]
        prediction = prediction[mask]

    if target.size < 2 or prediction.size < 2:
        print("Not enough valid samples to compute correlation.")
        return 0.0, 0.0

    std_target = np.std(target)
    std_prediction = np.std(prediction)
    if std_prediction == 0:
        print("Predictions have no variance.")
    if std_target == 0:
        print("Ground truth has no variance.")

    if std_target == 0 or std_prediction == 0:
        pearson_coefficient = 0.0
    else:
        mean_target = np.mean(target)
        mean_prediction = np.mean(prediction)
        covariance = np.mean((target - mean_target) * (prediction - mean_prediction))
        pearson_coefficient = covariance / (std_target * std_prediction)

    res = sp.stats.spearmanr(target, prediction, nan_policy='omit')
    spearman_coefficient = 0.0 if np.isnan(res.correlation) else float(res.correlation)

    return float(pearson_coefficient), float(spearman_coefficient)

