import pandas as pd
import numpy as np

def add_activity_column(csv_path,csv_out_path):
    """
    清洗并处理 gRNA 实验数据：
    - 排除 gene 为 NaN 和 coding == False 的记录；
    - 仅保留观测数 > 4 的基因；
    - 构造 activity 列；
    - 保存更新回原 CSV 文件。
    """

    # 读取原始数据
    df = pd.read_csv(csv_path)

    # 1. 删除 gene 为 NaN 或空字符串的行
    df = df.dropna(subset=['gene'])
    df = df[df['gene'].astype(str).str.strip() != ""]

    # 2. 删除 coding 为 False 的行
    df.drop(df[df['coding'] == False].index, inplace=True)
    
    # 统计每个基因的 gRNA 数量
    gene_counts = df['gene'].value_counts()

    # 选出观测数 > 4 的有效基因
    valid_genes = gene_counts[gene_counts > 4].index

    # 筛选这些有效基因对应的数据
    df_clean = df[df['gene'].isin(valid_genes)].copy()

    # 计算 gene_median（按基因对 fit75 求中位数）
    gene_medians = df_clean.groupby('gene')['fit75'].median()
    df_clean['gene_median'] = df_clean['gene'].map(gene_medians)

    # 构造 activity 列
    df_clean['activity'] = df_clean['gene_median'] - df_clean['fit75']

    df_clean.to_csv(csv_out_path, index=False)

add_activity_column(csv_path='../data/screen_data.csv',csv_out_path='../data/guide effiency.csv')