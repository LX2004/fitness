import pandas as pd
import numpy as np

def add_activity_column(csv_path, csv_out_path):
    """
    Clean and process gRNA experimental data:
    - Exclude rows with NaN/empty 'gene';
    - Drop records where 'coding' == False;
    - Keep only genes with > 4 observations;
    - Construct an 'activity' column;
    - Save the cleaned data to the output CSV.
    """

    # Read the raw data
    df = pd.read_csv(csv_path)

    # 1) Remove rows where 'gene' is NaN or empty
    df = df.dropna(subset=['gene'])
    df = df[df['gene'].astype(str).str.strip() != ""]

    # 2) Remove rows where 'coding' is False
    df.drop(df[df['coding'] == False].index, inplace=True)
    
    # Count the number of gRNAs per gene
    gene_counts = df['gene'].value_counts()

    # Select genes with > 4 observations
    valid_genes = gene_counts[gene_counts > 4].index

    # Filter to rows belonging to valid genes
    df_clean = df[df['gene'].isin(valid_genes)].copy()

    # Compute gene-level medians of 'fit75'
    gene_medians = df_clean.groupby('gene')['fit75'].median()
    df_clean['gene_median'] = df_clean['gene'].map(gene_medians)

    # Construct the 'activity' column
    df_clean['activity'] = df_clean['gene_median'] - df_clean['fit75']

    # Save to CSV
    df_clean.to_csv(csv_out_path, index=False)

add_activity_column(csv_path='../data/screen_data.csv',
                    csv_out_path='../data/guide effiency.csv')
