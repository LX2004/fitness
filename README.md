# fitness prediction
This project primarily accomplishes the task of predicting cellular fitness values based on the Transformer model.
# Preparation
run:

```
cd fitness

conda create --name fitness

conda activate fitness

conda install --file request.txt
```

# To train a model for predicting E. coli fitness values using a dataset, you may perform the following steps.
run :
```
cd E_coli

python prediction_transformer_ori_dim_bio_kfold.py

```
# To train a model for predicting Cyanobacteria fitness values using a dataset, you may perform the following steps.
run : 
```
cd Cyanobacteria

python prediction_transformer_ori_dim_bio_kfold.py

```
# Result.
The model obtained from training is stored in the "models" folder, and details such as training parameters and metrics are recorded in the corresponding text files.
