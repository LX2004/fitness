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
# To train a model for predicting E. coli fitness values using a dataset, you may perform the following steps.
run : 
```
cd train_prediction_model

python make_dataset.py

python prediction_transformer_dimer_original_kfold.py

python test_model_performance.py
```
