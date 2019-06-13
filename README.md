Use cases of the project:
1. If you want to reuse ready-to-go model of BERT for NER in TF Serving stack
2. If you want to train your own model from tfrecord
3. If you have your own BIO dataset

# Reuse BERT NER model for Predictions
## Reuse with TF Serving
0. under your virtual environment run `pip install -r requirements.txt`
1. First you need to download model
2. instal tensorflow_serving
3. From project folder launch docker image: 

`sudo docker run -t --rm -p 8501:8501 -v "${PWD}/res:/models" -e MODEL_NAME='BERT_NER_ESTIMATOR' -e MODEL_PATH='/models/BERT_NER_ESTIMATOR' --name='BERT_NER_ESTIMATOR'  tensorflow/serving`

4. Now you can poll Estimator with curl:

`curl -d '{"instances": [{"input_ids": [212, 14, 513, 3,11], "input_masks": [1,1,1,1,1], "y_masks": [1,1,1,1,1]}]}' -X POST http://localhost:8501/v1/models/BERT_NER_ESTIMATOR:predict`

Output:
```
{
    "predictions": [["B-PER", "O", "O", "B-PER", "O"]
    ]
}
```

## Reuse in testing mode 
If you would like to test component predictions with easy to debug python scripts:

`python ner_predict.py`

# Prepare custom dataset for training by BERT NER Estimator

TF Estimator is a core of the BERT Ner component, unfortunately BERT reuires specific preprocessing implemented 
in `bert_ner_preprocessor.py`. So if you have data in CONLL-2003 NER format (suppose it is placed in 
file <BIO_DATASET_PATH>) you need to cnvert it to tensorflow friendly tfrecord format.  

## convert dataset from BIO-markup into TF records dataset

General usage:
`cat <BIO_DATASET_PATH> | ./bio2tf.py <TFRECORD_DATASET_PATH>`

### Examples:

`cat data/train.txt | ./bio2tf.py data/train.tfrecord`

`cat data/valid.txt | ./bio2tf.py data/valid.tfrecord`


# UpTrain existing model
If you want to fit the component for your data (supposing you have the same set of entites to be predicted by NER) 
you can convert your dataset into tfrecord format.  And when you have tfrecord dataset 
(placed in <TFRECORD_DATASET_PATH>) you can launch training process by following command:

`python ner_train.py --train_dataset data/train.tfrecord --model_save_path res/BERT_NER_ESTIMATOR --training_steps 2`

`python ner_train.py --batch_size 29 --model_save_path res/BERT_NER_ESTIMATOR --train_dataset data/train_lowercased.tfrecord --training_steps 2`
## 

When it complete you can launch TF server with command 3 from paragraph "Reuse BERT NER model":

`sudo docker run -t --rm -p 8501:8501 -v "${PWD}/res:/models" -e MODEL_NAME='BERT_NER_ESTIMATOR' -e MODEL_PATH='/models/BERT_NER_ESTIMATOR' --name='BERT_NER_ESTIMATOR'  tensorflow/serving`

##Training:

python ner_train.py

## Evaluation

`python ner_evaluate.py --dataset data/valid.tfrecord` 

## Data Manipulation
if you want to lowercase prepare lowercased dataset from BIO markup you can use script `dataset_lowercaser.py`.

Example:

`python dataset_lowercaser.py --cased_dataset data/train.txt --target_path data/train_lowercased.txt`

If you want undeterministic lowercasing you can run with option `lowercasing_probability`:

`python dataset_lowercaser.py --cased_dataset data/train.txt --target_path data/train_lowercased.txt --lowercasing_probability=0.8`


# Useful Links:
https://guillaumegenthial.github.io/serving-tensorflow-estimator.html
https://medium.com/@yuu.ishikawa/serving-pre-modeled-and-custom-tensorflow-estimator-with-tensorflow-serving-12833b4be421  

# Docker & DevOps Cheatsheet
 sudo docker ps
 sudo docker stop BERT_NER_ESTIMATOR
 curl -d '{"instances": [{"input_ids": [212, 14, 513, 3,11], "input_masks": [1,1,1,1,1], "y_masks": [1,1,1,1,1]}]}' -X POST http://localhost:8501/v1/models/BERT_NER_ESTIMATOR:predict
 sudo docker run -t --rm -p 8501:8501 -v "${PWD}/resources/models_for_serving:/models" -e MODEL_NAME='BERT_NER_ESTIMATOR' -e MODEL_PATH='/models/BERT_NER_ESTIMATOR' --name='BERT_NER_ESTIMATOR'  tensorflow/serving
 tensorboard --logdir=/home/alx/Workspace/dp_bert_ner/