## Extract Triples from text

#### Step1: 
Predownload chinese bert model and extract it to pretrained_bert_model.
```
wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz
wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-vocab.txt
tar xvf bert-base-chinese.tar.gz && rm bert-base-chinese.tar.gz
mv pytorch_model.bin pretrained_bert_model/
mv bert_config.json pretrained_bert_model/config.json
mv bert-base-chinese-vocab.txt pretrained_bert_model/vocab.txt
```

#### Step2:
Preprocess raw data into Pytorch Dataset.
```
python run.py --do_preprocess
```

#### Step3:
Begin training our model. Note: log file is in log and saved model is in save.
```
python run.py --do_train
```

#### Step4:
Predict model and parse the output of model to csv file.
```
python run.py --do_predict
```
