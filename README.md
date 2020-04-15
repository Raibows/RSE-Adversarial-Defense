## Random Substitution Encoder for Defending Adversarial Examples in NLP
This repo contains an implementation of RSE.

### Environment
Our experiments environment is as below
1. Ubuntu 18.0.4
2. Python 3.7.4
3. PyTorch 1.2.0
4. Python packages installed could be seen in ``requirements.txt``

### Download
You need to download IMDB, AGNEWS and YAHOO dataset and place them in ``./dataset/``.  
And convert the origin dataset to standard data file used in this repo. The methods for each dataset are in ``tools.py``.
```
def read_IMDB_data(path):
    ###
def write_standard_data_to_file(datas, labels, path):
    ###

if __name__ == '__main__':
    origin_data_path = r'.../'
    path = r'./dataset/train_standard.txt'
    datas, labels = read_IMDB_data(origin_data_path)
    write_standard_data_to_file(datas, labels, path)
```

Download pretrained GloVe vectors (.6B.100d) and place it in ``./static/``

### Train target model
We prepared three main models (LSTM, BiLSTM, TextCNN), and their parameters could be edited in ``config.py``, ``network.py``, ``model_builder.py``.

Train your model like below(enhanced means using NoiseLayer, adv means using adversarial training (AT))  
If you are using NoiseLayer, you shall run ``python -u synonym.py --dataset IMDB`` to build synonyms tables first.
```
python -u train.py \
--dataset IMDB \
--model LSTM \
--enhanced yes \
--adv no \
--load_model no \
--save_acc_limit 0.89 \
--epoch 60 \
--batch 64 \
--lr 3e-3 \
--verbose no
```

The best model will be saved in ``./models/IMDB/LSTM_enhanced_acc_time.pt``. And remember to correct the model load path in ``config.py``.
```
config_model_load_path = {
    'IMDB': {
        'LSTM_enhanced': 'LSTM_enhanced_acc_time.pt',
    },
}
```

### Prepare clean data
Sample 1k data from origin test dataset for attackers to generate adversarial examples.
```
python tools.py \
--dataset IMDB \
--num 1000
```

### Try attack
Attackers supported is RANDOM, TEXTFOOL, PWWS.  
Make sure the ``config_dataset`` in ``config.py`` is as same as the ``dataset`` in below commands.
```
python -u fool.py \
--dataset IMDB \
--attack PWWS \
--model LSTM_enhanced \
--verbose no
```

The detailed attack results ``time.csv`` and generated adversarial examples ``time.txt`` are in ``static/DatasetName/foolresult/AttackerName/TargetModelName/``

### Evaluate results
The evaluation will show the target model's performance on origin task (origin test dataset), clean data and adversarial data.  
```
python -u ./evaluate.py \
--dataset IMDB \
--models LSTM_enhanced \
--adv_paths adv_data.txt \
--save_path ./evaluate_result.csv
```

``adv_data.txt`` is adversarial examples generated by the attacker.


### Code Reference
textfool (Zenodo/Github).  
PWWS (JHL/Github)