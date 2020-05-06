## Random Substitution Encoder
This repo contains a PyTorch implementation of Random Substitution Encoder (RSE) for the paper in <a href="https://arxiv.org/abs/2005.00446">arXiv</a>.


### Environment
Our experiments environment is as below
1. Ubuntu 18.0.4
2. Python 3.7.4
3. PyTorch 1.2.0
4. CUDA 10.0

### Download
You need to download IMDB, AGNEWS and YAHOO dataset and place them in ``./dataset/``.  
Then convert the origin dataset to standard data file used in this repo. The convert methods for each dataset are in ``tools.py``.
```
def read_IMDB_data(path):
    ###
def write_standard_data(datas, labels, path):
    ###

if __name__ == '__main__':
    origin_data_path = r'.../'
    path = r'./dataset/train_standard.txt'
    datas, labels = read_IMDB_data(origin_data_path)
    write_standard_data_to_file(datas, labels, path)
```

Download pretrained GloVe vectors (.6B.100d) and place it in ``./static/``

### Training
We prepared three main models (LSTM, BiLSTM, TextCNN), and their parameters could be edited in ``config.py``, ``network.py``, ``model_builder.py``.

Train your model like below (enhanced means using our method RSE)  
But you shall run ``python -u synonym.py --dataset IMDB`` to build synonyms tables first if you are using RSE. 
```
python -u train.py \
--dataset IMDB \
--model LSTM \
--enhanced yes \
--adv no \
--load_model no \
--epoch 100 \
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

The detailed attack results ``$time.csv`` and generated adversarial examples ``$time.txt`` are in ``static/DatasetName/foolresult/AttackerName/TargetModelName/``

### Evaluate results
The evaluation will show the target model's performance on origin test dataset, clean data and adversarial data.  
```
python -u ./evaluate.py \
--dataset IMDB \
--models LSTM_enhanced \
--adv_paths adv_data.txt \
--save_path ./evaluate_result.csv
```

``adv_data.txt`` is adversarial examples generated by the attacker.

### Benchmark
The result of our method RSE's performance on adversarial data or clean data.
| **Dataset**     | **Attack Model** | **LSTM** | **Bi-LSTM** | **Word-CNN** |
|-----------------|--------------|----------|-------------|--------------|
| IMDB            | No attack           | 87\.0    | 86\.5       | 87\.8        |
|                 | Random       | 83\.1    | 81\.9       | 83\.0        |
|                 | Textfool     | 84\.2    | 83\.7       | 83\.1        |
|                 | PWWS         | 82\.2    | 79\.3       | 81\.2        |
| AG’s News       | No attack           | 92\.9    | 94\.1       | 94\.8        |
|                 | Random       | 89\.2    | 92\.2       | 93\.1        |
|                 | Textfool     | 88\.7    | 90\.6       | 92\.2        |
|                 | PWWS         | 84\.2    | 88\.3       | 89\.9        |
| Yahoo\! Answers | No attack           | 72\.1    | 71\.8       | 70\.1        |
|                 | Random       | 68\.6    | 68\.9       | 67\.3        |
|                 | Textfool     | 67\.4    | 67\.1       | 66\.4        |
|                 | PWWS         | 64\.3    | 64\.6       | 62\.6        |

### Citation
```
@misc{wang2020defense,
    title={Defense of Word-level Adversarial Attacks via Random Substitution Encoding},
    author={Zhaoyang Wang and Hongtao Wang},
    year={2020},
    eprint={2005.00446},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```