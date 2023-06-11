from typing import Dict, List

import numpy as np
import os
import pandas as pd
import argparse
import torch
import math

from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torch.nn.functional as F
import sklearn
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import r2_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from torch.utils.data import Dataset, DataLoader

SEED = 227


class DataPreparation(Dataset):
    
    def __init__(self, tokenizer, data, target_values='circumplex', max_length=None):
        
        self.tokenizer = tokenizer
        self.data = data
        self.t_val = target_values

        if max_length == None:
            max_length_counted = data["text"].str.split(' ').str.len().max(axis=0)
            self.max_length = max_length_counted if max_length_counted < 512 else 512
        else:
            self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def tokenize(self, text):

        tokens = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt')

        return tokens
    

    def reorganize_labels(self, row):

        def return_zeros(row_data, classes):
          # print(row_data[classes])
          labels_zeros = np.zeros(len(set(self.data[classes].unique())))
          # print(labels_zeros)
          labels_zeros[row_data[classes]] = 1
          # print(labels_zeros)
          return labels_zeros

        return np.concatenate((return_zeros(row, 'raven_classes'), return_zeros(row, 'verb_classes')))

    def __getitem__(self, index):
        
        source_text = self.data['text'].iloc[index]
        source = self.tokenize(source_text)

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        
        labels = self.reorganize_labels(self.data.iloc[index])        

        return {
            "source_ids": source_ids.to(dtype=torch.long),
            "source_mask": source_mask.to(dtype=torch.long),
            "labels":  labels
        }


class BertBaseline_ResNet(nn.Module):
    
    def __init__(self, model_name, inner_feautes, out_features):
        super(BertBaseline_ResNet, self).__init__()

        self.bert = AutoModel.from_pretrained(model_name, return_dict=True)

        self.linear_modules = nn.ModuleList([torch.nn.Linear(self.bert.config.hidden_size, inner_feautes),
                                          torch.nn.Linear(inner_feautes, inner_feautes),
                                          torch.nn.Linear(inner_feautes, out_features)])
        
        self.dropout = nn.Dropout(0.4)
        self.layer_norm = nn.LayerNorm(inner_feautes)
        self.relu = nn.ReLU()   
    
    def forward(self, input_ids, attention_mask):

         bert = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True
            )
         x = bert[0][:, 0]  # last hidden state output

         for lin in self.linear_modules[:-1]:
            x = lin(x)
            # x = self.layer_norm(x)
            x = self.relu(x)
            x = self.dropout(x)
        
        #  x = self.relu(x)
         x = self.linear_modules[-1](x)
        
         return x


def get_intelligence_labels(dataset):
    dataset['raven_classes'] = dataset['raven_classes'].astype(int)
    RAVEN_LABELS = list(set(dataset['raven_classes'].unique()))
    RAVEN_SCORE_INDICES = range(0, 5)
    VERB_LABELS = list(set(dataset['verb_classes'].unique()))
    VERB_SCORE_INDICES = range(5, 10)

    INT_LABELS = RAVEN_LABELS + VERB_LABELS
    INDEXEX_CIRCLE = [RAVEN_SCORE_INDICES, VERB_SCORE_INDICES]
    return INDEXEX_CIRCLE, INT_LABELS


def initialize_scaling(data_org, intell):
    scale = StandardScaler().fit(np.array(data_org[intell]).reshape(-1, 1))
    return scale

def inverse_toorig(scaler, list_of_labels):
    inverse = scaler.inverse_transform(list_of_labels.reshape(-1, 1))
    return inverse


def analysing_logits(logits, class_indexes):
  
  def compute_maxes(logits, CUR_SCORE_INDEX):
    labels_zeros = np.zeros(logits[:, CUR_SCORE_INDEX].shape)
    pred_class = np.argmax(logits[:, CUR_SCORE_INDEX], axis=1)
    labels_zeros[list(range(len(labels_zeros))), pred_class] = 1
    return labels_zeros

  list_of_best_labels = []

  for circ_cl in class_indexes:
    labels_zeros = compute_maxes(logits, circ_cl)

    list_of_best_labels.append(labels_zeros)
  
  return np.concatenate((list_of_best_labels), axis=1)


def init_adding_weights():
    return (1/1.3, 1/1.3)


def comupte_custom_loss(labels, logits, criterion, class_indexes):
    
    custom_weights = init_adding_weights()

    custom_loss = 0
    for ind, circ_cl in enumerate(class_indexes):
      custom_loss += custom_weights[ind] * criterion(logits[:, circ_cl],
                                  labels[:, circ_cl])

    return custom_loss


def show_metrics(labels, predicted, class_indexes):

    labels_names = ['raven', 'verb']
    
    labels = np.vstack(labels)
    predicted = np.vstack(predicted)

    for lab, circ_cl in zip(labels_names, class_indexes):
      
      f1_sc = f1_score(labels[:, circ_cl],
               predicted[:, circ_cl], average="macro")
      accur = accuracy_score(labels[:, circ_cl],
               predicted[:, circ_cl])
      
      
      print(f'Scores for {lab}\n==================\n')
      print('Macro F1 score for {0} - {1}\n'.format(lab, f1_sc))
      print('Accuracy score for {0} - {1}\n'.format(lab, accur))
      print(classification_report(labels[:, circ_cl], predicted[:, circ_cl], zero_division=0))


def train(model, data_loader, device, optimizer, criterion, n_epoch, class_indexes):

    print('Epoch #{}\n'.format(n_epoch+1))

    train_losses = []
    train_labels = []
    train_predictions = []   

    progress_bar = tqdm(total=int(len(data_loader.dataset)/data_loader.batch_size), 
                        desc='Epoch {}'.format(n_epoch + 1))

    model.train()

    for _, data in enumerate(data_loader, 0):


          input_ids = data["source_ids"].to(device)
          attention_mask = data["source_mask"].to(device)
          labels = data['labels'].to(device)

          optimizer.zero_grad()

          pred = model(input_ids=input_ids, attention_mask=attention_mask)
          loss = comupte_custom_loss(labels, pred, criterion, class_indexes)
              
          loss.backward()
              
          optimizer.step()

          resulting_output = analysing_logits(pred.detach().cpu().numpy(), class_indexes)

          train_losses.append(loss.item())
          train_labels.extend(labels.cpu().detach().numpy())
          train_predictions.extend(resulting_output)
          progress_bar.set_postfix(loss=np.mean(train_losses))
          progress_bar.update(1)

    progress_bar.close()
  
    
    print('\n\nMean Loss after epoch #{0} - {1}'.format(str(n_epoch + 1), np.mean(train_losses)))
    print(f'\n Scores after {n_epoch + 1} on train:')
    show_metrics(train_labels, train_predictions, class_indexes)


def validating(model, data_loader, device, optimizer, criterion, n_epoch, class_indexes):

    val_losses, val_labels, val_predictions = [], [], []

    progress_bar = tqdm(total=int(len(data_loader.dataset)/data_loader.batch_size),
                        desc='Epoch {}'.format(n_epoch + 1))

    model.eval()

    for _, data in enumerate(data_loader, 0):
          
          input_ids = data["source_ids"].to(device)
          attention_mask = data["source_mask"].to(device)
          labels = data['labels'].to(device)

          with torch.no_grad():
              pred = model(input_ids, attention_mask)

          loss = comupte_custom_loss(labels, pred, criterion, class_indexes)
          
          resulting_output = analysing_logits(pred.detach().cpu().numpy(), class_indexes)

          val_losses.append(loss.item())
          val_labels.extend(labels.cpu().detach().numpy())
          val_predictions.extend(resulting_output)

          progress_bar.set_postfix(loss=np.mean(val_losses))
          progress_bar.update(1)

    progress_bar.close()
    
    
    valid_stats.append(
        {
            'Val Loss': np.mean(val_losses)
        }
    )

    print('\n\nMean Loss after epoch #{0} - {1}'.format(str(n_epoch + 1), np.mean(val_losses)))
    print(f'\n Scores after {n_epoch + 1} on validation:')
    show_metrics(val_labels, val_predictions, class_indexes)

    return valid_stats


def evaluate(model, train_dataset, val_dataset, device, epochs, target_value, curc_indexes):
    
    model = model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss().to(device)

    global valid_stats
    valid_stats = []
    best_valid_loss = float('inf')

    for epoch in range(epochs):
        # train
        try:
            train(model, train_dataset, device, optimizer, criterion, epoch, curc_indexes)
          # # validate
            valid_stats = validating(model, val_dataset, device, optimizer, criterion, epoch, curc_indexes)

            if valid_stats[epoch]['Val Loss'] < best_valid_loss:
                best_valid_loss = valid_stats[epoch]['Val Loss']

                name_to_save = f'model_baseline_basic_{target_value}'
                if os.path.isfile('results/'+name_to_save+'.pth'):
                    os.remove('results/'+name_to_save+'.pth')
                    torch.save(model.state_dict(), 'results/'+name_to_save+'.pth')
                else:
                    torch.save(model.state_dict(), 'results/'+name_to_save+'.pth')
        except KeyboardInterrupt:
            break




if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_data", default='dataset_all_nlp_features_target_classes.csv')
    # parser.add_argument("--task", default="Intelligence")
    # parser.add_argument("--target_value", default="verb")
    # parser.add_argument("--type_task", default="classification")
    parser.add_argument("--path_to_model", default="DeepPavlov/rubert-base-cased-sentence")
    # parser.add_argument("--model_arch", default="baseline")
    parser.add_argument("--epochs", default=15)

    args = parser.parse_args()

    path_to_data = args.path_to_data
    # target_value = args.target_value
    path_to_model = args.path_to_model
    epochs = args.epochs

    dataset = pd.read_csv(path_to_data, sep='\t')

    dataset = dataset[dataset.question_id != '129_Чтение текста - видео']

    dataset = dataset[dataset["verb"] > 0]
    dataset["verb"] = dataset["verb"].astype(int)
    dataset = dataset[dataset["raven"] > 0]
    dataset["raven"] = dataset["raven"].astype(int)
        
    tokenizer = AutoTokenizer.from_pretrained(path_to_model)

    # intelligence = target_value+'_classes'

    # dataset[intelligence] = dataset[intelligence].astype(int)

    curc_indexes, curc_labels = get_intelligence_labels(dataset)

    train_data, vaild_data, test_data = np.split(dataset.sample(frac=1, random_state=42), 
                                        [int(.80*len(dataset)), int(.91*len(dataset))])
        
    # scaler = initialize_scaling(train_data, target_value)

    train_dataset_data = DataPreparation(
            tokenizer=tokenizer,
            data = train_data,
            #intelligence = intelligence,
            max_length = 60
        )

    val_dataset_data = DataPreparation(
            tokenizer=tokenizer,
            data = vaild_data,
            #intelligence = intelligence,
            max_length = 60
        )

    test_dataset_data = DataPreparation(
            tokenizer=tokenizer,
            data = test_data,
            #intelligence = intelligence,
            max_length = 60
        )

    train_dataset = DataLoader(train_dataset_data, batch_size=16, drop_last=True, shuffle=True)
    val_dataset = DataLoader(val_dataset_data, batch_size=16)
    test_dataset = DataLoader(test_dataset_data, batch_size=16)
        
    # print('___', len(dataset[intelligence].unique()))
    model = BertBaseline_ResNet(pre_trained=path_to_model, out_features=len(curc_labels))
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    evaluate(model=model, train_dataset=train_dataset, val_dataset=val_dataset, device=device, epochs=epochs, target_value=target_value, curc_indexes=curc_indexes)