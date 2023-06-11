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
    
    def __init__(self, tokenizer, data, scale_init, intelligence='verb', max_length=None, if_scale=True):
        
        self.tokenizer = tokenizer
        self.data = data
        self.intell = intelligence
        self.scale = scale_init
        self.if_scale = if_scale
        
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


    def scaling(self, labels):
      
        scaled_target = self.scale.transform(np.array(labels).reshape(-1, 1))
        
        return scaled_target

     
    def __getitem__(self, index):
        
        source_text = self.data['text'].iloc[index]
        source = self.tokenize(source_text)

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        
        if self.if_scale:
            scaled_labels = self.scaling(self.data[self.intell])
            label = scaled_labels[index][0]
        else:
            label = self.data[self.intell].iloc[index]

        return {
            "source_ids": source_ids.to(dtype=torch.long),
            "source_mask": source_mask.to(dtype=torch.long),
            "labels":  label
        }


class BertBaseline_ResNet(nn.Module):
    
    def __init__(self, model_name, out_features, inner_feautes=256):
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
#          x = bert[0][:, 0]  # last hidden state output
         token_embeddings = bert[0]
         input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
         sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
         sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
         x = sum_embeddings / sum_mask
        
         h = []
        
         for lin in self.linear_modules[:-1]:
            x = lin(x)
            h.append(x)
            x = self.layer_norm(x)
            x = self.relu(x)
            x = self.dropout(x)
        
#          x = self.relu(x)
         x = self.relu(h[-1] + h[-2])
         x = self.linear_modules[-1](x)
        
         return x
    

def initialize_scaling(data_org, intell):
    scale = StandardScaler().fit(np.array(data_org[intell]).reshape(-1, 1))
    return scale

def inverse_toorig(scaler, list_of_labels):
    inverse = scaler.inverse_transform(list_of_labels.reshape(-1, 1))
    return inverse


def train(model, data_loader, device, optimizer, criterion, n_epoch):

    print('Epoch #{}\n'.format(n_epoch+1))

    train_losses = []
    train_labels = []
    train_predictions = []   

    progress_bar = tqdm(total=math.ceil(len(data_loader.dataset)/data_loader.batch_size), 
                        desc='Epoch {}'.format(n_epoch + 1))

    model.train()

    for _, data in enumerate(data_loader, 0):


          input_ids = data["source_ids"].to(device)
          attention_mask = data["source_mask"].to(device)
          labels = data['labels'].to(device)
        #   nlp_feat = data['nlp_feat'].to(device)
          # print(nlp_feat)

          optimizer.zero_grad()

          pred = model(input_ids=input_ids, attention_mask=attention_mask)
          loss = criterion(pred, labels)
              
          loss.backward()
              
          optimizer.step()

          _, predict = torch.max(pred.cpu().data, 1)
          train_losses.append(loss.item())
          train_labels.extend(labels.cpu().detach().numpy())
          train_predictions.extend(predict.cpu().detach().numpy())

          progress_bar.set_postfix(loss=np.mean(train_losses))
          progress_bar.update(1)
    
    progress_bar.update(1)
    progress_bar.close()
  
    
    print('\n\nMean Loss after epoch #{0} - {1}'.format(str(n_epoch + 1), np.mean(train_losses)))
    print('F1 score after epoch #{0} on train - {1}\n'.format(str(n_epoch + 1), f1_score(train_labels, train_predictions, average='macro')))
    print('Accuracy score after epoch #{0} on train - {1}\n'.format(str(n_epoch + 1), accuracy_score(train_labels, train_predictions)))

    print(classification_report(train_labels, train_predictions))
    
    return train_labels, train_predictions


def validating(model, data_loader, device, criterion, n_epoch):

    val_losses, val_labels, val_predictions = [], [], []

    progress_bar = tqdm(total=math.ceil(len(data_loader.dataset)/data_loader.batch_size),
                        desc='Epoch {}'.format(n_epoch + 1))

    model.eval()

    for _, data in enumerate(data_loader, 0):
          
          input_ids = data["source_ids"].to(device)
          attention_mask = data["source_mask"].to(device)
          labels = data['labels'].to(device)
        #   nlp_feat = data['nlp_feat'].to(device)

          with torch.no_grad():
              pred = model(input_ids, attention_mask)

          loss = criterion(pred, labels)
          
          _, predict = torch.max(pred.cpu().data, 1)

          val_losses.append(loss.item())
          val_labels.extend(labels.cpu().detach().numpy())
          val_predictions.extend(predict.cpu().detach().numpy())

          progress_bar.set_postfix(loss=np.mean(val_losses))
          progress_bar.update(1)

    progress_bar.update(1)
    progress_bar.close()
    
    
    valid_stats.append(
        {
            'Val Loss': np.mean(val_losses)
        }
    )

    print('\n\nMean Loss after epoch #{0} - {1}'.format(str(n_epoch + 1), np.mean(val_losses)))
    print('F1 score after epoch #{0} on validation - {1}\n'.format(str(n_epoch + 1), f1_score(val_labels, val_predictions, average='macro')))
    print('Accuracy score after epoch #{0} on validation - {1}\n'.format(str(n_epoch + 1), accuracy_score(val_labels, val_predictions)))
    
    print(classification_report(val_labels, val_predictions))
    return valid_stats


def evaluate(model, train_dataset, val_dataset, device, epochs, target_value):
    
    model = model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss().to(device)

    global valid_stats
    valid_stats = []
    best_valid_loss = float('inf')

    for epoch in range(epochs):
        # train
        try:
            train(model, train_dataset, device, optimizer, criterion,  epoch)
            # # validate
            validating(model, val_dataset, device, criterion, epoch)

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
    parser.add_argument("--target_value", default="verb")
    # parser.add_argument("--type_task", default="classification")
    parser.add_argument("--path_to_model", default="DeepPavlov/rubert-base-cased-sentence")
    # parser.add_argument("--model_arch", default="baseline")
    parser.add_argument("--epochs", default=15)

    args = parser.parse_args()

    path_to_data = args.path_to_data
    target_value = args.target_value
    path_to_model = args.path_to_model
    epochs = args.epochs

    dataset = pd.read_csv(path_to_data, sep='\t')

    dataset = dataset[dataset.question_id != '129_Чтение текста - видео']

    if target_value == 'raven':
            dataset = dataset[dataset["raven"] > 0]
    if target_value == 'verb':
            dataset = dataset[dataset["verb"] > 0]
        
    tokenizer = AutoTokenizer.from_pretrained(path_to_model)

    intelligence = target_value+'_classes'

    dataset[intelligence] = dataset[intelligence].astype(int)

    train_data, extra_data = train_test_split(dataset, test_size=0.22,
                                        stratify=dataset[intelligence],
                                        random_state=SEED)

    vaild_data, test_data = train_test_split(extra_data, test_size=0.45,
                                        stratify=extra_data[intelligence],
                                        random_state=SEED)
        
    scaler = initialize_scaling(train_data, target_value)

    train_dataset_data = DataPreparation(
            tokenizer=tokenizer,
            data = train_data,
            scale_init = scaler,
            intelligence = intelligence,
            max_length = 60,
            if_scale = False
        )

    val_dataset_data = DataPreparation(
            tokenizer=tokenizer,
            data = vaild_data,
            scale_init = scaler,
            intelligence = intelligence,
            max_length = 60,
            if_scale = False
        )

    test_dataset_data = DataPreparation(
            tokenizer=tokenizer,
            data = test_data,
            scale_init = scaler,
            intelligence = intelligence,
            max_length = 60,
            if_scale = False
        )

    train_dataset = DataLoader(train_dataset_data, batch_size=16, drop_last=True, shuffle=True)
    val_dataset = DataLoader(val_dataset_data, batch_size=16)
    test_dataset = DataLoader(test_dataset_data, batch_size=16)
        
    # print('___', len(dataset[intelligence].unique()))
    model = BertBaseline_ResNet(pre_trained=path_to_model, out_features=len(dataset[intelligence].unique()))
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    evaluate(model=model, train_dataset=train_dataset, val_dataset=val_dataset, device=device, epochs=epochs, target_value=target_value)