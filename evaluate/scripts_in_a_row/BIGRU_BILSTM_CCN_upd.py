from typing import Dict, List
from transformers import AutoTokenizer, AutoModel
import numpy as np
import pandas as pd
import torch
import random
import math
from sklearn.metrics import f1_score, accuracy_score, classification_report
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torch.nn.functional as F
import sklearn
import sys
import os
import argparse
import logging
from sklearn.utils import class_weight
from torch.cuda.amp import autocast, GradScaler
from transformers import BertModel
from torch.cuda.amp import autocast, GradScaler
from transformers import get_linear_schedule_with_warmup, AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

def seed_everything(seed_value=42):
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    return seed_value


class DataPreparation(Dataset):
    
    def __init__(self, tokenizer, data, intelligence='verb', max_length=None):
        
        self.tokenizer = tokenizer
        self.data = data
        self.intell = intelligence
        
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

     
    def __getitem__(self, index):
        
        source_text = self.data['text'].iloc[index]
        source = self.tokenize(source_text)

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()

        label = self.data[self.intell].iloc[index]
        # print('FEAT', nlp_features)
        return {
            "source_ids": source_ids.to(dtype=torch.long),
            "source_mask": source_mask.to(dtype=torch.long),
            "labels":  label, 
            # "nlp_feat": nlp_features.to(dtype=torch.float)
        }


class RNN_Block(nn.Module):
    
    def __init__(self, input_size=768, hidden_size=512, rnn='LSTM', biderectional=True):
        super().__init__()

        rnn_type = {"GRU": nn.GRU, "LSTM": nn.LSTM, "RNN": nn.RNN}
        self.rnn = rnn_type[rnn](input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=biderectional)


    def forward(self, input_ids):

        rnn_output = self.rnn(input_ids)
        return rnn_output


class CNN_Block(nn.Module):
    
    def __init__(self, input_size=1024, out_size=64, kernel_size=3, stride=1, padding=1):
        super().__init__()

        self.conv_2 = nn.Conv1d(in_channels=input_size, out_channels=out_size,
                                kernel_size=2, stride=stride, padding=padding)
        self.conv_3 = nn.Conv1d(in_channels=input_size, out_channels=out_size,
                                kernel_size=3, stride=stride, padding=padding)
        self.conv_5 = nn.Conv1d(in_channels=input_size, out_channels=out_size,
                                kernel_size=5, stride=stride, padding=padding+1)
        self.relu = nn.ReLU()


    def forward(self, sequence_input):

        conv_input = sequence_input.permute(0, 2, 1) # batch_size, hidden_size, sequence_length
        cnn_output2 = self.conv_2(conv_input)
        cnn_output2 = self.relu(cnn_output2)
        cnn_output2 = F.max_pool1d(cnn_output2, kernel_size=cnn_output2.shape[2])
        cnn_output3 = self.conv_3(conv_input)
        cnn_output3 = self.relu(cnn_output3)
        cnn_output3 = F.max_pool1d(cnn_output3, kernel_size=cnn_output3.shape[2])
        cnn_output5 = self.conv_5(conv_input)
        cnn_output5 = self.relu(cnn_output5)
        cnn_output5 = F.max_pool1d(cnn_output5, kernel_size=cnn_output5.shape[2])
        cnn_output = torch.cat([cnn_output2.squeeze(dim=2), cnn_output3.squeeze(dim=2), cnn_output5.squeeze(dim=2)], dim=1)

        return cnn_output


class BIGRU_BILSTM_CNN(nn.Module):
    
    def __init__(self, out_features, hidden_size=512, hidden_size_lin=512,
                 p_spatial_dropout=0.5, out_chanels_cnn=256, kernel_size_cnn=3, 
                 stride_cnn=1, padding_cnn=1, 
                 rnn_type='LSTM', biderectional=True, pre_trained='DeepPavlov/rubert-base-cased-sentence'):
        super(BIGRU_BILSTM_CNN, self).__init__()

        self.bert = AutoModel.from_pretrained(pre_trained)

        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.p_spatial_dropout = p_spatial_dropout
        self.bigru_block = RNN_Block(input_size=self.bert.config.hidden_size, hidden_size=hidden_size, rnn='GRU', biderectional=True) 
        self.bilstm_block = RNN_Block(input_size=self.bert.config.hidden_size, hidden_size=hidden_size, rnn='LSTM', biderectional=True) 
        
        if biderectional == True:
            self.cnn_block = CNN_Block(input_size=hidden_size*2, out_size=out_chanels_cnn,
                                      kernel_size=kernel_size_cnn, stride=stride_cnn, padding=padding_cnn)
        else:
            self.cnn_block = CNN_Block(input_size=hidden_size, out_size=out_chanels_cnn,
                                      kernel_size=kernel_size_cnn, stride=stride_cnn, padding=padding_cnn)

        self.fc2 = nn.Linear(out_chanels_cnn*3*2, 32)
        self.fc3 = nn.Linear(32, out_features)
    
    def forward(self, input_ids, attention_mask):

        # _, cls_hs = self.bert(input_ids, attention_mask = attention_mask, return_dict = False)

        encoded_layers = self.bert(input_ids=input_ids,attention_mask=attention_mask, output_hidden_states=True)
        encoded_layers = encoded_layers['last_hidden_state']#.permute(1, 0, 2)

        # spatial dropout
        x = encoded_layers.permute(0, 2, 1)   # convert to [batch, channels, time]
        x = F.dropout2d(x, self.p_spatial_dropout)
        x = x.permute(0, 2, 1)   # back to [batch, time, channels]

        bigru, (last_hidden_bigru, last_cell_bigru) = self.bigru_block(x) #(N,L,H in) - batch size, sequence length, input size
        bilstm, (last_hidden_bilstm, last_cell_bilstm) = self.bilstm_block(x)
        bilstm_cnn = self.cnn_block(bilstm)
        bigru_cnn = self.cnn_block(bigru)
        concat_blocks = torch.cat([bigru_cnn, bilstm_cnn], dim=1)
        x = self.fc2(concat_blocks)
        x = self.dropout(x)
        x = self.fc3(x)
        return x


def train(model, data_loader, device, optimizer, criterion, n_epoch):

    logger.info('Epoch #{}\n'.format(n_epoch+1))

    train_losses = []
    train_labels = []
    train_predictions = []

    progress_bar = tqdm(total=math.ceil(len(data_loader.dataset)/data_loader.batch_size), desc='Epoch {}'.format(n_epoch + 1))

    model.train()

    for _, data in enumerate(data_loader, 0):


        input_ids = data["source_ids"].to(device)
        attention_mask = data["source_mask"].to(device)
        labels = data['labels'].to(device)

        optimizer.zero_grad()

        with autocast():
            pred = model(input_ids=input_ids, attention_mask=attention_mask)
            
            loss = criterion(pred, labels)
              
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
        scheduler.step()
          
        _, predict = torch.max(pred.cpu().data, 1)
        train_losses.append(loss.item())
        train_labels.extend(labels.cpu().detach().numpy())
        train_predictions.extend(predict.cpu().detach().numpy())

        progress_bar.set_postfix(loss=np.mean(train_losses))
        progress_bar.update(1)
    
    progress_bar.update(1)
    progress_bar.close()
    
    logger.info('\n\nMean Loss after epoch #{0} - {1}'.format(str(n_epoch + 1), np.mean(train_losses)))
    logger.info('F1 score after epoch #{0} on train - {1}\n'.format(str(n_epoch + 1), f1_score(train_labels, train_predictions, average='macro')))
    logger.info('Accuracy score after epoch #{0} on train - {1}\n'.format(str(n_epoch + 1), accuracy_score(train_labels, train_predictions)))

    logger.info(classification_report(train_labels, train_predictions))
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

    logger.info('\n\nMean Loss after epoch #{0} - {1}'.format(str(n_epoch + 1), np.mean(val_losses)))
    logger.info('F1 score after epoch #{0} on validation - {1}\n'.format(str(n_epoch + 1), f1_score(val_labels, val_predictions, average='macro')))
    logger.info('Accuracy score after epoch #{0} on validation - {1}\n'.format(str(n_epoch + 1), accuracy_score(val_labels, val_predictions)))
    
    logger.info(classification_report(val_labels, val_predictions))
    return valid_stats


def test(data_loader, device, id_name):
    
#     model = BertBaseline_ResNet(model_name=path_to_model, out_features=len(dataset[intelligence].unique()))
    model.load_state_dict(torch.load(f'results/model_with_cnn_{id_name}.pth'))
    
    test_labels, test_predictions = [], []

    model.eval()

    for _, data in enumerate(data_loader, 0):
          
        input_ids = data["source_ids"].to(device)
        attention_mask = data["source_mask"].to(device)
        labels = data['labels'].to(device)

        with torch.no_grad():
            pred = model(input_ids, attention_mask)
          
        _, predict = torch.max(pred.cpu().data, 1)

        test_labels.extend(labels.cpu().detach().numpy())
        test_predictions.extend(predict.cpu().detach().numpy())

    logger.info('F1 macro score on test - {0}\n'.format(f1_score(test_labels, test_predictions, average='macro')))
    logger.info('F1 score on test - {0}\n'.format(f1_score(test_labels, test_predictions, average='weighted')))
    logger.info('Accuracy score on test - {0}\n'.format(accuracy_score(test_labels, test_predictions)))
    
    logger.info(classification_report(test_labels, test_predictions))


def evaluate(model, train_dataset, val_dataset, device, epochs, target_value, weights):
    
    model = model.to(device)
    lr = 1e-5
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=2e-5)
    criterion = nn.CrossEntropyLoss(weight=weights,reduction='mean').to(device)
    global scaler
    scaler = GradScaler()
    global lr_list
    lr_list = []
    global scheduler
    total_steps = len(train_dataset) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(total_steps * 0.1),
                                                num_training_steps=total_steps)
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

                name_to_save = f'model_with_cnn_{target_value}'
                if os.path.isfile('results'+name_to_save+'.pth'):
                    os.remove('results'+name_to_save+'.pth')
                    torch.save(model.state_dict(), 'results'+name_to_save+'.pth')
                else:
                    torch.save(model.state_dict(), 'results'+name_to_save+'.pth')
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_data", default='dataset_all_nlp_features_target_classes_no_naive.csv')
    parser.add_argument("--target_value", default="verb")
    parser.add_argument("--path_to_model", default="DeepPavlov/rubert-base-cased-sentence")
    parser.add_argument("--epochs", default=15)

    args = parser.parse_args()

    path_to_data = args.path_to_data
    target_value = args.target_value
    path_to_model = args.path_to_model
    epochs = args.epochs
    seed_value = 42

    _ = seed_everything(seed_value)
    logger.info(f"Seed values {seed_value}...")
    logger.info(f"Model name: {path_to_model}...")
    logger.info(f"Prediction value: {target_value}...")
    LR = 2e-5
    logger.info(f"Learning rate: {LR}...")
    logger.info(f"Epochs: {epochs}...")
    maxlength = 120
    logger.info(f"Max length: {maxlength}...")
    minlength = 2
    logger.info(f"Min length: {minlength+1}...")
    bsize = 8
    logger.info(f"Batch size: {bsize}...")

    dataset = pd.read_csv(path_to_data, sep='\t')

    dataset = dataset[dataset.question_id != '129_Чтение текста - видео']

    if target_value == 'raven':
            dataset = dataset[dataset["raven"] > 0]
    if target_value == 'verb':
            dataset = dataset[dataset["verb"] > 0]
        
    tokenizer = AutoTokenizer.from_pretrained(path_to_model)

    intelligence = target_value+'_classes'

    dataset = dataset[dataset['N_words'] > minlength]

    dataset[intelligence] = dataset[intelligence].astype(int)

    train_data, extra_data = train_test_split(dataset, test_size=0.22,
                                        stratify=dataset[intelligence],
                                        random_state=seed_value)

    vaild_data, test_data = train_test_split(extra_data, test_size=0.45,
                                        stratify=extra_data[intelligence],
                                        random_state=seed_value)


    train_dataset_data = DataPreparation(
            tokenizer=tokenizer,
            data = train_data,
            intelligence = intelligence,
            max_length = maxlength
        )

    val_dataset_data = DataPreparation(
            tokenizer=tokenizer,
            data = vaild_data,
            intelligence = intelligence,
            max_length = maxlength
        )

    test_dataset_data = DataPreparation(
            tokenizer=tokenizer,
            data = test_data,
            intelligence = intelligence,
            max_length = maxlength
        )
    
    logger.info("All data is prepared")

    weights = class_weight.compute_class_weight(class_weight='balanced',classes=np.unique(dataset[intelligence]), y=dataset[intelligence].to_numpy())

    logger.info("Class weights is used to update the Criterion")

    wights_tensor = torch.tensor(weights,dtype=torch.float)

    train_dataset = DataLoader(train_dataset_data, batch_size=bsize, drop_last=True, shuffle=True)
    val_dataset = DataLoader(val_dataset_data, batch_size=bsize)
    test_dataset = DataLoader(test_dataset_data, batch_size=bsize)
        
    model = BIGRU_BILSTM_CNN(pre_trained=path_to_model, out_features=len(dataset[intelligence].unique()))

    logger.info("Model has been initialized")

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    evaluate(model=model, train_dataset=train_dataset, val_dataset=val_dataset, device=device, epochs=epochs, target_value=target_value, weights=wights_tensor)

    logger.info("Evaluation is finished")

    logger.info("Presictions for training set on best saved model")

    test(train_dataset, device, target_value) 

    logger.info("Presictions for validing set on best saved model")

    test(val_dataset, device, target_value)

    logger.info("Presictions for testing set on best saved model")

    test(test_dataset, device, target_value)