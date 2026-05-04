import wandb
from sklearn.preprocessing import RobustScaler
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from src.model import MLP
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def prepare_dataloaders(train_df, test_df, target_col, batch_size):
    X_train = train_df.drop(columns=[target_col]).values.astype(np.float32)
    y_train = train_df[target_col].values.astype(np.float32).reshape(-1, 1)
    X_test = test_df.drop(columns=[target_col]).values.astype(np.float32)
    y_test = test_df[target_col].values.astype(np.float32).reshape(-1, 1)
    
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, scaler

def train_model(config, train_loader, test_loader, input_dim):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(input_dim, config['model']['hidden_sizes'], dropout=config['model']['dropout']).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['model']['learning_rate'])
    
    best_loss = float('inf')
    patience = config['model']['early_stopping_patience']
    counter = 0

    wandb.watch(model, log="all")

    for epoch in range(config['model']['epochs']):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        correct = 0
        all_preds = []   #F1
        all_labels = []  #F1
        all_probs = []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                output = model(X_batch)
                loss = criterion(output, y_batch)
                #Vall_loss need for see dates what not are see, if her be too much high than train_loss, mean that model is only remenber the train ago
                val_loss += loss.item() * X_batch.size(0) 
                
                threshold = 0.2
                pred = (torch.sigmoid(output) > threshold   ).float()
                probs = torch.sigmoid(output)
                correct += (pred == y_batch).sum().item()

                all_probs.extend(probs.cpu().numpy().flatten())
                all_preds.extend(pred.cpu().numpy())      #F1
                all_labels.extend(y_batch.cpu().numpy())  #F1
        try:
            #
            auc_roc = roc_auc_score(all_labels, all_probs)
        except:
            auc_roc = 0.5  

        val_loss /= len(test_loader.dataset) 
        acc = correct / len(test_loader.dataset)
        f1 = f1_score(all_labels, all_preds, zero_division=0) #New Metrics
        recall = recall_score(all_labels, all_preds, zero_division=0)      
        precision = precision_score(all_labels, all_preds, zero_division=0) 

        wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_acc": acc,
                   "val_f1": f1, "val_auc": auc_roc,"val_recall": recall,"val_precision": precision}) #New metrics

        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), "best_model.pt")

        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break
    model_artifact = wandb.Artifact("trained_model", type="model")
    model_artifact.add_file("best_model.pt")
    wandb.log_artifact(model_artifact)    
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
        
    wandb.log({f"confusion_matrix - epoch {epoch}": wandb.Image(plt)})
    plt.close()

    model.load_state_dict(torch.load("best_model.pt"))
    return model

def training_save_model(config):
    wandb.init(project="cervical-cancer-analysis", job_type="train", name="test", config=config)

    train_df = pd.read_csv("temp_train.csv")
    test_df = pd.read_csv("temp_test.csv")

    train_loader, test_loader, scaler = prepare_dataloaders(
        train_df, test_df, config['data']['target_col'], config['model']['batch_size']
    )
    input_dim = train_loader.dataset.tensors[0].shape[1]

    model = train_model(config, train_loader, test_loader, input_dim)

    wandb.finish()
    print("Treinamento concluído e melhor modelo salvo no W&B.")