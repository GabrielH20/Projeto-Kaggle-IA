import wandb
import pandas as pd 
from src.data_cleaning import clean_data_save
from src.split_data import split_data_save
from src.train import training_save_model
def raw_data_save(df):

    
    wandb.init(project="cervical-cancer-analysis", job_type="load_raw", name="load_raw")
    artifact = wandb.Artifact("raw_data", type="dataset")

    temp_path = "temp_raw.csv"

    df.to_csv(temp_path, index=False)
    artifact.add_file(temp_path)
    wandb.log_artifact(artifact)

    wandb.summary["rows"] = len(df)
    wandb.summary["columns"] = list(df.columns)
    wandb.finish()

def run_pipeline(config):
    #1-Carregar dados raw
    df_raw = pd.read_csv(config['data']['raw_path'])
    

    # 2-Salvar raw (Wandb) e limpar
    raw_data_save(df_raw)

    #3-Realizar Clean data, Salvar Wandb
    df_clean = clean_data_save(df_raw)
    
    split_data_save(df_clean, config)
      
    training_save_model(config)

if __name__ == "__main__":
    run_pipeline()