import yaml
from pipelines.run_pipeline import run_pipeline
from src.utils import set_seed
import wandb

def load_config(path="config/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()  
    set_seed(config['data']['random_state'])
    df_features = run_pipeline(config)

if __name__ == "__main__":
    main()