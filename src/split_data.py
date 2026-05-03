import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import ks_2samp, chi2_contingency
import wandb
from src.feature_selection import select_top_features
def split_train_test(df, target_col, test_size=0.2, random_state=42):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    return train_df, test_df

def compare_distributions(train_df, test_df, columns):
    results = {}
    for col in columns:
        train_vals = train_df[col].dropna()
        test_vals = test_df[col].dropna()
        if train_df[col].dtype in ['int64', 'float64']:
            ks_stat, p_value = ks_2samp(train_vals, test_vals)
            results[col] = {'test': 'KS', 'statistic': ks_stat, 'p_value': p_value}
        else:
            train_counts = train_vals.value_counts(normalize=True)
            test_counts = test_vals.value_counts(normalize=True)
            all_cats = sorted(set(train_counts.index).union(set(test_counts.index)))
            train_probs = [train_counts.get(cat, 0) for cat in all_cats]
            test_probs = [test_counts.get(cat, 0) for cat in all_cats]
            chi2, p_value, _, _ = chi2_contingency([train_probs, test_probs])
            results[col] = {'test': 'Chi2', 'statistic': chi2, 'p_value': p_value}
    return results

def train_test_df(df,config):
    train_df, test_df = split_train_test(df, 
                                        target_col=config['data']['target_col'],
                                        test_size=config['data']['test_size'],
                                        random_state=config['data']['random_state'])

    #Feacture Select
    print(f"Treino: {len(train_df)} amostras")
    print(f"Teste:  {len(test_df)} amostras")
    print(train_df.shape)
    X_train = train_df.drop(columns=['Biopsy'])
    y_train = train_df['Biopsy']
    top_cols = select_top_features(X_train, y_train, n_top=12 ) #Numbers of select feacture, 12<15<10
    train_df = train_df[top_cols + ['Biopsy']] #Select Biopsy of own train_df
    test_df = test_df[top_cols + ['Biopsy']]


    print(f"Treino após FS: {train_df.shape}")
    print(f"Teste após FS: {test_df.shape}")
    print(f"Features: {top_cols}")

    print()
    return train_df,test_df

def split_data_save(df, config):
    train_df,test_df = train_test_df(df,config)

    feature_cols = [c for c in train_df.columns if c != config['data']['target_col']]
    comp_results = compare_distributions(train_df, test_df, feature_cols)

    wandb.init(project="cervical-cancer-analysis", job_type="split_data", name="split_data")
    train_artifact = wandb.Artifact("train_data", type="dataset")
    train_df.to_csv("temp_train.csv", index=False)
    train_artifact.add_file("temp_train.csv")
    wandb.log_artifact(train_artifact)

    test_artifact = wandb.Artifact("test_data", type="dataset")
    test_df.to_csv("temp_test.csv", index=False)
    test_artifact.add_file("temp_test.csv")
    wandb.log_artifact(test_artifact)

    comp_df = pd.DataFrame(comp_results).T
    comp_table = wandb.Table(dataframe=comp_df)
    wandb.log({"distribution_comparison": comp_table})

    wandb.summary["train_size"] = len(train_df)
    wandb.summary["test_size"] = len(test_df)

    wandb.finish()
    
    print("Artefatos de treino e teste salvos com sucesso.")