import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LassoCV
from sklearn . ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import mutual_info_classif
from statsmodels.stats.outliers_influence import variance_inflation_factor
import wandb

def changed_question_values(df):
    df = df.replace("?", np.nan)
    return df

def droping_duplicates(df):
    total_duplicatas = df.duplicated().sum() 
    print(f"Numbers of Duplicates Removed {total_duplicatas}")
    df = df.drop_duplicates() 
    return df

def remove_missing_values(df):
    limite_linhas = len(df.columns) * 0.5 
    limite_coluna = len(df) * 0.5

    print(df.shape)

    rows_before = df.shape[0]
    coluns_before = df.shape[1]

    numbers_columns_before = len(df.columns)

    df = df.dropna(axis=1,thresh=limite_coluna)

    df = df.dropna(thresh=limite_linhas)

    rows_after = df.shape[0]
    coluns_after = df.shape[1]

    print(f"Columns removed: {coluns_before - coluns_after}")
    print(f"Rows removed: {rows_before - rows_after}")

    return df

def imput_missing_values(df):

    num_imputer = SimpleImputer(strategy='median')


    df_preenchido = df.apply(pd.to_numeric, errors='coerce')


    colunas_faltando = [
        'Number of sexual partners', 'First sexual intercourse', 'Num of pregnancies', 
        'Smokes', 'Smokes (years)', 'Smokes (packs/year)', 
        'Hormonal Contraceptives', 'Hormonal Contraceptives (years)', 'IUD', 
        'IUD (years)', 'STDs', 'STDs (number)', 
        'STDs:condylomatosis', 'STDs:cervical condylomatosis', 'STDs:vaginal condylomatosis', 
        'STDs:vulvo-perineal condylomatosis', 'STDs:syphilis', 'STDs:pelvic inflammatory disease', 
        'STDs:genital herpes', 'STDs:molluscum contagiosum', 'STDs:AIDS', 
        'STDs:HIV', 'STDs:Hepatitis B', 'STDs:HPV'
    ]


    df_preenchido[colunas_faltando] = num_imputer.fit_transform(df_preenchido[colunas_faltando])

    return df_preenchido


def pipeline_clean_date(df_raw):

    df = changed_question_values(df_raw)
    df = remove_missing_values(df)
    df = imput_missing_values(df)
    df = droping_duplicates(df)

    print(df.shape)

    return df

def clean_data_save(df):

    wandb.init(project="cervical-cancer-analysis", job_type="clean_data", name="clean_data")
    artifact = wandb.Artifact("clean_data", type="dataset")
    temp_path = "temp_clean.csv"
    df_clean = pipeline_clean_date(df)
    df_clean.to_csv(temp_path, index=False)
    artifact.add_file(temp_path)
    wandb.log_artifact(artifact)
    wandb.summary["rows"] = len(df_clean)
    wandb.finish()

    return df_clean


