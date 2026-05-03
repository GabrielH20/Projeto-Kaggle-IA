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


def scaling_datas(df):
    scaler = RobustScaler()
    df_escalado = pd.DataFrame(
        scaler.fit_transform(df),
        columns=df.columns
    )
    return df_escalado

def lasso_feacture(X,y):
    lasso = LassoCV(cv=5, random_state=42)

    lasso.fit(X, y) 

    coef_df = pd.Series(np.abs(lasso.coef_), index=X.columns)

    scaler = MinMaxScaler()

    lasso_valores_2d = coef_df.values.reshape(-1, 1)
    lasso_norm_values = scaler.fit_transform(lasso_valores_2d)

    lasso_df_norm = pd.Series(lasso_norm_values.flatten(), index=coef_df.index).sort_values(ascending=False)

    return lasso_df_norm

def random_forest(X,y):
    rf = RandomForestClassifier (n_estimators =100,random_state=42)

    rf.fit(X,y)

    mais_importantes = pd.Series(rf.feature_importances_, index=X.columns)

    scaler = MinMaxScaler()

    rf_valores_2d = mais_importantes.values.reshape(-1, 1)
    rf_norm_values = scaler.fit_transform(rf_valores_2d)

    rf_df_norm = pd.Series(rf_norm_values.flatten(),index=mais_importantes.index).sort_values(ascending=False)

    return rf_df_norm

def mi_feacture_select(X,y):
    mi_scores = mutual_info_classif(X,y,random_state=42)
    mi_df = pd.Series(mi_scores,index=X.columns).sort_values(ascending=False)
    scaler = MinMaxScaler()
    mi_valores = mi_df.values.reshape(-1,1)
    mi_normalizado = scaler.fit_transform(mi_valores)

    mi_df_norm = pd.Series(mi_normalizado.flatten(),index=mi_df.index).sort_values(ascending=False)

    return mi_df_norm

def vif_cleaning(X):
    vif_df = pd.DataFrame()
    vif_df["Feature"] = X.columns
    vif_df['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    #print(vif_df.sort_values(by="VIF",ascending=False))

    colunas_excluir = []

    for nome, row in vif_df.iterrows():
            if row["VIF"]>=10:
                colunas_excluir.append(row["Feature"])

    X_final = X.drop(columns=colunas_excluir,errors='ignore')

    return X_final

def select_top_features(X, y, n_top=10):
    x_vif = vif_cleaning(X) 
    rf_scores = random_forest(x_vif, y)       
    mi_scores = mi_feacture_select(x_vif, y)    
    lasso_scores = lasso_feacture(x_vif, y)     
    
    table = pd.DataFrame({
        'RF': rf_scores,
        'MI': mi_scores,
        'Lasso': lasso_scores
    })
    
    table['Mean'] = table.mean(axis=1)
    
    table = table.sort_values('Mean', ascending=False)
    
    top_features = table.head(n_top).index.tolist()
    
    return top_features

""""
def pipeline_feacture_selection(df, target_colum='Biopsy'):
    X = df.drop(columns=[target_colum]) 
    y = df[target_colum]

    X_final = vif_cleaning(X)
    
    top_features = select_top_features(X_final, y)
    X_top = X_final[top_features]
    
    X_scaled = scaling_datas(X_top)
    
    df_final = pd.concat([X_scaled, y.reset_index(drop=True)], axis=1)
    
    print("testando Biopsy direto na fonte",df_final.columns)
    
    return df_final  

def feature_data_save(df, config):
    wandb.init(
        project="cervical-cancer-analysis", 
        job_type="feature_selection", 
        name="feature_selection"
    )
    
    df_final = pipeline_feacture_selection(df, target_colum=config['data']['target_col'])
    
    artifact = wandb.Artifact("selected_features", type="dataset")
    temp_path = "temp_selected_features.csv"
    df_final.to_csv(temp_path, index=False)
    artifact.add_file(temp_path)
    wandb.log_artifact(artifact)
    wandb.summary["final_features"] = df_final.shape[1]
    wandb.summary["final_shape"] = str(df_final.shape)
    
    wandb.finish()
        
    return df_final   
""" 