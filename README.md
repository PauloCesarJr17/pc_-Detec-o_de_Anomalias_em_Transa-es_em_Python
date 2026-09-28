# pc_-Detec-o_de_Anomalias_em_Transa-es_em_Python

# ==========================================
# 1. IMPORTAÇÃO DE BIBLIOTECAS E CARREGAMENTO
# ==========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    precision_recall_curve, 
    average_precision_score,
    roc_auc_score
)
from imblearn.under_sampling import RandomUnderSampler
import shap

# Carregamento do dataset (Link oficial do Kaggle/OpenML)
url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
df = pd.read_csv(url)

print(f"Dimensões do Dataset: {df.shape}")
print("Distribuição da variável alvo (Class):")
print(df['Class'].value_counts(normalize=True))

# ==========================================
# 2. PRÉ-PROCESSAMENTO E DIVISÃO TREINO/TESTE
# ==========================================
# Padronização da variável 'Amount' e remoção de 'Time'
scaler = StandardScaler()
df['Amount_Scaled'] = scaler.fit_transform(df[['Amount']])
df['Log_Amount'] = np.log1p(df['Amount'])

X = df.drop(columns=['Class', 'Time', 'Amount'])
y = df['Class']

# Divisão estratificada (mantém 0.17% de fraudes no treino e no teste)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Treino: {X_train.shape[0]} amostras | Teste: {X_test.shape[0]} amostras")

# ==========================================
# 3. TRATAMENTO DE DESBALANCEAMENTO
# ==========================================
# Aplicando Random Undersampling apenas no conjunto de TREINO
rus = RandomUnderSampler(random_state=42)
X_train_res, y_train_res = rus.fit_resample(X_train, y_train)

print(f"Distribuição do treino original: {y_train.value_counts().to_dict()}")
print(f"Distribuição após Undersampling: {y_train_res.value_counts().to_dict()}")

# ==========================================
# 4. TREINAMENTO E COMPARAÇÃO DE MODELOS
# ==========================================
models = {
    "Logistic Regression (Baseline)": LogisticRegression(max_iter=1000, random_state=42),
    "Logistic Regression (Balanced)": LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
    "Random Forest (Undersampled)": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost (Scale Pos Weight)": XGBClassifier(scale_pos_weight=(len(y_train) - sum(y_train))/sum(y_train), random_state=42)
}

# Treinamento e avaliação
results = []

for name, model in models.items():
    if "Undersampled" in name:
        model.fit(X_train_res, y_train_res)
    else:
        model.fit(X_train, y_train)
        
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    report = classification_report(y_test, y_pred, output_dict=True)
    
    results.append({
        "Modelo": name,
        "Precision (Fraude)": report['1']['precision'],
        "Recall (Fraude)": report['1']['recall'],
        "F1-Score (Fraude)": report['1']['f1-score'],
        "PR-AUC": average_precision_score(y_test, y_proba)
    })

df_results = pd.DataFrame(results)
print(df_results)

# ==========================================
# 5. AJUSTE DE LIMIAR (THRESHOLD TUNING)
# ==========================================
# Utilizando o modelo XGBoost
best_model = models["XGBoost (Scale Pos Weight)"]
y_proba = best_model.predict_proba(X_test)[:, 1]

precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)

plt.figure(figsize=(8, 5))
plt.plot(thresholds, precisions[:-1], label="Precisão", color="blue")
plt.plot(thresholds, recalls[:-1], label="Recall", color="red")
plt.xlabel("Limiar de Decisão (Threshold)")
plt.ylabel("Pontuação")
plt.title("Curva Precisão vs. Recall por Limiar (XGBoost)")
plt.legend()
plt.grid(True)
plt.show()

# Testando um limiar ajustado (ex: 0.3)
custom_threshold = 0.3
y_pred_custom = (y_proba >= custom_threshold).astype(int)

print(f"=== Relatório com Threshold Ajustado ({custom_threshold}) ===")
print(classification_report(y_test, y_pred_custom))

# ==========================================
# 6. EXPLICABILIDADE COM SHAP
# ==========================================
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

# Plot de impacto global das variáveis
plt.title("Importância das Variáveis no XGBoost (SHAP)")
shap.summary_plot(shap_values, X_test)

