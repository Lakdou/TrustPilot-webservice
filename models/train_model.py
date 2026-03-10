
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# =============================================
# 🔹 Préparation des données texte
# =============================================

# ⚠️ Remplace ces lignes par ton propre jeu de données :
# X_train_encoded['text'] = tes phrases
# X_train_encoded['label'] = tes classes (1, 2, 3 par ex.)
# Exemple :
# X_train_encoded = pd.DataFrame({'text': x_text, 'label': y})




X_train_tfidf = pd.read_pickle("../preprocessed_data/X_train_tfidf.pickle")
X_test_tfidf = pd.read_pickle("../preprocessed_data/X_test_tfidf.pickle")

# =============================================
# 🔹 Configuration du modèle LightGBM
# =============================================

params = {
    'objective': 'multiclass',
    'num_class': 3,  # adapter selon le nb de classes
    'learning_rate': 0.05,
    'num_leaves': 64,
    'max_depth': -1,
    'n_estimators': 1000,
    'colsample_bytree': 0.8,
    'subsample': 0.8,
    'random_state': 42,
    'metric': 'multi_logloss'
}
def train_model(model, X_train_tfidf, y_train):
    model = lgb.LGBMClassifier(**params)

# =============================================
# 🔹 Entraînement avec early stopping
# =============================================

model.fit(
    X_train_tfidf, y_train,
    eval_set=[(X_test_tfidf, y_test)],
    eval_metric='multi_logloss',
    callbacks=[lgb.early_stopping(50)]
)

# =============================================
# 🔹 Prédictions et évaluation
# =============================================

y_pred = model.predict(X_test_tfidf)

print("\n🔹 Rapport de classification :")
print(classification_report(y_test, y_pred, digits=3))

# =============================================
# 🔹 Matrice de confusion
# =============================================

plt.figure(figsize=(6,5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Matrice de confusion - LightGBM")
plt.xlabel("Prédictions")
plt.ylabel("Véritables classes")
plt.show()

# =============================================
# 🔹 Importance des features
# =============================================

importances = pd.DataFrame({
    'feature': tfidf.get_feature_names_out(),
    'importance': model.feature_importances_
}).sort_values(by='importance', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(data=importances.head(20), x='importance', y='feature')
plt.title("Top 20 mots les plus importants - LightGBM")
plt.show()




# 1. Sauvegarde du modèle entraîné
joblib.dump(model, 'trustpilot_lgbm_model.pkl')

# 2. Sauvegarde du vectoriseur (TRES IMPORTANT)
# Il contient le vocabulaire exact appris pendant l'entraînement
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')


