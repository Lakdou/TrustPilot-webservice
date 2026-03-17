
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn import metrics
import lightgbm as lgb
import json
from ml_service import get_model
import logging


log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger(__name__)

# =============================================
#  Configuration du modèle LightGBM
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

def train_model(model, X_train_tfidf, y_train, X_test_tfidf=None, y_test=None):
    model, vectorizer = get_model()
    model.fit(
        X_train_tfidf, y_train,
        eval_set=[(X_test_tfidf, y_test)],
        eval_metric='multi_logloss',
        callbacks=[lgb.early_stopping(50)]
    )
    logger.ingo("Modèle entraîné")
    return model


def calculate_feature_importance(model, tfidf):
    importances = pd.DataFrame({
        'feature': tfidf.get_feature_names_out(),
        'importance': model.feature_importances_
    }).sort_values(by='importance', ascending=False)
    return importances

def train_model_and_get_importance(model, X_train_tfidf, y_train, X_test_tfidf=None, y_test=None, tfidf=None):
    model = train_model(model, X_train_tfidf, y_train, X_test_tfidf, y_test)
    importances = calculate_feature_importance(model, tfidf)
    return model, importances

def main():
    try:        
        X_train_tfidf = np.load("data/processed_data/X_train_tfidf.npy")
        X_test_tfidf = np.load("data/processed_data/X_test_tfidf.npy")
        y_train = pd.read_csv("data/processed_data/y_train.csv", sep=',')
        y_test = pd.read_csv("data/processed_data/y_test.csv", sep=',')
        tfidf = joblib.load("models/tfidf_vectorizer_sample.pickle")
        X_test = pd.read_pickle("data/processed_data/X_test.pickle")
        
        logger.info("Training LightGBM model and Calculating feature importance")
        model, importances = train_model_and_get_importance(lgb.LGBMClassifier(**params), X_train_tfidf, y_train['score'], X_test_tfidf, y_test['score'], tfidf=tfidf)

        logger.info("Saving model")
        joblib.dump(model, 'models/trustpilot_lgbm_model_sample.pkl') 

        logger.info("Saving feature importance in metrics/feature_importance.csv")
        importances.to_csv("metrics/feature_importance.csv", index=False)

        return importances
        
    except Exception as e:
        logger.error(e)

if __name__ == "__main__":
    main()



