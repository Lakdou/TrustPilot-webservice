
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn import metrics
import lightgbm as lgb
import json


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
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train_tfidf, y_train,
        eval_set=[(X_test_tfidf, y_test)],
        eval_metric='multi_logloss',
        callbacks=[lgb.early_stopping(50)]
    )
    return model


def calculate_feature_importance(model, tfidf):
    importances = pd.DataFrame({
        'feature': tfidf.get_feature_names_out(),
        'importance': model.feature_importances_
    }).sort_values(by='importance', ascending=False)
    return importances


if __name__ == "__main__":
    try:
        log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(level=logging.INFO, format=log_fmt)
        logger = logging.getLogger(__name__)
        
        X_train_tfidf = pd.read_pickle("../processed_data/X_train_tfidf.pickle")
        X_test_tfidf = pd.read_pickle("../processed_data/X_test_tfidf.pickle")
        y_train = pd.read_pickle("../processed_data/y_train.pickle")
        y_test = pd.read_pickle("../processed_data/y_test.pickle")
        tfidf = joblib.load("../processed_data/tfidf_vectorizer.pkl")
        X_test = pd.read_pickle("../processed_data/X_test.pickle")
        
        logger.info("Training LightGBM model")
        model = train_model(lgb.LGBMClassifier(**params), X_train_tfidf, y_train, X_test_tfidf, y_test)

        logger.info("Saving model")
        joblib.dump(model, 'trustpilot_lgbm_model.pkl') 
        
        logger.info("Calculating feature importance")
        importances = calculate_feature_importance(model, tfidf)

        logger.info("Saving feature importance in metrics/feature_importance.csv")
        importances.to_csv("../metrics/feature_importance.csv", index=False)
        
    except Exception as e:
        logger.error(e)



