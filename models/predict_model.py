
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn import metrics
import lightgbm as lgb
from sklearn.metrics import classification_report, confusion_matrix
import json


def predict(model, X_test_tfidf):
    y_pred = model.predict(X_test_tfidf)
    return y_pred


def calculate_metrics(y_test, y_pred):
    return classification_report(y_test, y_pred, digits=3), confusion_matrix(y_test, y_pred)


def save_metrics(report, conf_matrix):
    metrics = {
        "classification_report": report,
        "confusion_matrix": conf_matrix.tolist()  # Convertir en liste pour JSON
    }
    with open("../metrics/scores.json", "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info("Métriques sauvegardées dans metrics/scores.json")


def save_predictions(X_test, y_test, y_pred):
    predictions_df = X_test
    predictions_df["y_true"] = y_test
    predictions_df["y_pred"] = y_pred

    predictions_df.to_csv("../data/predictions/predictions.csv", index=False)
    logger.info("Prédictions sauvegardées dans data/predictions.csv")





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
        model = joblib.load("trustpilot_lgbm_model.pkl")
        

        logger.info("Predicting on test set")
        y_pred = predict(model, X_test_tfidf)

        logger.info("Saving model")
        joblib.dump(model, 'trustpilot_lgbm_model.pkl')

        logger.info("Calculating metrics")  
        report, conf_matrix = calculate_metrics(y_test, y_pred)
        logger.info("Classification Report:\n", report)
        logger.info("Confusion Matrix:\n", conf_matrix)
        
        save_metrics(report, conf_matrix)
        logger.info("Metrics saved")
        
        save_predictions(X_test, y_test, y_pred)
        logger.info("Predictions saved")


    except Exception as e:
        logger.error(e)