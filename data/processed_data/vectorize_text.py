import joblib
import pandas as pd
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import numpy as np


def vectorize(X_train, X_test) :
    try:

        tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))

        X_train_tfidf = tfidf.fit_transform(X_train['lemmes'])
        X_test_tfidf = tfidf.transform(X_test['lemmes'])


        # feature_names=tfidf.get_feature_names_out()

        # X_train_tfidf_cols = pd.DataFrame(X_train_tfidf.toarray(), columns=feature_names, index=X_train.index)
        # X_test_tfidf_cols = pd.DataFrame(X_test_tfidf.toarray(), columns=feature_names, index=X_test.index)

        # return X_train_tfidf_cols, X_test_tfidf_cols
        return X_train_tfidf, X_test_tfidf, tfidf
    except Exception as e:
        logger.error(e)
        raise


def get_vader_score(text, sia):
    return sia.polarity_scores(str(text))['compound']


if __name__ == '__main__':
    try:
        log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(level=logging.INFO, format=log_fmt)

        logger = logging.getLogger(__name__)
        logger.info('vectorizing text')

        X_train = pd.read_pickle("X_train.pickle")
        X_test = pd.read_pickle("X_test.pickle")
        X_train_tfidf, X_test_tfidf, tfidf = vectorize(X_train, X_test)
        
        # logger.info('calculating VADER score')
        # nltk.download('vader_lexicon', quiet=True)
        # sia = SentimentIntensityAnalyzer()

        # X_train_vader = X_train['text'].apply(get_vader_score, sia).values.reshape(-1, 1)
        # X_test_vader = X_test['text'].apply(get_vader_score, sia).values.reshape(-1, 1)

        # X_train_final = np.hstack((X_train_tfidf_array, X_train_vader))
        # X_test_final = np.hstack((X_test_tfidf_array, X_test_vader))

        logger.info('saving vectorized data set')
        X_train_tfidf.to_pickle("X_train_tfidf.pickle")
        X_test_tfidf.to_pickle("X_test_tfidf.pickle")
        joblib.dump(tfidf, "tfidf_vectorizer.pkl")


    except Exception as e:
        logger.error(e)