import requests
import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

def check_existing_file(file_path):
    '''Check if a file already exists. If it does, ask if we want to overwrite it.'''
    if os.path.isfile(file_path):
        while True:
            response = input(f"File {os.path.basename(file_path)} already exists. Do you want to overwrite it? (y/n): ")
            if response.lower() == 'y':
                return True
            elif response.lower() == 'n':
                return False
            else:
                print("Invalid response. Please enter 'y' or 'n'.")
    else:
        return True
    
    
def check_existing_folder(folder_path):
    '''Check if a folder already exists. If it doesn't, ask if we want to create it.'''
    if os.path.exists(folder_path) == False :
        while True:
            response = input(f"{os.path.basename(folder_path)} doesn't exists. Do you want to create it? (y/n): ")
            if response.lower() == 'y':
                return True
            elif response.lower() == 'n':
                return False
            else:
                print("Invalid response. Please enter 'y' or 'n'.")
    else:
        return False

def import_raw_data(raw_data_relative_path, 
                    filenames,
                    bucket_folder_url):
    '''import filenames from bucket_folder_url in raw_data_relative_path'''
    if check_existing_folder(raw_data_relative_path):
        os.makedirs(raw_data_relative_path)
    # download all the files
    for filename in filenames :
        input_file = os.path.join(bucket_folder_url,filename)
        output_file = os.path.join(raw_data_relative_path, filename)
        if check_existing_file(output_file):
            object_url = input_file
            print(f'downloading {input_file} as {os.path.basename(output_file)}')
            response = requests.get(object_url)
            if response.status_code == 200:
                # Process the response content as needed
                content = response.text
                text_file = open(output_file, "wb")
                text_file.write(content.encode('utf-8'))
                text_file.close()
            else:
                print(f'Error accessing the object {input_file}:', response.status_code)


# Regrouper les notes
def regrouper_notes(y):
    """
    Regroupe les notes :
    - 1, 2 -> 1 (négatif)
    - 3    -> 2 (neutre)
    - 4, 5 -> 3 (positif)
    """
    return y.replace({1: 1, 2: 1, 3: 2, 4: 3, 5: 3})


def equilibrer_par_index(x_df, y_series_grouped):
    # Obtenir les index par classe
    classes = y_series_grouped.unique()
    series_by_class = [y_series_grouped[y_series_grouped == c] for c in classes]
    
    # Taille minimale pour équilibrer
    min_count = min(len(s) for s in series_by_class)
    
    # Sous-échantillonnage équilibré
    sampled_idx = [s.sample(n=min_count, random_state=42).index for s in series_by_class]
    final_idx = pd.Index(np.concatenate(sampled_idx)).sort_values()
    
    # Appliquer aux X et Y
    x_balanced = x_df.loc[final_idx]
    y_balanced = y_series_grouped.loc[final_idx]
    
    return x_balanced, y_balanced



                
def main(raw_data_relative_path="./", 
        filenames = ["raw.csv"],
        bucket_folder_url= "https://datascientest-mlops.s3.eu-west-1.amazonaws.com/mlops_dvc_fr/"          
        ):
    """ Upload data from AWS s3 in ./
    """
    import_raw_data(raw_data_relative_path, filenames, bucket_folder_url)
    logger = logging.getLogger(__name__)
    logger.info('making raw data set')
    df = pd.read_csv("df_merged_clean.csv", sep=",")
    df = df.dropna(subset=['summary', 'reviewText', 'overall'])
    df = df.rename(columns={'overall': 'score'})
    df['score'] = df['score'].astype(int)


    target = df['score']
    feats = df[['summary', 'reviewText']]

    X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=0.3, random_state = 42)

    y_train_grouped = regrouper_notes(y_train)
    y_test_grouped = regrouper_notes(y_test)

    x_train_bal, y_train_bal = equilibrer_par_index(X_train, y_train_grouped)
    x_test_bal, y_test_bal = equilibrer_par_index(X_test, y_test_grouped)

    y_train_bal = y_train_bal - 1
    y_test_bal = y_test_bal - 1

    logger.info("Train équilibré :", y_train_bal.value_counts())
    logger.info("Test équilibré :", y_test_bal.value_counts())

    for file, filename in zip([x_train_bal, x_test_bal, y_train_bal, y_test_bal], ['X_train', 'X_test', 'y_train', 'y_test']):
        output_filepath = os.path.join("../processed_data", f'{filename}.csv')
        file.to_csv(output_filepath, index=False)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    
    main()