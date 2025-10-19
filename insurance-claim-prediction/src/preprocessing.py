import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE

def load_data(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test

def clean_gender(df):
    df['Gender'].fillna('Not Specified', inplace=True)
    return df

def drop_columns(df):
    drop_cols = ['Distribution Channel', 'Destination', 'Agency Type']
    return df.drop(columns=drop_cols)

def encode_categorical(df, encoder=None, fit=True):
    cat_cols = ['Agency', 'Gender', 'Product Name']
    cat_df = df[cat_cols]
    if fit:
        encoder = OneHotEncoder(handle_unknown='ignore')
        encoder.fit(cat_df)
    transformed = encoder.transform(cat_df).toarray()
    encoded_df = pd.DataFrame(transformed, columns=encoder.get_feature_names_out())
    df = pd.concat([df.drop(columns=cat_cols), encoded_df], axis=1)
    return df, encoder

def balance_data(X, y):
    sm = SMOTE(random_state=25, sampling_strategy=1.0)
    X_resampled, y_resampled = sm.fit_resample(X, y)
    return X_resampled, y_resampled
