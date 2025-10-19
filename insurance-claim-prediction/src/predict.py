
import pandas as pd
from preprocessing import load_data, clean_gender, drop_columns, encode_categorical
from modeling import get_random_forest
from sklearn.model_selection import train_test_split

# Load and preprocess data
train, test = load_data("data/train.csv", "data/test.csv")
train = clean_gender(train)
test = clean_gender(test)
train = drop_columns(train)
test = drop_columns(test)

# Encode categorical features
train_encoded, encoder = encode_categorical(train, fit=True)
test_encoded, _ = encode_categorical(test, encoder=encoder, fit=False)

# Separate features and target
X = train_encoded.drop(columns=['Claim'])
y = train_encoded['Claim']

# Train model
model = get_random_forest()
model.fit(X, y)

# Predict on test set
preds = model.predict(test_encoded)

# Save predictions
pd.DataFrame({'prediction': preds}).to_csv("outputs/predictions.csv", index=False)
print("✅ Predictions saved to outputs/predictions.csv")
