# 🛫 Travel Insurance Claim Prediction

Questo progetto partecipa alla challenge AI Planet per prevedere se un cliente richiederà un rimborso assicurativo. Il dataset è sbilanciato e include variabili categoriche, numeriche e testuali. Sono state applicate tecniche di bilanciamento, selezione delle feature e tuning iperparametrico.

## 📦 Contenuto

- Preprocessing con One-Hot Encoding e SMOTE
- Modelli: Logistic Regression, Decision Tree, Random Forest
- GridSearchCV e RFE per tuning e selezione
- Valutazione con Accuracy, F1-score e Confusion Matrix
- Predizione su test set e salvataggio CSV

## 🚀 Avvio rapido


cd insurance-claim-prediction
bash setup.sh

## 📊 Risultati
Modello	Accuracy	F1-score
Logistic Regression	
Decision Tree	
Random Forest	
Random Forest + GridSearchCV	
Random Forest + RFE + GridSearchCV

## 📚 Notebook
Il notebook insurance_claim_model.ipynb mostra l’intero flusso: EDA, preprocessing, modellazione, tuning e predizione.
