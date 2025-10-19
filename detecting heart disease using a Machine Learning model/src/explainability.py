import shap

def explain_model(model, X_train_scaled, X_test_scaled):
    explainer = shap.LinearExplainer(model, X_train_scaled)
    shap_values = explainer.shap_values(X_test_scaled)
    return explainer, shap_values
