# Load the saved predictor
from autogluon.tabular import TabularPredictor

predictor = TabularPredictor.load(
    r"C:\Users\Tristan Leiter\Documents\MT\03_Output\Models\AutoGluon\ag_predictor"
)

# Print hyperparameters for all models
for model_name in predictor.model_names():
    info = predictor.info()["model_info"][model_name]
    print(f"\n{'='*50}")
    print(f"Model: {model_name}")
    print(f"Hyperparameters: {info.get('hyperparameters', 'N/A')}")