import pickle
import pandas as pd

def load_test_data(test_data_path):
    test_data = pd.read_csv('test_data.csv')
    return test_data

def load_model(model_path):
    with open('project.py', 'rb') as file:
        model = pickle.load(file)
    return model

def load_column_transformer(column_transformer_path):
    with open('column_transformer.pkl', 'rb') as file:
        ct = pickle.load(file)
    return ct

def transform_new_data(test_data, ct):
    return ct.transform(test_data)

def make_predictions(model, transformed_data):
    return model.predict(transformed_data)

def save_predictions_to_csv(test_data, predictions, output_filepath):
    # Create a new DataFrame for saving predictions
    predictions_df = test_data.copy()
    predictions_df['Predicted Weather Type'] = predictions
    predictions_df.to_csv(output_filepath, index=False)
    print(f"Predictions saved to '{output_filepath}'")

# Load the ColumnTransformer and logistic regression model
model_path = 'project.pkl'
column_transformer_path = 'column_transformer.pkl'

model = load_model(model_path)
column_transformer = load_column_transformer(column_transformer_path)

# Load new test data for prediction
test_data_path = 'test_data.csv'
test_data = load_test_data(test_data_path)

# Transform new test data
test_data_transformed = transform_new_data(test_data, column_transformer)

# Make predictions
predictions = make_predictions(model, test_data_transformed)

# Save predictions to a new CSV file
output_filepath = 'test_data_with_predictions.csv'
save_predictions_to_csv(test_data, predictions, output_filepath)