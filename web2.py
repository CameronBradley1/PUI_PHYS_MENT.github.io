from flask import Flask, render_template, request
import pandas as pd
import os
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score


matplotlib.use('agg')

def evaluate_model(pipeline_rf, pipeline_gb, X_test, y_test):
    # Make predictions on the test set using the best Random Forest model
    y_pred_rf = pipeline_rf.predict(X_test)

    # Make predictions on the test set using the best Gradient Boosting model
    y_pred_gb = pipeline_gb.predict(X_test)

    # Calculate accuracy for each column (multi-output classification task) for both models
    accuracy_rf_phys14d = accuracy_score(y_test['_PHYS14D'], y_pred_rf[:, 0])
    accuracy_rf_ment14d = accuracy_score(y_test['_MENT14D'], y_pred_rf[:, 1])

    accuracy_gb_phys14d = accuracy_score(y_test['_PHYS14D'], y_pred_gb[:, 0])
    accuracy_gb_ment14d = accuracy_score(y_test['_MENT14D'], y_pred_gb[:, 1])

    # Calculate F1-score for each column for both models
    f1_score_rf_phys14d = f1_score(y_test['_PHYS14D'], y_pred_rf[:, 0], average='weighted')
    f1_score_rf_ment14d = f1_score(y_test['_MENT14D'], y_pred_rf[:, 1], average='weighted')

    f1_score_gb_phys14d = f1_score(y_test['_PHYS14D'], y_pred_gb[:, 0], average='weighted')
    f1_score_gb_ment14d = f1_score(y_test['_MENT14D'], y_pred_gb[:, 1], average='weighted')

    # Calculate Precision score for each column for both models
    precision_rf_phys14d = precision_score(y_test['_PHYS14D'], y_pred_rf[:, 0], average='weighted')
    precision_rf_ment14d = precision_score(y_test['_MENT14D'], y_pred_rf[:, 1], average='weighted')

    precision_gb_phys14d = precision_score(y_test['_PHYS14D'], y_pred_gb[:, 0], average='weighted')
    precision_gb_ment14d = precision_score(y_test['_MENT14D'], y_pred_gb[:, 1], average='weighted')

    # Calculate Recall score for each column for both models
    recall_rf_phys14d = recall_score(y_test['_PHYS14D'], y_pred_rf[:, 0], average='weighted')
    recall_rf_ment14d = recall_score(y_test['_MENT14D'], y_pred_rf[:, 1], average='weighted')

    recall_gb_phys14d = recall_score(y_test['_PHYS14D'], y_pred_gb[:, 0], average='weighted')
    recall_gb_ment14d = recall_score(y_test['_MENT14D'], y_pred_gb[:, 1], average='weighted')

    '''
    # Calculate AUC for '_PHYS14D' column for both models
    auc_rf_phys14d = roc_auc_score(y_test_phys14d, y_pred_rf_phys14d, multi_class='ovr')
    auc_gb_phys14d = roc_auc_score(y_test_phys14d, y_pred_gb[:, 1], multi_class='ovr')

    # ROC AUC for '_MENT14D' column
    auc_rf_ment14d = roc_auc_score(y_test['_MENT14D'], y_pred_rf[:, 1], multi_class='ovr')
    auc_gb_ment14d = roc_auc_score(y_test['_MENT14D'], y_pred_gb[:, 1], multi_class='ovr')
    '''
    return (
         accuracy_rf_phys14d, accuracy_rf_ment14d,
        accuracy_gb_phys14d, accuracy_gb_ment14d,
        f1_score_rf_phys14d, f1_score_rf_ment14d,
        f1_score_gb_phys14d, f1_score_gb_ment14d,
        precision_rf_phys14d, precision_rf_ment14d,
        precision_gb_phys14d, precision_gb_ment14d,
        recall_rf_phys14d, recall_rf_ment14d,
        recall_gb_phys14d, recall_gb_ment14d
    )

app = Flask(__name__)
current = os.getcwd()
from sklearn.preprocessing import StandardScaler

# Function to load the dataset and extract features
def load_dataset():
    data = pd.read_csv(os.path.join(current, 'HumanData', 'output.csv'))
    features = data.columns.tolist()
    features.remove('_PHYS14D')
    features.remove('_MENT14D')
    features.remove('PHYSHLTH')
    features.remove('MENTHLTH')
    return features

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV


def preprocess_data(X, y):
    # Combine the features and target variables
    data = pd.concat([X, y], axis=1)

    # Drop rows where either _PHYS14D or _MENT14D is NaN or equal to 9
    data_cleaned = data.dropna(subset=['_PHYS14D', '_MENT14D'])
    data_cleaned = data_cleaned[~data_cleaned['_PHYS14D'].isin([9])]
    data_cleaned = data_cleaned[~data_cleaned['_MENT14D'].isin([9])]

    # Separate the cleaned features and target variables
    X_cleaned = data_cleaned.drop(columns=['_PHYS14D', '_MENT14D'], errors='ignore')
    y_cleaned = data_cleaned[['_PHYS14D', '_MENT14D']]

    # Standardize the features using StandardScaler
    scaler = StandardScaler()
    X_cleaned = scaler.fit_transform(X_cleaned)

    # Impute missing values in features using SimpleImputer with 'mean' strategy
    imputer = SimpleImputer(strategy='mean')
    X_cleaned = imputer.fit_transform(X_cleaned)

    return X_cleaned, y_cleaned

def train_model(selected_features, perform_tuning, hyperparameter_grid):
    data = pd.read_csv(os.path.join(current, 'HumanData', 'output.csv'))
    X = data[selected_features]
    y = data[['_PHYS14D', '_MENT14D']]

    selected_features = [feat for feat in selected_features if feat not in ['_PHYS14D', '_MENT14D']]

    X_cleaned, y_cleaned = preprocess_data(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X_cleaned, y_cleaned, test_size=0.1, random_state=50)

    class_proportions = [0.671760, 0.210522, 0.117718]

    # Calculate class weights for each label (output) based on class proportions
    class_weights_rf = {i + 1: 1.0 / class_proportions[i] for i in range(len(class_proportions))}


    base_model_rf = RandomForestClassifier(random_state=50, class_weight=class_weights_rf) 
    base_model_gb = GradientBoostingClassifier(random_state=50) 

    model_rf = MultiOutputClassifier(base_model_rf)
    model_gb = MultiOutputClassifier(base_model_gb)

    pipeline_rf = Pipeline(steps=[('model', model_rf)])
    pipeline_gb = Pipeline(steps=[('model', model_gb)])
    
    if perform_tuning:
        hyperparameter_grid = {
            'n_estimators': [100],
            'max_depth': [20],
            'min_samples_split': [5]
        }

        param_grid = {
            'model__estimator__n_estimators': hyperparameter_grid['n_estimators'],
            'model__estimator__max_depth': hyperparameter_grid['max_depth'],
            'model__estimator__min_samples_split': hyperparameter_grid['min_samples_split']
        }

        # Create GridSearchCV for RandomForestClassifier
        grid_rf = GridSearchCV(pipeline_rf, param_grid, cv=3, verbose=2, n_jobs=-1)
        # Fit the grid search with training data
        grid_rf.fit(X_train, y_train)

        # Create GridSearchCV for GradientBoostingClassifier
        grid_gb = GridSearchCV(pipeline_gb, param_grid, cv=3, verbose=2, n_jobs=-1)
        # Fit the grid search with training data
        grid_gb.fit(X_train, y_train)

        # Get the best models from the GridSearchCV
        pipeline_rf = grid_rf.best_estimator_
        pipeline_gb = grid_gb.best_estimator_

        # Get the best hyperparameters
        best_params_rf = grid_rf.best_params_
        best_params_gb = grid_gb.best_params_

    else:

        # Training the RandomForestClassifier model with undersampled data for both labels
        pipeline_rf.fit(X_train, y_train)

        # Training the GradientBoostingClassifier model with original data
        pipeline_gb.fit(X_train, y_train)
    
    y_pred_rf = pipeline_rf.predict(X_test)
    y_pred_gb = pipeline_gb.predict(X_test)

    return pipeline_rf, pipeline_gb, X_test, y_test, y_pred_rf, y_pred_gb




import json

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        data = json.loads(request.data)
        selected_features = data.get('selected_features', [])
        perform_tuning = data.get('perform_tuning', False)  # Get the user's choice for hyperparameter tuning
        n_estimators = [int(x) for x in data.get('n_estimators', '').split(',') if x.strip()]
        max_depth = [int(x) if x.strip().lower() != 'none' else None for x in data.get('max_depth', '').split(',') if x.strip()]
        min_samples_split = [int(x) for x in data.get('min_samples_split', '').split(',') if x.strip()]
        hyperparameter_grid = {
            'model__estimator__n_estimators': n_estimators,
            'model__estimator__max_depth': max_depth,
            'model__estimator__min_samples_split': min_samples_split
        }

        # Check if any features are selected
        if not selected_features:
            print("Error")
            error_message = "Error: You have not selected any features."
            features = load_dataset()  # Load the dataset to display on the web form
            return render_template('index2.html', features=features, selected_features=[], error_message=error_message)

        # Perform model training using the selected features
        print("Calling train_model function...")
        print(selected_features)
        pipeline_rf, pipeline_gb, X_test_cleaned, y_test_cleaned, y_pred_rf, y_pred_gb = train_model(selected_features, perform_tuning, hyperparameter_grid)
        print("Model training completed!")

        # Evaluate the models and get performance metrics
        (
            accuracy_rf_phys14d, accuracy_rf_ment14d,
            accuracy_gb_phys14d, accuracy_gb_ment14d,
            f1_score_rf_phys14d, f1_score_rf_ment14d,
            f1_score_gb_phys14d, f1_score_gb_ment14d,
            precision_rf_phys14d, precision_rf_ment14d,
            precision_gb_phys14d, precision_gb_ment14d,
            recall_rf_phys14d, recall_rf_ment14d,
            recall_gb_phys14d, recall_gb_ment14d
        ) = evaluate_model(pipeline_rf, pipeline_gb, X_test_cleaned, y_test_cleaned)

        # Return the result template with all evaluation metrics
        return json.dumps({
            '_PHYS14D': {
                'accuracy_rf': accuracy_rf_phys14d,
                'accuracy_gb': accuracy_gb_phys14d,
                'f1_score_rf': f1_score_rf_phys14d,
                'f1_score_gb': f1_score_gb_phys14d,
                'precision_rf': precision_rf_phys14d,
                'precision_gb': precision_gb_phys14d,
                'recall_rf': recall_rf_phys14d,
                'recall_gb': recall_gb_phys14d
            },
            '_MENT14D': {
                'accuracy_rf': accuracy_rf_ment14d,
                'accuracy_gb': accuracy_gb_ment14d,
                'f1_score_rf': f1_score_rf_ment14d,
                'f1_score_gb': f1_score_gb_ment14d,
                'precision_rf': precision_rf_ment14d,
                'precision_gb': precision_gb_ment14d,
                'recall_rf': recall_rf_ment14d,
                'recall_gb': recall_gb_ment14d
            }
        })

        
    features = load_dataset()
    return render_template('index2.html', features=features, selected_features=json.dumps([]), error_message="")



if __name__ == '__main__':
    app.run(debug=True)
