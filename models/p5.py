import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
import joblib

# Load your dataset from an Excel file
df = pd.read_excel('/Users/vedantiawate/Desktop/gymapp 2/ninja1234.xlsx')

# Print column names to verify
print("Columns in DataFrame:", df.columns.tolist())

# Define feature and target columns
features = ['Sex', 'Age', 'Height', 'Weight', 'Hypertension', 'Diabetes', 'BMI', 'Level', 'Fitness Goal', 'Fitness Type', 'Diet Type']
target_columns = {
    'Exercises': ('Exercises', 'categorical'),
    'Equipment': ('Equipment', 'categorical'),
    'Recommendation': ('Recommendation', 'categorical'),
    'Weeks to Reach Goal': ('Weeks to Reach Goal', 'numerical'),
    'Total Protein Intake': ('Total Protein Intake (grams)', 'numerical'),
    'BMR': ('BMR', 'numerical'),
    'Total Calorie Intake': ('Total Calorie Intake', 'numerical'),
    'Breakfast': ('Breakfast', 'categorical'),
    'Lunch': ('Lunch', 'categorical'),
    'Snacks': ('Evening Snacks', 'categorical'),
    'Dinner': ('Dinner', 'categorical')
}

# Ensure all required columns are present
required_columns = features + [col for col, _ in target_columns.values()]
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    print(f"Missing columns: {missing_columns}")
else:
    print("All required columns are present.")

# Preprocessing function
def get_preprocessor():
    categorical_features = ['Sex', 'Hypertension', 'Diabetes', 'Level', 'Fitness Goal', 'Fitness Type', 'Diet Type']
    numeric_features = ['Age', 'Height', 'Weight', 'BMI']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ]), numeric_features),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_features)
        ])

    return preprocessor

# Prepare data for each target
def prepare_data(df, target):
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in the DataFrame.")

    X = df[features]
    y = df[target]
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate model
def train_and_evaluate(X_train, X_test, y_train, y_test, model_type):
    preprocessor = get_preprocessor()
    
    if model_type == 'numerical':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:  # categorical
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    model_pipeline.fit(X_train, y_train)
    y_pred = model_pipeline.predict(X_test)

    if model_type == 'numerical':
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return model_pipeline, {'MAE': mae, 'MSE': mse, 'R²': r2}
    else:  # categorical
        accuracy = accuracy_score(y_test, y_pred)
        return model_pipeline, {'Accuracy': accuracy}

# Train models for each target
def train_all_models(df):
    models = {}
    accuracies = {}

    for target_name, (target_column, model_type) in target_columns.items():
        try:
            X_train, X_test, y_train, y_test = prepare_data(df, target_column)
            model, accuracy = train_and_evaluate(X_train, X_test, y_train, y_test, model_type)

            # Save model
            joblib.dump(model, f'model_{target_name}.pkl')

            # Store model and accuracy metrics
            models[target_name] = model
            accuracies[target_name] = accuracy

            print(f"Model for {target_name} - Metrics: {accuracy}")
        except ValueError as e:
            print(f"Error for target '{target_name}': {e}")

    return models, accuracies

# Train and save all models
models, accuracies = train_all_models(df)

# Print all accuracies
for target_name, accuracy in accuracies.items():
    print(f"{target_name} Model Accuracy:", accuracy)

# Define a function to make recommendations
def recommend(user_input):
    # Transform the user input into DataFrame
    user_input_df = pd.DataFrame([user_input], columns=features)

    # Predict recommendations using all the models
    results = {}
    for target_name, (_, model_type) in target_columns.items():
        try:
            model = joblib.load(f'model_{target_name}.pkl')
            if model_type == 'numerical':
                results[target_name] = model.predict(user_input_df)[0]
            else:  # categorical
                results[target_name] = model.predict(user_input_df)[0]
        except FileNotFoundError:
            results[target_name] = "Model not trained"
        except Exception as e:
            results[target_name] = f"Error: {e}"

    return results

# Example user input
user_input = {
    'Sex': 'Male',
    'Age': 25,
    'Height': 175,
    'Weight': 70,
    'Hypertension': 'No',
    'Diabetes': 'No',
    'BMI': 22.8,
    'Level': 'Intermediate',
    'Fitness Goal': 'Muscle Gain',
    'Fitness Type': 'Strength Training',
    'Diet Type': 'Veg'
}

# Get recommendations
recommendations = recommend(user_input)
print("Recommendations:", recommendations)

# Save the trained models for future use
for target_name in target_columns.keys():
    try:
        joblib.dump(models[target_name], f'model_{target_name}.pkl')
        print(f"Model {target_name} saved successfully.")
    except Exception as e:
        print(f"Error saving model {target_name}: {e}")
