import pandas as pd
import joblib
from sklearn.pipeline import Pipeline

# Define the features and target columns
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

# Function to load the models and make predictions
def recommend(user_input):
    # Transform the user input into DataFrame
    user_input_df = pd.DataFrame([user_input], columns=features)

    # Predict recommendations using all the models
    results = {}
    for target_name, (_, model_type) in target_columns.items():
        try:
            # Load the saved model
            model = joblib.load(f'model_{target_name}.pkl')
            # Make a prediction
            if model_type == 'numerical':
                results[target_name] = model.predict(user_input_df)[0]
            else:  # categorical
                results[target_name] = model.predict(user_input_df)[0]
        except FileNotFoundError:
            results[target_name] = "Model not trained"
        except Exception as e:
            results[target_name] = f"Error: {e}"

    return results

# Example function to get user input (you can replace this with actual input gathering code)
def get_user_input():
    user_input = {
        'Sex': input("Enter Sex (Male/Female): "),
        'Age': int(input("Enter Age: ")),
        'Height': int(input("Enter Height (cm): ")),
        'Weight': int(input("Enter Weight (kg): ")),
        'Hypertension': input("Do you have Hypertension? (Yes/No): "),
        'Diabetes': input("Do you have Diabetes? (Yes/No): "),
        'BMI': float(input("Enter BMI: ")),
        'Level': input("Enter Fitness Level (Beginner/Intermediate/Advanced): "),
        'Fitness Goal': input("Enter Fitness Goal (Muscle Gain/Fat Loss/Endurance): "),
        'Fitness Type': input("Enter Fitness Type (Strength Training/Cardio/Mixed): "),
        'Diet Type': input("Enter Diet Type (Veg/Non-Veg): ")
    }
    return user_input

# Get user input
user_input = get_user_input()

# Get recommendations
recommendations = recommend(user_input)

# Display the results
print("\nPredicted Recommendations:")
for target_name, prediction in recommendations.items():
    print(f"{target_name}: {prediction}")
