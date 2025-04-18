from flask import Flask, render_template, request, redirect, session, flash, url_for
import mysql.connector
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import joblib
import decimal
from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import time
import numpy as np
from functools import wraps
from flask import redirect, url_for, session, flash
from flask import jsonify
import google.generativeai as genai
import os


app = Flask(__name__)

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' in session:
            flash('You are already logged in.', 'info')
            return redirect(url_for('profile'))
        return f(*args, **kwargs)
    return decorated_function



app.secret_key = 'your_secret_key'  # Set a secret key for session management

# Database connection parameters
db_config = {
    'user': 'root',
    'password': 'Rohit@9324',
    'host': 'localhost',
    'database': 'gymusers',
    'auth_plugin': 'mysql_native_password'
}

# Define the features used in the model
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

# Configure Gemini API Key
os.environ["GOOGLE_API_KEY"] = "AIzaSyC5bhMlFnnuKjOihoBRRte-7H8wrF5rhXc"
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize the Gemini Pro model
try:
    model = genai.GenerativeModel("gemini-1.5-pro-latest")
except Exception as e:
    print("Error initializing model:", e)

def get_diet_workout_recommendation(user_message):
    """Gets AI-generated diet and workout recommendations."""
    try:
        response = model.generate_content(user_message)
        if response and hasattr(response, "text"):  # Ensure response has text
            formatted_response = response.text.replace("*", "").replace("\n\n", "\n").strip()
            return formatted_response
        else:
            return "Error: Empty response from API."
    except Exception as e:
        return f"Error fetching response: {e}"



@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    print("Received request:", data)  # ✅ Debugging line
    
    user_message = data.get("message", "")
    if not user_message:
        return jsonify({"response": "Please enter a valid message."})

    bot_response = get_diet_workout_recommendation(user_message)
    print("Bot response:", bot_response)  # ✅ Debugging line

    return jsonify({"response": bot_response})

@app.route('/')
@admin_required
def home():
    return render_template("home.html")

@app.route('/about')
@admin_required
def about():
    return render_template("about.html")

@app.route('/login')
@admin_required
def login():
    return render_template("login.html")

@app.route('/details')
@login_required
def details():
    return render_template("details.html")

@app.route('/signup', methods=['GET', 'POST'])
@admin_required
def signup():
    if request.method == 'POST':
        try:
            conn = mysql.connector.connect(**db_config)
            cursor = conn.cursor()

            name = request.form.get('name')
            email = request.form.get('email')
            phone = request.form.get('phone')
            username = request.form.get('username')
            password = request.form.get('password')
            confirm_password = request.form.get('confirm_password')
            weight = request.form.get('weight')
            age = request.form.get('age')
            height = request.form.get('height')

            if password != confirm_password:
                return render_template('signup.html', error='Passwords do not match')

            # Check if username or email already exists
            cursor.execute("SELECT * FROM users WHERE username = %s OR email = %s", (username, email))
            existing_user = cursor.fetchone()

            if existing_user:
                return render_template('signup.html', error='Username or Email already exists')

            # Hash the password
            hashed_password = generate_password_hash(password, method='pbkdf2:sha256')

            # Insert new user
            query = "INSERT INTO users (name, email, phone, username, password, weight, age, height) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
            cursor.execute(query, (name, email, phone, username, hashed_password, weight, age, height))
            conn.commit()

            cursor.close()
            conn.close()

            return render_template('login.html', success_message='Signup successful, Welcome to MyFitness Club ❚█══█❚!')

        except mysql.connector.Error as err:
            print(f"Error signing up: {err}")
            return render_template('signup.html', error='Failed to sign up')

    return render_template("signup.html")


@app.route('/logout')
@login_required
def logout():
    session.pop('username', None)
    session.pop('user_id', None)
    flash('You have been logged out', 'info')
    return redirect(url_for('login'))

@app.route('/login_validation', methods=['POST'])
def login_validation():
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        username = request.form.get('username')
        password = request.form.get('password')

        query = "SELECT * FROM users WHERE username = %s"
        cursor.execute(query, (username,))
        user = cursor.fetchone()

        cursor.close()
        conn.close()

        if user and check_password_hash(user[5], password):  # Assuming password is at index 5
            session['username'] = username
            session['user_id'] = user[0]  # Storing user ID for easy access
            flash('Login Successful ✅', 'success')
            return redirect(url_for('profile'))
        else:
            return render_template('login.html', error='Invalid credentials')

    except mysql.connector.Error as err:
        print(f"Error connecting to MySQL: {err}")
        return render_template('login.html', error='Database connection error')

@app.route('/profile')
@login_required
def profile():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        
        user_id = session['user_id']
        query = "SELECT * FROM users WHERE id = %s"
        cursor.execute(query, (user_id,))
        user = cursor.fetchone()
        print(f"User data: {user}")
        
        cursor.close()
        conn.close()

        # Check if 'weeks' column is empty
        weeks_index = 16  # Adjust index based on your database schema
        has_weeks = user[weeks_index] is not None and user[weeks_index] != ''

        # Extract exercises column (Update index based on your schema)
        exercises_index = 18  # Adjust this index as per your DB schema
        exercises_data = user[exercises_index] if user[exercises_index] else ""

        # Split exercises, capitalize each word, and assign to days
        exercises = [exercise.strip().title() for exercise in exercises_data.split(',')] if exercises_data else []
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        workout_plan = {day: (exercises[i] if i < len(exercises) else "Rest") for i, day in enumerate(days)}

        has_none_details = any(detail is None for detail in user[1:])

        return render_template(
            'profile.html',
            user=user,
            has_none_details=has_none_details,
            has_weeks=has_weeks,
            workout_plan=workout_plan
        )

    except mysql.connector.Error as err:
        print(f"Error fetching user data: {err}")
        return render_template('profile.html', error='Failed to load user data')

@app.route('/update_profile', methods=['POST'])
@login_required
def update_profile():
    if request.method == 'POST':
        try:
            conn = mysql.connector.connect(**db_config)
            cursor = conn.cursor()

            # Get form data
            weight = request.form.get('weight')
            age = request.form.get('age')
            height = request.form.get('height')
            gender = request.form.get('gender')
            exercise_frequency = request.form.get('exercise-frequency')
            supplement_usage = request.form.get('supplement-usage')
            goal = request.form.get('goal')
            diet_type = request.form.get('diet-type')  # Added diet type

            # Example: Assuming user ID is stored in session
            user_id = session.get('user_id')  # Adjust as per your session setup

            # Update user details in the database
            query = """
            UPDATE users
            SET weight = %s, age = %s, height = %s, gender = %s,
                exercise_frequency = %s, supplement_usage = %s, goal = %s, diet_type = %s
            WHERE id = %s
            """
            cursor.execute(query, (weight, age, height, gender, exercise_frequency, supplement_usage, goal, diet_type, user_id))
            conn.commit()

            flash('Profile updated successfully ✅', 'success')
            return redirect(url_for('profile'))

        except mysql.connector.Error as err:
            print(f"Error updating profile: {err}")
            flash('Failed to update profile', 'error')
            return redirect(url_for('profile'))

        finally:
            cursor.close()
            conn.close()

    return render_template('profile.html')
from flask import Flask, render_template, request, session, redirect, url_for, flash
import mysql.connector
from flask_mail import Mail, Message
from apscheduler.schedulers.background import BackgroundScheduler
import mysql.connector
from datetime import datetime, timedelta
# Configure Flask-Mail
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'rohitnagula9324@gmail.com'  # Replace with your email
app.config['MAIL_PASSWORD'] = 'hdrw dxje ncjh rubi'  # Replace with your email password
app.config['MAIL_DEFAULT_SENDER'] = 'rohitnagula9324@gmail.com'
mail = Mail(app)
scheduler = BackgroundScheduler()
scheduler.start()

def send_email(user_email, subject, body):
    """Function to send email"""
    msg = Message(subject, recipients=[user_email])
    msg.body = body
    mail.send(msg)

def schedule_daily_emails(user_email, weeks_to_goal):
    """Schedules daily emails for the user"""
    start_date = datetime.now()
    end_date = start_date + timedelta(weeks=weeks_to_goal)

    while start_date < end_date:
        day_of_week = start_date.strftime('%A')  # Get day name

        if day_of_week == "Sunday":
            # On Sundays, send only breakfast, lunch, and dinner emails
            scheduler.add_job(send_email, 'date', run_date=start_date.replace(hour=10, minute=0), 
                              args=[user_email, "Breakfast Reminder", "Time for your healthy breakfast! 🍎"])
            scheduler.add_job(send_email, 'date', run_date=start_date.replace(hour=13, minute=0), 
                              args=[user_email, "Lunch Reminder", "Don't skip lunch! Stay energized. 🍱"])
            scheduler.add_job(send_email, 'date', run_date=start_date.replace(hour=21, minute=0), 
                              args=[user_email, "Dinner Reminder", "Enjoy your dinner! 🍽️"])
        else:
            # Regular days: Send all reminders (breakfast, exercise, lunch, dinner)
            scheduler.add_job(send_email, 'date', run_date=start_date.replace(hour=7, minute=0), 
                              args=[user_email, "Exercise Reminder", "Time to workout! 🏋️"])
            scheduler.add_job(send_email, 'date', run_date=start_date.replace(hour=10, minute=0), 
                              args=[user_email, "Breakfast Reminder", "Time for your healthy breakfast! 🍎"])
            scheduler.add_job(send_email, 'date', run_date=start_date.replace(hour=13, minute=0), 
                              args=[user_email, "Lunch Reminder", "Don't skip lunch! Stay energized. 🍱"])
            scheduler.add_job(send_email, 'date', run_date=start_date.replace(hour=21, minute=0), 
                              args=[user_email, "Dinner Reminder", "Enjoy your dinner! 🍽️"])

        start_date += timedelta(days=1)  # Move to the next day

@app.route('/submit_details', methods=['POST'])
def submit_details():
    try:
        user_input = {
            'Sex': request.form.get('Sex'),
            'Age': int(request.form.get('Age', 0)) if request.form.get('Age') else 0,
            'Height': int(request.form.get('Height', 0)) if request.form.get('Height') else 0,
            'Weight': int(request.form.get('Weight', 0)) if request.form.get('Weight') else 0,
            'Hypertension': request.form.get('Hypertension'),
            'Diabetes': request.form.get('Diabetes'),
            'BMI': float(request.form.get('BMI', 0.0)) if request.form.get('BMI') else 0.0,
            'Level': request.form.get('Level'),
            'Fitness Goal': request.form.get('Fitness_Goal'),
            'Fitness Type': request.form.get('Fitness_Type'),
            'Diet Type': request.form.get('Diet_Type')
        }

        user_id = session.get('user_id')
        if not user_id:
            flash('User not authenticated. Please log in.', 'error')
            return redirect(url_for('login'))

        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        # Fetch user's name & email
        cursor.execute("SELECT name, email FROM users WHERE id = %s", (user_id,))
        result = cursor.fetchone()
        if not result:
            flash("User not found in database.", "error")
            return redirect(url_for('login'))

        user_name, user_email = result  # Unpack fetched data

        flash('Details updated successfully ✅', 'success')

        recommendations = recommend(user_input)
        weeks_to_goal = recommendations.get('Weeks to Reach Goal', 0)
        session['weeks_to_goal'] = weeks_to_goal
        
        print(f"Weeks to reach goal: {weeks_to_goal}")

        # Process recommended exercises
        recommended_exercises = recommendations.get('Exercises', '')
        if isinstance(recommended_exercises, list):
            recommended_exercises = ", ".join(recommended_exercises).strip()
        recommended_exercises = recommended_exercises.replace(" and ", ", ").replace(",,", ",").strip()

        print(f"Recommended Exercises: {recommended_exercises}")  # Debugging output

        # Fetch additional dietary recommendations
        total_protein_intake = recommendations.get('Total Protein Intake', 0.0)
        bmr = recommendations.get('BMR', 0.0)
        total_calorie_intake = recommendations.get('Total Calorie Intake', 0.0)
        breakfast = recommendations.get('Breakfast', '')
        lunch = recommendations.get('Lunch', '')
        snacks = recommendations.get('Snacks', '')
        dinner = recommendations.get('Dinner', '')

        # Update user details in the database
        query = """
        UPDATE users
        SET Sex = %s, Age = %s, Height = %s, Weight = %s, 
            Hypertension = %s, Diabetes = %s, BMI = %s, Level = %s, 
            Fitness_Goal = %s, Fitness_Type = %s, Diet_Type = %s, weeks = %s, 
            exercises = %s, total_protein_intake = %s, bmr = %s, total_calorie_intake = %s, 
            breakfast = %s, lunch = %s, snacks = %s, dinner = %s
        WHERE id = %s
        """
        cursor.execute(query, (
            user_input['Sex'], user_input['Age'], user_input['Height'], user_input['Weight'],
            user_input['Hypertension'], user_input['Diabetes'], user_input['BMI'], user_input['Level'],
            user_input['Fitness Goal'], user_input['Fitness Type'], user_input['Diet Type'], weeks_to_goal, 
            recommended_exercises, total_protein_intake, bmr, total_calorie_intake, 
            breakfast, lunch, snacks, dinner, user_id
        ))
        conn.commit()

        # Send immediate confirmation email
        subject = "Your Home Workout Plan Starts Tomorrow – Get Ready!"
        body = f"""
        Dear {user_name},

        Your fitness journey with My Fitness Club, Dadar begins tomorrow, and we’re excited to have you on board! 🎉

        ✅ Weeks to Reach Your Goal: {weeks_to_goal}
        🏋️ Fitness Goal: {user_input['Fitness Goal']}
        🔥 Workout Type: {user_input['Fitness Type']} (Home Workouts – No Gym Required!)
        🥗 Diet Plan: {user_input['Diet Type']}
        🏋️ Recommended Exercises: {recommended_exercises}

        Stay consistent, follow the plan, and achieve your transformation! 💪

        Best regards,
        Team My Fitness Club, Dadar
        """
        send_email(user_email, subject, body)

        # Schedule daily reminders
        schedule_daily_emails(user_email, weeks_to_goal)

        return render_template('results.html', recommendations=recommendations)
    
    except mysql.connector.Error as db_err:
        print(f"Database error: {db_err}")
        flash('A database error occurred. Please try again.', 'error')
        return redirect(url_for('details'))
    
    except Exception as e:
        print(f"Error in submit_details: {e}")
        flash('An unexpected error occurred. Please try again.', 'error')
        return redirect(url_for('details'))
    
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()
def create_tables():
    try:
        # Connect to the database
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        # Check if 'users' table exists
        cursor.execute("SHOW TABLES LIKE 'users'")
        if cursor.fetchone():
            print("Table 'users' already exists.")
        else:
            cursor.execute("""
                CREATE TABLE users (
                    id INT(11) NOT NULL AUTO_INCREMENT,
                    name VARCHAR(100) NOT NULL,
                    email VARCHAR(100) NOT NULL,
                    phone VARCHAR(15),
                    username VARCHAR(50) NOT NULL,
                    password VARCHAR(255) NOT NULL,
                    Sex VARCHAR(10),
                    Age INT(11),
                    Height INT(11),
                    Weight INT(11),
                    Hypertension VARCHAR(5),
                    Diabetes VARCHAR(5),
                    BMI FLOAT,
                    Level VARCHAR(20),
                    Fitness_Goal VARCHAR(50),
                    Fitness_Type VARCHAR(50),
                    Diet_Type VARCHAR(50),
                    weeks DECIMAL(20),
                    PRIMARY KEY (id)
                )
            """)
            print("Table 'users' created successfully.")

        # Check if 'user_progress' table exists
        cursor.execute("SHOW TABLES LIKE 'user_progress'")
        if cursor.fetchone():
            print("Table 'user_progress' already exists.")
        else:
            cursor.execute("""
                CREATE TABLE user_progress (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NOT NULL,
                    week INT NOT NULL,
                    day INT NOT NULL,
                    task VARCHAR(255) NOT NULL,
                    completed TINYINT(1) NOT NULL DEFAULT 0,
                    completed_at DATETIME NULL,
                    UNIQUE KEY unique_progress (user_id, week, day, task),
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                )
            """)
            print("Table 'user_progress' created successfully.")

    except mysql.connector.Error as err:
        print(f"Error: {err}")

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

@app.route('/todo')
@login_required
def todo():
    try:
        user_id = session.get('user_id')  # Adjust as per your session setup

        if not user_id:
            flash('User not authenticated. Please log in.', 'error')
            return redirect(url_for('login'))  # Redirect to login if User_ID is not found

        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        # Fetch weeks_to_goal from the database
        query = "SELECT weeks FROM users WHERE id = %s"
        cursor.execute(query, (user_id,))
        user = cursor.fetchone()

        if user and user[0] is not None:
            weeks_to_goal = user[0]  # Access the first element
            if isinstance(weeks_to_goal, decimal.Decimal):
                weeks_to_goal = int(weeks_to_goal)
            print(f"Weeks to reach goal: {weeks_to_goal:.2f}")
            cursor.close()
            conn.close()
            return render_template('todo.html', weeks_to_goal=weeks_to_goal)
        else:
            cursor.close()
            conn.close()
            flash("No data found for the user", 'warning')
            return render_template('notodo.html')  # Render a different page if no data

    except Exception as e:
        print(f"Error in todo: {e}")
        return render_template('error.html', error='An error occurred while processing your request.')

from datetime import datetime
@app.route('/save_progress', methods=['POST'])
def save_progress():
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.json
    user_id = session['user_id']
    today_date = datetime.now().date()

    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()

    # Ensure table exists
    create_table_query = """
    CREATE TABLE IF NOT EXISTS user_progress (
        id INT AUTO_INCREMENT PRIMARY KEY,
        user_id INT NOT NULL,
        week INT NOT NULL,
        day INT NOT NULL,
        task VARCHAR(255) NOT NULL,
        completed TINYINT(1) NOT NULL DEFAULT 0,
        completed_at DATE NULL,
        UNIQUE KEY unique_progress (user_id, week, day, task)
    )
    """
    cursor.execute(create_table_query)

    # Insert or update task progress
    query = """INSERT INTO user_progress (user_id, week, day, task, completed)
               VALUES (%s, %s, %s, %s, %s)
               ON DUPLICATE KEY UPDATE completed = VALUES(completed)"""
    cursor.execute(query, (user_id, data['week'], data['day'], data['task'], data['completed']))

    # Check if all tasks for the current day are completed
    check_query = """SELECT COUNT(*) FROM user_progress 
                     WHERE user_id = %s AND week = %s AND day = %s AND completed = 0"""
    cursor.execute(check_query, (user_id, data['week'], data['day']))
    remaining_tasks = cursor.fetchone()[0]

    if remaining_tasks == 0:
        # Mark this day as fully completed
        update_query = """UPDATE user_progress 
                          SET completed_at = %s 
                          WHERE user_id = %s AND week = %s AND day = %s"""
        cursor.execute(update_query, (today_date, user_id, data['week'], data['day']))

    conn.commit()
    cursor.close()
    conn.close()
    
    return jsonify({"message": "Progress saved successfully"})

@app.route('/reset_progress', methods=['POST'])
def reset_progress():
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    user_id = session['user_id']
    
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()
    
    cursor.execute("DELETE FROM user_progress WHERE user_id = %s", (user_id,))
    conn.commit()

    cursor.close()
    conn.close()

    return jsonify({"message": "Progress reset successfully", "progress": 0})


from flask import jsonify, session
import mysql.connector
@app.route('/get_progress', methods=['GET'])
def get_progress():
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    user_id = session['user_id']
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor(dictionary=True)

    try:
        cursor.execute("SELECT week, day, task, completed FROM user_progress WHERE user_id = %s", (user_id,))
        progress = cursor.fetchall()

        cursor.execute("""
            SELECT MAX(day) AS last_completed_day FROM user_progress 
            WHERE user_id = %s AND completed = 1
        """, (user_id,))
        last_completed_day = cursor.fetchone()['last_completed_day'] or 0

        cursor.execute("""
            SELECT COUNT(*) AS incomplete_tasks FROM user_progress 
            WHERE user_id = %s AND day = %s AND completed = 0
        """, (user_id, last_completed_day))
        incomplete_tasks = cursor.fetchone()['incomplete_tasks']

        unlock_next_day = last_completed_day + 1 if incomplete_tasks == 0 else last_completed_day

        return jsonify({"progress": progress, "unlocked_day": unlock_next_day})

    finally:
        cursor.close()
        conn.close()

@app.route('/workout')
def workout():
    return render_template('workout.html')


@app.route('/PushUp')
def pushup():
    return render_template('PushUp.html')

@app.route('/Pressflat')
def dumbbell_press_flat():
    return render_template('DumbbellPressFlat.html')

@app.route('/PressIncline')
def dumbbell_press_incline():
    return render_template('DumbbellPressincline.html')

@app.route('/Abs')
def abs_workout():
    return render_template('abs.html')

@app.route('/Shoulder')
def shoulder_press():
    return render_template('ShoulderPress.html')

@app.route('/Bicep')
def bicep_curl():
    return render_template('BicepCurl.html')

@app.route('/Deadlift')
def deadlift():
    return render_template('DumbellDeadlifts.html')
import cv2
import numpy as np
import mediapipe as mp
import time
import json

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

cap = None
count = 0
direction = 0
form_status = "Waiting"  # Track form quality

def calculate_angle(a, b, c):
    """Calculate the angle between three points."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return angle

def detect_pose(frame, exercise_type):
    """Detects pose, displays angles, and tracks exercise repetitions with form feedback."""
    global count, direction, form_status

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Extract key points
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

        # Calculate angles
        left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_arm_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
        left_leg_angle = calculate_angle(left_hip, left_knee, left_ankle)
        right_leg_angle = calculate_angle(right_hip, right_knee, right_ankle)
        
        # Calculate arm symmetry for form checking
        arm_symmetry = abs(left_arm_angle - right_arm_angle)
        
        # Calculate trunk angle (for deadlift)
        trunk_angle = calculate_angle(
            [(left_shoulder[0] + right_shoulder[0])/2, (left_shoulder[1] + right_shoulder[1])/2],
            [(left_hip[0] + right_hip[0])/2, (left_hip[1] + right_hip[1])/2],
            [(left_knee[0] + right_knee[0])/2, (left_knee[1] + right_knee[1])/2]
        )

        # Initialize form_status for this frame
        form_status = "Correct Form"
        form_color = (0, 255, 0)  # Green for correct form

        # Exercise logic with form checking
        if exercise_type == "shoulder_press":
            # Check symmetry of arms
            if arm_symmetry > 15:
                form_status = "Fix: Uneven Arms"
                form_color = (0, 0, 255)  # Red for incorrect form
            
            # Check full extension at top
            if direction == 1 and (left_arm_angle < 160 or right_arm_angle < 160):
                form_status = "Fix: Extend Arms Fully"
                form_color = (0, 0, 255)
            
            # Rep counting logic
            if left_arm_angle > 160 and right_arm_angle > 160:
                direction = 1
            if direction == 1 and left_arm_angle < 100 and right_arm_angle < 100:
                count += 1
                direction = 0

        elif exercise_type == "bicep_curl":
            # Check arm positioning
            if arm_symmetry > 15:
                form_status = "Fix: Uneven Curl"
                form_color = (0, 0, 255)
            
            if left_arm_angle > 140 and right_arm_angle > 140:
                direction = 1
            if direction == 1 and left_arm_angle < 50 and right_arm_angle < 50:
                count += 1
                direction = 0

        elif exercise_type == "bench_abs":
            # Check for proper leg movement
            if left_leg_angle < 60 or right_leg_angle < 60:
                form_status = "Fix: Don't Go Too Low"
                form_color = (0, 0, 255)
                
            if left_leg_angle > 120:
                direction = 1
            if direction == 1 and left_leg_angle < 90:
                count += 1
                direction = 0

        elif exercise_type == "pushups":
            # Check elbow alignment
            if arm_symmetry > 15:
                form_status = "Fix: Uneven Pushup"
                form_color = (0, 0, 255)
                
            if left_arm_angle > 160 and right_arm_angle > 160:
                direction = 1
            if direction == 1 and left_arm_angle < 90 and right_arm_angle < 90:
                count += 1
                direction = 0

        elif exercise_type == "dumbbell_deadlift":
            # Check back position
            if trunk_angle < 140 and direction == 0:
                form_status = "Fix: Keep Back Straighter"
                form_color = (0, 0, 255)
                
            if trunk_angle > 160 and left_leg_angle > 160 and right_leg_angle > 160:
                direction = 1
            if direction == 1 and trunk_angle < 120 and left_leg_angle < 120 and right_leg_angle < 120:
                count += 1
                direction = 0

        elif exercise_type == "dumbbell_press_incline":
            if arm_symmetry > 15:
                form_status = "Fix: Uneven Press"
                form_color = (0, 0, 255)
                
            if left_arm_angle > 160 and right_arm_angle > 160:
                direction = 1
            if direction == 1 and left_arm_angle < 80 and right_arm_angle < 80:
                count += 1
                direction = 0

        elif exercise_type == "dumbbell_press_flat":
            if arm_symmetry > 15:
                form_status = "Fix: Uneven Press"
                form_color = (0, 0, 255)
                
            if left_arm_angle > 160 and right_arm_angle > 160:
                direction = 1
            if direction == 1 and left_arm_angle < 90 and right_arm_angle < 90:
                count += 1
                direction = 0

        # Draw landmarks
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Display angles
        cv2.putText(frame, f"Left Arm: {int(left_arm_angle)}°", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Right Arm: {int(right_arm_angle)}°", (50, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Left Leg: {int(left_leg_angle)}°", (50, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Right Leg: {int(right_leg_angle)}°", (50, 140), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if exercise_type == "dumbbell_deadlift":
            cv2.putText(frame, f"Trunk Angle: {int(trunk_angle)}°", (50, 170),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display rep count and form status
        cv2.putText(frame, f"Reps: {count}/15", (50, 200), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, form_status, (50, 230),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, form_color, 2)

    return frame

def generate_frames(exercise_type):
    """Generates frames for the video feed with a 40-second blank screen before tracking starts."""
    global cap, count

    cap = cv2.VideoCapture(0)
    count = 0

    # Show 40-second blank screen
    start_time = time.time()
    while time.time() - start_time < 42:
        black_screen = np.zeros((480, 640, 3), dtype=np.uint8)
        remaining_time = 42 - (time.time() - start_time)
        cv2.putText(black_screen, f"Starting in {int(remaining_time)} seconds...", (160, 240), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        _, buffer = cv2.imencode('.jpg', black_screen)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
               
    # Begin capture and processing after countdown
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            processed_frame = detect_pose(frame, exercise_type)
            _, buffer = cv2.imencode('.jpg', processed_frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

def get_exercise_data():
    """Return the current exercise data for frontend display."""
    global count, form_status
    
    is_correct = form_status == "Correct Form"
    
    # Create a JSON-compatible response with updated exercise data
    response = {
        "count": count,
        "is_correct_form": is_correct,
        "feedback_message": form_status
    }
    
    return response  # This will be converted to JSON by your Flask route

@app.route('/get_exercise_data')
def exercise_data():
    data = get_exercise_data()
    return jsonify(data)  # Properly convert the data to JSON

# Function to release the camera when needed
def release_camera():
    global cap
    if cap is not None:
        cap.release()
    return frame

def generate_frames(exercise_type):
    """Generates frames for the video feed with a 40-second blank screen before tracking starts."""
    global cap, count

    cap = cv2.VideoCapture(0)
    count = 0

    # Show 40-second blank screen
    start_time = time.time()
    while time.time() - start_time < 42:
        black_screen = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(black_screen, "Starting soon...", (200, 250), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        _, buffer = cv2.imencode('.jpg', black_screen)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    # Start exercise tracking
    while cap.isOpened():
        success, frame = cap.read()
        if not success or count >= 15:
            break

        frame = detect_pose(frame, exercise_type)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

    # Show black screen after completion
    black_screen = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(black_screen, "Workout Complete!", (150, 250), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    _, buffer = cv2.imencode('.jpg', black_screen)
    yield (b'--frame\r\n'
           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
@app.route('/video_feed/shoulder_press')
def video_feed_shoulder_press():
    return Response(generate_frames("shoulder_press"), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed/bicep_curl')
def video_feed_bicep_curl():
    return Response(generate_frames("bicep_curl"), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed/bench_abs')
def video_feed_bench_abs():
    return Response(generate_frames("bench_abs"), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed/dumbbell_press_flat')
def video_feed_dumbbell_press_flat():
    return Response(generate_frames("dumbbell_press_flat"), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed/dumbbell_press_incline')
def video_feed_dumbbell_press_incline():
    return Response(generate_frames("dumbbell_press_incline"), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed/dumbbell_deadlift')
def video_feed_dumbbell_deadlift():
    return Response(generate_frames("dumbbell_deadlift"), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed/pushups')
def video_feed_pushups():
    return Response(generate_frames("pushups"), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    create_tables()
    app.run(debug=True)


