from flask import Flask, render_template, request
import torch
from torchvision import models, transforms
from PIL import Image
import joblib
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

# Load the Heart Disease Model
with open('models/heart_disease.pkl', 'rb') as f:
    heart_disease_model = pickle.load(f)

# Load the Diabetes Model
diabetes_model = joblib.load('models/diabetes_model.pkl')

# Define expected columns for diabetes model
expected_columns = [
    'age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level',
    'gender_Male', 'gender_Other', 'smoking_history_current', 'smoking_history_ever',
    'smoking_history_former', 'smoking_history_never', 'smoking_history_not current'
]

# Define the AlexNet model class for lung cancer prediction
class AlexNet(torch.nn.Module):
    def __init__(self, num_classes=3):
        super(AlexNet, self).__init__()
        self.model = models.alexnet(pretrained=True)
        self.model.classifier[6] = torch.nn.Linear(self.model.classifier[6].in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Load the Lung Cancer Model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
lung_cancer_model = AlexNet(num_classes=3).to(device)
lung_cancer_model.load_state_dict(torch.load('models/alexnet_lung_cancer.pth'))
lung_cancer_model.eval()

# Define transformations for lung cancer model
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Landing Page
@app.route('/')
def index():
    return render_template('index.html')

# Heart Disease Prediction Page
@app.route('/heart_disease', methods=['GET', 'POST'])
def heart_disease():
    if request.method == 'POST':
        # Get user input from the form
        user_input = np.array([[  
            int(request.form['age']), int(request.form['sex']), int(request.form['cp']),
            int(request.form['trestbps']), int(request.form['chol']), int(request.form['fbs']),
            int(request.form['restecg']), int(request.form['thalach']), int(request.form['exang']),
            float(request.form['oldpeak']), int(request.form['slope']), int(request.form['ca']),
            int(request.form['thal'])
        ]])

        # Make a prediction
        prediction = heart_disease_model.predict(user_input)

        # Prepare the result
        result = "Heart Disease Detected" if prediction[0] == 1 else "No Heart Disease"

        return render_template('heart_disease.html', result=result)

    return render_template('heart_disease.html')

# Lung Cancer Prediction Page
@app.route('/lung_cancer', methods=['GET', 'POST'])
def lung_cancer():
    if request.method == 'POST':
        # Get the uploaded image
        file = request.files['image']
        image = Image.open(file.stream).convert('RGB')

        # Preprocess the image
        input_tensor = data_transforms(image).unsqueeze(0).to(device)

        # Make a prediction
        with torch.no_grad():
            output = lung_cancer_model(input_tensor)
            _, predicted_class = torch.max(output, 1)

        # Map the predicted class index to the class name
        class_names = ['Beginning', 'Malignant', 'Normal']
        predicted_class_name = class_names[predicted_class.item()]

        return render_template('lung_cancer.html', prediction=predicted_class_name, image_url=file.filename)

    return render_template('lung_cancer.html')

# Diabetes Prediction Page
@app.route('/diabetes', methods=['GET', 'POST'])
def diabetes():
    if request.method == 'POST':
        # Get user input from the form
        user_data = {
            'gender': request.form['gender'],
            'age': float(request.form['age']),
            'hypertension': int(request.form['hypertension']),
            'heart_disease': int(request.form['heart_disease']),
            'smoking_history': request.form['smoking_history'],
            'bmi': float(request.form['bmi']),
            'HbA1c_level': float(request.form['hba1c_level']),
            'blood_glucose_level': int(request.form['blood_glucose_level'])
        }

        # Preprocess the user input
        user_df = pd.DataFrame([user_data])
        user_df = pd.get_dummies(user_df, columns=['gender', 'smoking_history'], drop_first=True)
        for col in expected_columns:
            if col not in user_df.columns:
                user_df[col] = 0
        user_df = user_df[expected_columns]

        # Make a prediction
        prediction = diabetes_model.predict(user_df)
        prediction_proba = diabetes_model.predict_proba(user_df)

        # Prepare the result message
        result = "At risk of diabetes." if prediction[0] == 1 else "Not at risk of diabetes."
        probability = f"{prediction_proba[0][1] * 100:.2f}%"

        return render_template('diabetes.html', result=result, probability=probability)

    return render_template('diabetes.html')


liver_disease_model = joblib.load('models/liver_disease_model.pkl')

@app.route('/liver_disease', methods=['GET', 'POST'])
def liver_disease():
    if request.method == 'POST':
        try:
            # Get user input from the form
            age = float(request.form['age'])
            gender = request.form['gender'].strip().lower()
            total_bilirubin = float(request.form['total_bilirubin'])
            direct_bilirubin = float(request.form['direct_bilirubin'])
            alkphos = float(request.form['alkphos'])
            sgpt = float(request.form['sgpt'])
            sgot = float(request.form['sgot'])
            total_proteins = float(request.form['total_proteins'])
            albumin = float(request.form['albumin'])
            ag_ratio = float(request.form['ag_ratio'])

            # Encode Gender (Assuming Male=1, Female=0 as used in training)
            gender_encoded = 1 if gender == "male" else 0

            # Create input array (same order as training data)
            input_data = np.array([[age, gender_encoded, total_bilirubin, direct_bilirubin,
                                    alkphos, sgpt, sgot, total_proteins, albumin, ag_ratio]])

            # Predict
            prediction = liver_disease_model.predict(input_data)

            # Prepare result message
            result = "Liver Disease Detected" if prediction[0] == 1 else "No Liver Disease"

            return render_template('liver_disease.html', result=result)

        except ValueError:
            return render_template('liver_disease.html', result="Invalid input! Please enter valid numeric values.")

    return render_template('liver_disease.html')


if __name__ == '__main__':
    app.run(debug=True)
