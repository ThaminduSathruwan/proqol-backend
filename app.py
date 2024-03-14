import pickle
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)

def predict(data):
    model_cs_rf = pickle.load(open('rf_model1.pkl', 'rb'))
    model_bo_rf = pickle.load(open('rf_model2.pkl', 'rb'))
    model_sts_rf = pickle.load(open('rf_model3.pkl', 'rb'))
    one_hot_encoder = pickle.load(open('one_hot_encoder.pkl', 'rb'))

    categorical_cols = ['Hospital', 'Main Unit']
    ordinal_cols = ['Daily travelling Distance', 'Hours of Sleeping', 'Work Experience in this Hospital', 'Total Working Hours in a Week', 'Relationship with the Superiors', 'Frequency of involving patient emergencies']     

    ordinal_variables_order = {
        'Daily travelling Distance': ['Less than 1 km', '1 - 5 km', '5 - 10 km','More than 10 km'],
        'Hours of Sleeping': ['Less than 3 hours','3 - 5 hours','5 - 7 hours','More than 7 hours'],
        'Work Experience in this Hospital': ['Less than 5 years','5 - 10 years','10 - 20 years','More than 20 years'],
        'Total Working Hours in a Week': ['<42 ','42 - 63','63 - 84','>84 '],
        'Relationship with the Superiors': ['Very low','Low','Average','High','Very high'],
        'Frequency of involving patient emergencies': ['Very low','Low','Average','High','Very high'],
    }
    
    encoded_data = one_hot_encoder.transform(data[categorical_cols]).toarray()
    
    for col in ordinal_cols:
        # Map the values to their corresponding indices
        data[col] = data[col].map({val: idx for idx, val in enumerate(ordinal_variables_order[col])})
            
    # drop categorical columns from data and concatenate with encoded_data
    data = data.drop(categorical_cols, axis=1)
    processed_data = np.concatenate((data, encoded_data), axis=1)
    
    processed_df = pd.DataFrame(processed_data, columns=list(data.columns) + list(one_hot_encoder.get_feature_names_out(categorical_cols)))

    # Make predictions
    predictions_cs = model_cs_rf.predict(processed_df)
    predictions_bo = model_bo_rf.predict(processed_df)
    predictions_sts = model_sts_rf.predict(processed_df)

    # return predictions
    return jsonify({
        'CS': int(predictions_cs[0]),
        'BO': int(predictions_bo[0]),
        'STS': int(predictions_sts[0])
    })

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predictPage():
    try:
        if request.method == 'POST':
            input_data = {
                'Age': int(request.form['Age']),
                'Daily travelling Distance': request.form['Daily travelling Distance'],
                'Hours of Sleeping': request.form['Hours of Sleeping'],
                'Work Experience in this Hospital': request.form['Work Experience in this Hospital'],
                'Total Working Hours in a Week': request.form['Total Working Hours in a Week'],
                'Relationship with the Superiors': request.form['Relationship with the Superiors'],
                'Frequency of involving patient emergencies': request.form['Frequency of involving patient emergencies'],
                'BMI': float(request.form['BMI']),
                'P/N Ratio': int(request.form['P/N Ratio']),
                'Hospital': request.form['Hospital'],
                'Main Unit': request.form['Main Unit']
            }
            df = pd.DataFrame([input_data])
            return predict(df)
    except Exception as e:
        print(e)
        error_msg = 'Something went wrong'
        return jsonify({"error": error_msg})
