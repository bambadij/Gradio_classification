import gradio as gr
import pandas as pd
import numpy as np
import joblib


#Load scaler, imputer, model
num_imputer = joblib.load(r'toolkit\numerical_imputer.joblib')
cat_imputer = joblib.load(r'toolkit\categorical_imputer.joblib')
encoder = joblib.load(r'toolkit\encoder.joblib')
scaler = joblib.load(r'toolkit\scaler.joblib')
model = joblib.load(r'toolkit\Final_model.joblib')


# Create a function that applies the ML pipeline and makes predictions
def predict(gender,SeniorCitizen,Partner,Dependents, tenure, PhoneService, MultipleLines,
            InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,
            Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges):



    # Create a dataframe with the input data
     input_df = pd.DataFrame({
        'gender': [gender],
        'SeniorCitizen': [SeniorCitizen],
        'Partner': [Partner],
        'Dependents': [Dependents],
        'tenure': [tenure],
        'PhoneService': [PhoneService],
        'MultipleLines': [MultipleLines],
        'InternetService': [InternetService],
        'OnlineSecurity': [OnlineSecurity],
        'OnlineBackup': [OnlineBackup],
        'DeviceProtection': [DeviceProtection],
        'TechSupport': [TechSupport],
        'StreamingTV': [StreamingTV],
        'StreamingMovies': [StreamingMovies],
        'Contract': [Contract],
        'PaperlessBilling': [PaperlessBilling],
        'PaymentMethod': [PaymentMethod],
        'MonthlyCharges': [MonthlyCharges],
        'TotalCharges': [TotalCharges]

 })

# Selecting categorical and numerical columns separately
     cat_columns = [col for col in input_df.columns if input_df[col].dtype == 'object']
     num_columns = [col for col in input_df.columns if input_df[col].dtype != 'object']

    # Apply the imputers on the input data
     input_df_imputed_cat = cat_imputer.transform(input_df[cat_columns])
     input_df_imputed_num = num_imputer.transform(input_df[num_columns])

    # Encode the categorical columns
     input_encoded_df = pd.DataFrame(encoder.transform(input_df_imputed_cat).toarray(),
                                   columns=encoder.get_feature_names_out(cat_columns))

    # Scale the numerical columns
     input_df_scaled = scaler.transform(input_df_imputed_num)
     input_scaled_df = pd.DataFrame(input_df_scaled , columns = num_columns)


    #joining the cat encoded and num scaled
     final_df = pd.concat([input_scaled_df, input_encoded_df], axis=1)

     final_df = final_df.reindex(columns=['tenure', 'MonthlyCharges', 'TotalCharges', 'gender_Female',
       'gender_Male', 'SeniorCitizen_0', 'SeniorCitizen_1', 'Partner_No',
       'Partner_Yes', 'Dependents_No', 'Dependents_Yes', 'PhoneService_No',
       'PhoneService_Yes', 'MultipleLines_No',
       'MultipleLines_No phone service', 'MultipleLines_Yes',
       'InternetService_DSL', 'InternetService_Fiber optic',
       'InternetService_No', 'OnlineSecurity_No',
       'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
       'OnlineBackup_No', 'OnlineBackup_No internet service',
       'OnlineBackup_Yes', 'DeviceProtection_No',
       'DeviceProtection_No internet service', 'DeviceProtection_Yes',
       'TechSupport_No', 'TechSupport_No internet service', 'TechSupport_Yes',
       'StreamingTV_No', 'StreamingTV_No internet service', 'StreamingTV_Yes',
       'StreamingMovies_No', 'StreamingMovies_No internet service',
       'StreamingMovies_Yes', 'Contract_Month-to-month', 'Contract_One year',
       'Contract_Two year', 'PaperlessBilling_No', 'PaperlessBilling_Yes',
       'PaymentMethod_Bank transfer (automatic)',
       'PaymentMethod_Credit card (automatic)',
       'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'])

    # Make predictions using the model
     predictions = model.predict(final_df)

     # prediction_label = "This customer is likely to Churn" if predictions.item() == "Yes" else "This customer is Not likely churn"
     prediction_label = {"Prediction: CHURN": float(predictions[0]), "Prediction: STAY": 1-float(predictions[0])}


     return prediction_label


input_interface = []

with gr.Blocks(theme=gr.themes.Soft()) as app:

    img = gr.Image("customer churn.png")

    Title = gr.Label('Predicting Customer Churn App')

    with gr.Row():
        Title

    with gr.Row():
        img

    with gr.Row():
        gr.Markdown("This app predicts likelihood of a customer to churn or stay with the company")

    with gr.Accordion("Open for information on inputs"):
        gr.Markdown("""This app receives the following as inputs and processes them to return the prediction on whether a customer, given the inputs, will churn or not.
                    - Contract: The contract term of the customer (Month-to-Month, One year, Two year)
                    - Dependents: Whether the customer has dependents or not (Yes, No)
                    - DeviceProtection: Whether the customer has device protection or not (Yes, No, No internet service)
                    - Gender: Whether the customer is a male or a female
                    - InternetService: Customer's internet service provider (DSL, Fiber Optic, No)
                    - MonthlyCharges: The amount charged to the customer monthly
                    - MultipleLines: Whether the customer has multiple lines or not
                    - OnlineBackup: Whether the customer has online backup or not (Yes, No, No Internet)
                    - OnlineSecurity: Whether the customer has online security or not (Yes, No, No Internet)
                    - PaperlessBilling: Whether the customer has paperless billing or not (Yes, No)
                    - Partner: Whether the customer has a partner or not (Yes, No)
                    - Payment Method: The customer's payment method (Electronic check, mailed check, Bank transfer(automatic), Credit card(automatic))
                    - Phone Service: Whether the customer has a phone service or not (Yes, No)
                    - SeniorCitizen: Whether a customer is a senior citizen or not
                    - StreamingMovies: Whether the customer has streaming movies or not (Yes, No, No Internet service)
                    - StreamingTV: Whether the customer has streaming TV or not (Yes, No, No internet service)
                    - TechSupport: Whether the customer has tech support or not (Yes, No, No internet)
                    - Tenure: Number of months the customer has stayed with the company
                    - TotalCharges: The total amount charged to the customer
                    """)

    with gr.Row():
        with gr.Column():
            input_interface_column_1 = [
                gr.components.Radio(['male', 'female'], label='Select your gender'),
                gr.components.Dropdown(['1', '0'], label="Are you a Seniorcitizen; No=0 and Yes=1"),
                gr.components.Radio(['Yes', 'No'], label='Do you have Partner'),
                gr.components.Dropdown(['No', 'Yes'], label='Do you have any Dependents?'),
                gr.components.Number(label='Lenght of tenure (no. of months with Telco)', minimum=0, maximum=73),
                gr.components.Radio(['No', 'Yes'], label='Do you have PhoneService? '),
                gr.components.Radio(['No', 'Yes'], label='Do you have MultipleLines'),
                gr.components.Radio(['DSL', 'Fiber optic', 'No'], label='Do you have InternetService'),
                gr.components.Radio(['No', 'Yes'], label='Do you have OnlineSecurity?'),
                gr.components.Radio(['No', 'Yes'], label='Do you have OnlineBackup?')
            ]

        with gr.Column():
            input_interface_column_2 = [
                gr.components.Radio(['No', 'Yes'], label='Do you have DeviceProtection?'),
                gr.components.Radio(['No', 'Yes'], label='Do you have TechSupport?'),
                gr.components.Radio(['No', 'Yes'], label='Do you have StreamingTV?'),
                gr.components.Radio(['No', 'Yes'], label='Do you have StreamingMovies?'),
                gr.components.Dropdown(['Month-to-month', 'One year', 'Two year'], label='which Contract do you use?'),
                gr.components.Radio(['Yes', 'No'], label='Do you prefer PaperlessBilling?'),
                gr.components.Dropdown(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], label='Which PaymentMethod do you prefer?'),
                gr.components.Slider(label="Enter monthly charges"),
                gr.components.Slider(label="Enter total charges", maximum=10000)
            ]

    with gr.Row():
        input_interface.extend(input_interface_column_1)
        input_interface.extend(input_interface_column_2)

    with gr.Row():
        predict_btn = gr.Button('Predict')

# Define the output interfaces
    output_interface = gr.Label(label="churn")

    predict_btn.click(fn=predict, inputs=input_interface, outputs=output_interface)

app.launch(share=True)