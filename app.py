import gradio as gr
import joblib
import pandas as pd

model = joblib.load('models/best_model_gradient_boosting.pkl')

def get_choices_mapping():
    '''
    Iterate through all the columns and get their name -> categories mapping
    Need to remove column name prefix due to how one_hot_encoder stores them
    '''
    choices_mapping = {}

    for mapping in model[0][0]['one_hot_encoder'].category_mapping:
        col_name = mapping['col']
        choices_mapping[col_name] = mapping['mapping'].columns.map(lambda x: x[len(col_name)+1:]).tolist()

    return choices_mapping

def predict(norm_title, location_state, company_industries, experience_level, work_type, remote_allowed, company_employee_count):
    input_data = pd.DataFrame({
        'norm_title': [norm_title],
        'location_state': [location_state],
        'company_industries': [company_industries],
        'formatted_experience_level': [experience_level],
        'formatted_work_type': [work_type],
        'remote_allowed': [remote_allowed],
        'company_employee_count': [company_employee_count],
        'clustered_edu_req': '',
        'clustered_pref_qual': '',
        'clustered_req_skill': '',
    })
    
    prediction = model.predict(input_data)
    return f"Predicted Salary: ${prediction[0]:,.2f}"

choices_mapping = get_choices_mapping()
interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Dropdown(choices=choices_mapping['norm_title'], label="Norm Title"),
        gr.Dropdown(choices=choices_mapping['location_state'], label="Location (State)"),
        gr.Dropdown(choices=choices_mapping['company_industries'], label="Company Industry"),
        gr.Dropdown(choices=choices_mapping['formatted_experience_level'], label="Experience Level"),
        gr.Dropdown(choices=choices_mapping['formatted_work_type'], label="Work Type"),
        gr.Radio(choices=[0, 1], label="Remote Allowed"),
        gr.Number(label="Company Employee Count")
    ],
    outputs="text",
    title="US Job Salary Prediction (April 2024)",
    description="Enter job details to estimate the salary you should post.",
    theme="ocean" 
)

if __name__ == "__main__":
    interface.launch()