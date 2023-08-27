# Import libraries
import pandas as pd
import streamlit as st
from joblib import load

# Load dataset
merged_data = pd.read_csv("merged_data.csv")

# Load the pipeline using joblib 
# Load the pipeline (assuming it's stored as a tuple: (transformer, regressor))
with open("sale_pipeline.joblib", 'rb') as file:
    transformer, regressor = joblib.load(file)

# Create the web app
def main():
    st.title('Sales Prediction')
    st.write('Enter the details to predict sales.')

    # Collect user inputs
    country = st.selectbox('Country', sorted(merged_data['Country'].unique()))
    state = st.selectbox('State', sorted(merged_data['State'].unique()))
    city = st.selectbox('City', sorted(merged_data['City'].unique()))
    region = st.selectbox('Region', sorted(merged_data['Region'].unique()))
    segment = st.selectbox('Segment', sorted(merged_data['Segment'].unique()))
    shipmode = st.selectbox('Ship Mode', sorted(merged_data['Ship Mode'].unique()))
    actual_discount = st.number_input('Actual Discount')
    quantity = st.number_input('Quantity', min_value=1, step=1)
    category = st.selectbox('Category', sorted(merged_data['Category'].unique()))
    subcategory = st.selectbox('Sub-Category', sorted(merged_data['Sub-Category'].unique()))
    product_name = st.selectbox('Product Name', sorted(merged_data['Product Name'].unique()))

    data = pd.DataFrame({
        'Country': [country],
        'State': [state],
        'City': [city],
        'Region': [region],
        'Segment': [segment],
        'Ship Mode': [shipmode],
        'Actual Discount': [actual_discount],
        'Quantity': [quantity],
        'Category': [category],
        'Sub-Category': [subcategory],
        'Product Name': [product_name]
    })

    if st.button('Predict Sales'):
        try:
            # Preprocess input data
            data_encoded = transformer.transform(data)
            
            # Make predictions using the regressor
            prediction = regressor.predict(data_encoded)
            st.success(f'Predicted Sales: {prediction[0]:.2f}')
        except Exception as e:
            st.error(f"An error occurred: {e}")

    # Add contact details
    st.write("For any query, reach out to:")
    st.write("**Avani Shah**")
    st.write("Email: [avanishah128@gmail.com]")

# Run the web app
if __name__ == '__main__':
    main()
