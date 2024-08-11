# import required libraries
import pandas as pd
import numpy as np
import streamlit as st
import sklearn
import pickle
from datetime import datetime, timedelta, time

# load the encoder and trained model
with open('encoder/airline_encoder', 'rb') as file:
    airline_encoder = pickle.load(file)
with open('encoder/arr_time_encoder', 'rb') as file:
    arr_time_encoder = pickle.load(file)
with open('encoder/dept_time_encoder', 'rb') as file:
    dept_time_encoder = pickle.load(file)
with open('encoder/dest_city_encoder', 'rb') as file:
    dest_city_encoder = pickle.load(file)
with open('encoder/src_city_encoder', 'rb') as file:
    src_city_encoder = pickle.load(file)
with open('encoder/ord_encoder', 'rb') as file:
    ord_encoder = pickle.load(file)
with open('encoder/label_encoder', 'rb') as file:
    label_encoder = pickle.load(file)

with open('model/best_regressor', 'rb') as file:
    best_regressor = pickle.load(file)

# load the dataset and define the variables for options
data = pd.read_csv('data/Clean_Dataset.csv')

airline = data.airline.unique()
flight_class = data['class'].unique()
source_city = data['source_city'].unique()
destination_city = data['destination_city'].unique()
departure_time = data['departure_time'].unique()
arrival_time = data['arrival_time'].unique()
stops = [0, 1, '>2']

# helper function to process user input
def categorize_time(time_input):
    # Define time ranges for each category
    early_morning_start = time(0, 0)
    early_morning_end = time(6, 0)
    morning_start = time(6, 0)
    morning_end = time(11, 59)
    afternoon_start = time(12, 0)
    afternoon_end = time(17, 59)
    evening_start = time(18, 0)
    evening_end = time(20, 59)
    night_start = time(21, 0)
    night_end = time(22, 59)
    late_night_start = time(23, 0)
    late_night_end = time(23, 59)

    # check if the input is a string or already a time object
    if isinstance(time_input, str):
        # Convert input time string to a datetime object and then to a time object
        input_time = datetime.strptime(time_input, "%H:%M").time()
    elif isinstance(time_input, time):
        # If it's already a time object, use it directly
        input_time = time_input
    else:
        raise ValueError("Input must be a string or a datetime.time object")

    # determine the category based on time ranges
    if early_morning_start <= input_time <= early_morning_end:
        return "Early_Morning"
    elif morning_start <= input_time <= morning_end:
        return "Morning"
    elif afternoon_start <= input_time <= afternoon_end:
        return "Afternoon"
    elif evening_start <= input_time <= evening_end:
        return "Evening"
    elif night_start <= input_time <= night_end:
        return "Night"
    elif late_night_start <= input_time <= late_night_end:
        return "Late_Night"
    else:
        return "Unknown"
    
def convertstops(stp):
    if stp == 0:
        return "zero"
    elif stp == 1:
        return "one"
    else:
        return "two_or_more"

# Streamlit UI --- form to collect user input
with st.form('prediction'):
    st.title('Flight Ticket Prediction')

    airline = st.selectbox('Select Airline', airline)
    flight_class = st.selectbox('Select Flight Class:', flight_class)

    col1, col2 = st.columns(2)
    flight_date = col1.date_input('Enter your Departure Date', value=None)
    booking_date = col2.date_input('Enter the Flight Booking Date', value=None)

    col3, col4 = st.columns(2)
    src_city = col3.selectbox('Select Departure City', source_city)
    dest_city = col4.selectbox('Select Destination City', destination_city)

    num_stops = st.selectbox('Select Number of Stops:', stops)

    col5, col6 = st.columns(2)
    dept_time2 = col5.time_input('Select Departure Time', value=None)
    arr_time2 = col6.time_input('Select Arrival Time', value=None)

    submitted = st.form_submit_button('Submit')
    if submitted:

        days_left = (flight_date - booking_date).days


        if dept_time2 and arr_time2:
            # combine with a fixed date
            fixed_date = datetime(2000, 1, 1)
            dept_datetime = datetime.combine(fixed_date, dept_time2)
            arr_datetime = datetime.combine(fixed_date, arr_time2)

            # Handle crossing midnight
            if arr_datetime < dept_datetime:
                arr_datetime += timedelta(days=1)

            duration = (arr_datetime - dept_datetime).total_seconds() / 3600
        else:
            duration = None
        
        dept_time = categorize_time(dept_time2)
        arr_time = categorize_time(arr_time2)
        number_stops = convertstops(num_stops)

        user_input = np.array([[days_left, duration]])

        
        airline_encoded = airline_encoder.transform([[airline]])
        airline_df = airline_encoded.toarray().flatten()

        flight_class_encoded = label_encoder.transform([flight_class])
        flight_class_df = flight_class_encoded.flatten()

        src_city_encoded = src_city_encoder.transform([[src_city]])
        src_city_df = src_city_encoded.toarray().flatten()

        dest_city_encoded = dest_city_encoder.transform([[dest_city]])
        dest_city_df = dest_city_encoded.toarray().flatten()

        stops_encoded = ord_encoder.transform([[number_stops]])
        stops_df = stops_encoded.flatten()

        dept_time_encoded = dept_time_encoder.transform([[dept_time]])
        dept_time_df = dept_time_encoded.toarray().flatten()

        arr_time_encoded = arr_time_encoder.transform([[arr_time]])
        arr_time_df = arr_time_encoded.toarray().flatten()

        res = np.concatenate([flight_class_df, user_input.flatten(), stops_df, airline_df, src_city_df, dest_city_df, dept_time_df, arr_time_df])
        res_df = pd.DataFrame(res)

        y_pred = best_regressor.predict(res_df.T)
        st.write(f'The flight ticket price is predicted to be INR {round(y_pred[0],2)}.')

