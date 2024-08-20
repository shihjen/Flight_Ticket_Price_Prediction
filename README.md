## Flight Ticket Prediction
![flight_ticket](assets/img/flight_ticket.jpg)

### Project Description

In this project, I explored an open-source flight booking dataset from Kaggle to perform predictive analysis. The dataset includes the following features:

1. **Airline:** Categorical feature representing the airline company, with 6 different airlines.
2. **Source City:** Categorical feature indicating the city of departure, with 6 unique cities.
3. **Departure Time:** A derived categorical feature that groups time periods into 6 unique time labels, representing the flight’s departure time.
4. **Stops:** Categorical feature with 3 distinct values indicating the number of stops between the source and destination cities.
5. **Arrival Time:** A derived categorical feature that groups time intervals into 6 distinct time labels, representing the flight’s arrival time.
6. **Destination City:** Categorical feature representing the city where the flight lands, with 6 unique cities.
7. **Class:** Categorical feature indicating the seat class, with 2 distinct values: Business and Economy.
8. **Duration:** Continuous feature showing the total travel time between cities in hours.
9. **Days Left:** A derived feature calculated by subtracting the booking date from the travel date.
10. **Price:** The target variable representing the flight ticket price.

I constructed a regression model using the Random Forest algorithm to predict flight ticket prices based on airline, source city, destination city, departure time, arrival time, number of stops, seat class, flight duration, and the time between booking and travel dates. The model achieved a Mean Absolute Error (MAE) of INR 1,129.81 on the ticket price. The trained model was deployed as a Streamlit application. 

However, the project had some limitations. The dataset was relatively small and covered only a few cities in India, restricting the model's ability to generalize predictions beyond these specific cities. The primary goal of this project was to demonstrate the practical application of machine learning to business use cases. With a more comprehensive dataset, the workflow used here could be applied to develop a generalized model for predicting flight ticket prices across a broader range of locations.


<div><a href = "https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction?select=Clean_Dataset.csv">Data Source</a></div>
<div><a href = " ">Analysis Notebook</a></div>
<div><a href = " ">Web Application</a></div>

