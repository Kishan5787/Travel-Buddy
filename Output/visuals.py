import pandas as pd
import matplotlib.pyplot as plt

def func:
    # Load data from Excel file
    file_path = 'D:\VIT\sem_7_fall\RPA\project\TravelBuddy\Data\data.xlsx'  # Change to your actual Excel file path
    df = pd.read_excel(file_path)

    # Preprocessing data
    df['Fair'] = df['Fair'].replace({'₹ ': '', ',': ''}, regex=True)  # Remove ₹ symbol and commas
    df['Fair'] = pd.to_numeric(df['Fair'], errors='coerce')  # Convert to numeric, forcing errors to NaN
    df = df.dropna(subset=['Fair'])  # Drop rows where 'Fair' is NaN
    df['Fair'] = df['Fair'].astype(int)  # Convert to integer

    # Extracting the number of stops
    df['Number of Stops'] = df['Stops'].apply(lambda x: len(x.split(',')))

    # Converting departure time to datetime format for accurate plotting
    df['Departure Time'] = pd.to_datetime(df['Departure'], format='%H:%M').dt.hour

    # Plot: Number of Stops vs Price
    stops_vs_price_plot = plt.figure(figsize=(10, 6))  # Assign the plot to a variable
    plt.scatter(df['Number of Stops'], df['Fair'], color='blue', marker='o')
    plt.title('Number of Stops vs Price')
    plt.xlabel('Number of Stops')
    plt.ylabel('Fair (₹)')
    plt.grid(True)
    plt.savefig('stops_vs_price.png')
    plt.show()

    # Plot: Time of Departure vs Price
    departure_vs_price_plot = plt.figure(figsize=(10, 6))  # Assign the plot to a variable
    plt.scatter(df['Departure Time'], df['Fair'], color='red', marker='o')
    plt.title('Time of Departure vs Price')
    plt.xlabel('Time of Departure (Hour)')
    plt.ylabel('Fair (₹)')
    plt.grid(True)
    plt.savefig('departure_vs_price.png')
    plt.show()