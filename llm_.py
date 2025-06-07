import pandas as pd
import matplotlib.pyplot as plt
import io
import smtplib
from email.message import EmailMessage
import google.generativeai as genai

def JoinName(firstname, lastname):
    name = firstname + " " + lastname

    # Load data from Excel file (now without headers)
    file_path = r"D:\VIT\sem_7_fall\RPA\project\TravelBuddy\Data\flightdata.xlsx"
    df = pd.read_excel(file_path, header=None)  # Read without headers

    # Assign custom column names for easier reference
    df.columns = ['Airlines', 'Departure', 'Arrival', 'Duration', 'Fare', 'Stops']

    # Preprocessing data (Fare column without '₹' symbol and commas)
    df['Fare'] = df['Fare'].replace({'₹ ': '', ',': ''}, regex=True)
    df['Fare'] = pd.to_numeric(df['Fare'], errors='coerce')
    df = df.dropna(subset=['Fare'])
    df['Fare'] = df['Fare'].astype(int)

    # Function to handle 'Duration' (convert to total minutes)
    def duration_to_minutes(duration):
        parts = str(duration).split()  # Split by space
        total_minutes = 0
        for part in parts:
            if part[:-1].isdigit():  # Check if the numeric part is a digit
                if part[-1] == 'h':  # Hours
                    total_minutes += int(part[:-1]) * 60
                elif part[-1] == 'm':  # Minutes
                    total_minutes += int(part[:-1])
        return total_minutes

    # Apply duration conversion
    df['Duration'] = df['Duration'].apply(duration_to_minutes)

    # Extract the number of stops (based on commas in the stops)
    df['Number of Stops'] = df['Stops'].apply(lambda x: len(str(x).split(',')))

    # Create plots and save to buffers
    plots = [
        ('Average Cost per Airline', 'avg_cost_per_airline.png'),
        ('Cheapest Flight', 'cheapest_flight.png'),
        ('Number of Stops vs Airline Name', 'stops_vs_airline.png'),
        ('Fastest Flight', 'fastest_flight.png')
    ]
    img_buffers = {}

    # 1. Average Cost per Airline (Bar Plot)
    plt.figure(figsize=(10, 6))
    avg_cost_per_airline = df.groupby('Airlines')['Fare'].mean().sort_values()
    avg_cost_per_airline.plot(kind='bar', color='skyblue')
    plt.title('Average Cost per Airline')
    plt.xlabel('Airline Name')
    plt.ylabel('Average Fare (₹)')
    plt.xticks(rotation=90)
    img_buffers['avg_cost_per_airline.png'] = io.BytesIO()
    plt.savefig(img_buffers['avg_cost_per_airline.png'], format='png')
    plt.close()

    # 2. Cheapest Flight (Bar Plot)
    plt.figure(figsize=(10, 6))
    cheapest_flight = df.loc[df['Fare'].idxmin()]
    plt.bar(cheapest_flight['Airlines'], cheapest_flight['Fare'], color='green')
    plt.title('Cheapest Flight')
    plt.ylabel('Fare (₹)')
    plt.xlabel('Airline Name')
    img_buffers['cheapest_flight.png'] = io.BytesIO()
    plt.savefig(img_buffers['cheapest_flight.png'], format='png')
    plt.close()

    # 3. Number of Stops vs Airline Name (Bar Plot)
    plt.figure(figsize=(10, 6))
    stops_per_airline = df.groupby('Airlines')['Number of Stops'].mean()
    stops_per_airline.plot(kind='bar', color='orange')
    plt.title('Number of Stops vs Airline')
    plt.xlabel('Airline Name')
    plt.ylabel('Average Number of Stops')
    plt.xticks(rotation=90)
    img_buffers['stops_vs_airline.png'] = io.BytesIO()
    plt.savefig(img_buffers['stops_vs_airline.png'], format='png')
    plt.close()

    # 4. Fastest Flight (Bar Plot)
    plt.figure(figsize=(10, 6))
    fastest_flight = df.loc[df['Duration'].idxmin()]  # Flight with least duration
    plt.bar(fastest_flight['Airlines'], fastest_flight['Duration'], color='blue')
    plt.title(f"Fastest Flight: {fastest_flight['Airlines']}")
    plt.ylabel('Travel Time (Minutes)')
    plt.xlabel('Airline Name')
    img_buffers['fastest_flight.png'] = io.BytesIO()
    plt.savefig(img_buffers['fastest_flight.png'], format='png')
    plt.close()

    # Integrate LLM to provide restaurants and resting places at stops
    genai.configure(api_key="AIzaSyA8Wsh5G0Coz5Mx1-FDcmX0I5uGrwcr6gc")

    stops = df['Stops'].tolist()
    airlines = df['Airlines'].tolist()

    prompt = f"These are the stops in flight {stops} and also this the the respective airline {airlines}. I need you to just tell me the types of food available inside airport and restaurants nearby the stops and also if the does this airline provide free lounge in this stop(yes/no) give it in a table format [stop name - food option and restaurants - Free_lounge]. give me a single table with rows as no of stops, so i need a single table nothing more (note: please fill in the table yourself and give me a complete table with all the contents, dont ask me to find any information on my own)"
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    response = model.generate_content(prompt)
    response_text = response.text
    print(f"Response for stops {stops}:\n", response_text)

    # Send the images via email
    #email_address = "dharshan.b2021@vitstudent.ac.in"  # Replace with recipient's email
    #sender_email = "dharshanbalajikaruppa18@gmail.com"  # Replace with sender's email
    #sender_password = "vpqy tsad nqri rlec"  # Replace with sender's email password

    #msg = EmailMessage()
    #msg['Subject'] = "Flight Data Analysis Plots"
    #msg['From'] = sender_email
    #msg['To'] = email_address
    #msg.set_content("Attached are the requested plots from the flight data analysis.")

    # Attach each image from the buffer
    #for plot_title, plot_file in plots:
    #    img_buffers[plot_file].seek(0)
    #    msg.add_attachment(img_buffers[plot_file].read(), maintype='image', subtype='png', filename=plot_file)

    # # Send email
    # with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
    #     smtp.login(sender_email, sender_password)
    #     smtp.send_message(msg)

    return name


JoinName("hi","dag")