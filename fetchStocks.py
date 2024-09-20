import requests
import pandas as pd
import datetime
import time

API_KEY = 'temp'

ticker = 'SPY'
multiplier = 1
timespan = 'minute'

end_date = datetime.datetime.now()
start_date = end_date - datetime.timedelta(days=730)  # last 2 years
date_ranges = []

delta = datetime.timedelta(days=5)
current_start_date = start_date

while current_start_date < end_date:
    current_end_date = current_start_date + delta
    if current_end_date > end_date:
        current_end_date = end_date
    date_ranges.append((current_start_date.strftime('%Y-%m-%d'), current_end_date.strftime('%Y-%m-%d')))
    current_start_date = current_end_date

all_data = []

for _, (from_date, to_date) in enumerate(date_ranges):
    url = f'https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}?adjusted=true&sort=asc&limit=50000&apiKey={API_KEY}'
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data['results'])
    df['Date'] = pd.to_datetime(df['t'], unit='ms')
    df.rename(columns={'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'}, inplace=True)
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    all_data.append(df)
    print(f"Fetched data from {from_date} to {to_date}")

    time.sleep(12) # polygon only lets up to 5 api calls per minute

full_df = pd.concat(all_data, ignore_index=True)
full_df.to_csv('spy.csv', index=False)
print("All data saved to csv")
