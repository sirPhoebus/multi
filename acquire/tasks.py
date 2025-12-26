from celery import Celery
import requests

# Configure Celery
app = Celery('tasks', broker='redis://localhost:6380/0')

# Set the new configuration to avoid the deprecation warning
app.conf.broker_connection_retry_on_startup = True

@app.task
def check_bitcoin_price():
    print("Task check_bitcoin_price is running")
    try:
        # Fetch Bitcoin price from an API
        response = requests.get('https://api.coindesk.com/v1/bpi/currentprice.json')
        data = response.json()
        price = data['bpi']['USD']['rate_float']
        print(f"Current Bitcoin price: ${price:.2f}")
    except Exception as e:
        print(f"Error fetching Bitcoin price: {e}")

# Schedule the task to run every minute
app.conf.beat_schedule = {
    'check-bitcoin-price-every-minute': {
        'task': 'tasks.check_bitcoin_price',
        'schedule': 60.0,  # Every minute
    },
}