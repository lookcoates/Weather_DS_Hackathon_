import sys
from flask import Flask, render_template, send_from_directory, request
import pandas as pd
from datetime import datetime
import os
import logging


# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler('supaboard.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
logger.debug("Starting application initialization")

try:
    app = Flask(__name__)
    app.static_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
    app.static_url_path = '/static'
    logger.debug("Flask app initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Flask app: {str(e)}")
    raise

def load_weather_data():
    """Load weather data"""
    try:
        df = pd.read_csv(os.path.join(os.path.dirname(__file__), '../ml_model/data/weather_data.csv'))
        df['date_time'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception as e:
        logger.error(f"Error loading weather data: {str(e)}")
        raise

@app.route('/')
def show_dashboard():
    """Render simplified weather dashboard"""
    try:
        weather_df = load_weather_data()
        latest = weather_df.iloc[-1].to_dict()
        latest['date_time'] = latest['date_time'].strftime('%Y-%m-%d %H:%M')
        
        logger.info("Rendering dashboard with latest data")
        return render_template('index.html',
                            latest_data=latest,
                            plot_path=None)
    except Exception as e:
        logger.error(f"Dashboard error: {str(e)}")
        return f"Error loading dashboard: {str(e)}", 500

@app.route('/api/historical')
def historical_data():
    """Return historical weather data for charting"""
    try:
        df = load_weather_data()
        logger.info(f"Loaded weather data with {len(df)} records")
        
        # Get requested cities from query params
        cities = request.args.get('cities')
        if cities:
            cities = cities.split(',')
            df = df[df['city'].isin(cities)]
            logger.info(f"Filtered data for cities: {cities}")
        
        # Prepare data for the chart with robust timestamp handling
        df['timestamp'] = pd.to_datetime(
            df['timestamp'],
            format='mixed',
            dayfirst=False,
            yearfirst=True,
            errors='coerce'
        )
        # Filter out invalid timestamps before aggregation
        df = df[df['timestamp'].notna()]
        # Group by both timestamp and city
        historical = df.groupby(['timestamp', 'city']).agg({'temperature': 'mean'}).reset_index()
        logger.info(f"Prepared historical data with {len(historical)} time points")
        
        # Format response with data per city
        response = {
            'labels': historical['timestamp'].dt.strftime('%Y-%m-%d %H:%M').tolist(),
            'data': {},
            'predictions': {}
        }
        
        for city in historical['city'].unique():
            city_data = historical[historical['city'] == city]
            response['data'][city] = city_data['temperature'].tolist()
        logger.info(f"Returning historical data: {response}")
        return response
    except Exception as e:
        logger.error(f"Error loading historical data: {str(e)}")
        return {"error": str(e)}, 500

# Remove the custom static file handler - Flask will handle static files automatically
# with the static_url_path='/static' we set earlier

if __name__ == '__main__':
    logger.info("Starting Flask application")
    app.run(host='0.0.0.0', port=5000, debug=True)