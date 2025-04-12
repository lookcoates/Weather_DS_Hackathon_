import sys
import os
from apscheduler.schedulers.blocking import BlockingScheduler

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .ingest_data import fetch_weather, save_data
from ml_model.train import train_and_save_model

def update_pipeline():
    """Run the complete data update and model retraining pipeline"""
    try:
        print("Starting data update pipeline...")
        new_data = fetch_weather()
        save_data(new_data)
        train_and_save_model()
        print("Pipeline completed successfully")
    except Exception as e:
        print(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        scheduler = BlockingScheduler()
        scheduler.add_job(update_pipeline, 'interval', hours=6)
        print("Scheduler started - updates will run every 6 hours")
        scheduler.start()
    except KeyboardInterrupt:
        print("\nScheduler stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"Scheduler failed: {str(e)}")
        sys.exit(1)
