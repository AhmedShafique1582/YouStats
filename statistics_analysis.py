import pandas as pd
import numpy as np
from youtube_api import parse_duration
from scipy import stats
from scipy.stats import norm, poisson
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

def prepare_dataframe(videos):
    # Convert videos list to pandas DataFrame
    df = pd.DataFrame(videos)
    
    # Convert published_at to datetime
    df["published_at"] = pd.to_datetime(df["published_at"])
    
    # Sort by date oldest to newest
    df = df.sort_values("published_at").reset_index(drop=True)
    
    # Add derived variables
    df["engagement_rate"] = ((df["likes"] + df["comments"]) / df["views"] * 100).round(2)
    df["likes_per_view"] = (df["likes"] / df["views"]).round(4)
    df["duration_seconds"] = df["duration"].apply(parse_duration)
    df["duration_minutes"] = (df["duration_seconds"] / 60).round(2)
    
    # Upload frequency — day number since first video
    df["days_since_start"] = (df["published_at"] - df["published_at"].min()).dt.days
    
    # Cumulative views over time (for regression)
    df["cumulative_views"] = df["views"].cumsum()
    
    # Month-Year column for grouping
    df["month_year"] = df["published_at"].dt.to_period("M")
    
    return df