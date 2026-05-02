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
    #Convert videos list to pandas DataFrame
    df = pd.DataFrame(videos)
    
    #Convert published_at to datetime
    df["published_at"] = pd.to_datetime(df["published_at"])
    
    #Sort by date oldest to newest
    df = df.sort_values("published_at").reset_index(drop=True)
    
    #Add derived variables
    df["engagement_rate"] = ((df["likes"] + df["comments"]) / df["views"] * 100).round(2)
    df["likes_per_view"] = (df["likes"] / df["views"]).round(4)
    df["duration_seconds"] = df["duration"].apply(parse_duration)
    df["duration_minutes"] = (df["duration_seconds"] / 60).round(2)
    
    #Upload frequency — day number since first video
    df["days_since_start"] = (df["published_at"] - df["published_at"].min()).dt.days
    
    #Cumulative views over time (for regression)
    df["cumulative_views"] = df["views"].cumsum()
    
    #Month-Year column for grouping
    df["month_year"] = df["published_at"].dt.to_period("M")
    
    return df

def get_descriptive_stats(df):
    #Select the columns 
    numeric_cols=["views","likes","comments","engagement_rate","likes_per_view","duration_minutes"]
    stats={}
    for col in numeric_cols:
        data=df[col].dropna()
        stats[col]={
            #Using Functions to find the statistics
            "mean":round(data.mean(),2),
            "median":round(data.median(),2),
            "mode":round(data.mode()[0],2),
            "std_dev":round(data.std(),2),
            "variance":round(data.var(),2),
            "min":round(data.min(),2),
            "max":round(data.max(),2),
            "q1":round(data.quantile(0.25),2),
            "q3":round(data.quantile(0.75),2),
            "iqr":round(data.quantile(0.75)-data.quantile(0.25),2),
            "skewness":round(data.skew(),2),
            "kurtosis":round(data.kurtosis(),2),
        }
    return stats

def get_confidence_intervals(df, confidence_level=0.95):
    numeric_cols=["views","likes","comments","engagement_rate","likes_per_view","duration_minutes"]
    ci_results={}
    for col in numeric_cols:
        data=df[col].dropna()

        n=len(data)
        mean=data.mean()
        std_err=stats.sem(data)

        #Calculate the confidence interval
        ci=stats.t.interval(
            confidence=confidence_level,
            df=n-1, #degrees of freedom
            loc=mean, #center
            scale=std_err #spread
        )

        ci_results[col] = {
            "mean": round(mean, 2),
            "confidence_level": confidence_level,
            "ci_lower": round(ci[0], 2),
            "ci_upper": round(ci[1], 2),
            "margin_of_error": round((ci[1] - ci[0])/2, 2),
        }
    return ci_results

        
def fit_probability_distribution(df):
    results = {}
    
    for col in ["views", "likes", "comments"]:
        data = df[col].dropna()
        
        # Fit Normal Distribution
        mu, sigma = norm.fit(data) #mu= mean, sigma= standard deviation
        
        # Fit Poisson Distribution
        lambda_poisson = data.mean()
        
        # Kolmogorov-Smirnov test
        # Tests how well normal distribution fits our data
        ks_stat, p_value = stats.kstest(
            data, 
            'norm', 
            args=(mu, sigma)
        )
        
        results[col] = {
            "normal": {
                "mu": round(mu, 2),        # mean
                "sigma": round(sigma, 2),   # std dev
                "ks_stat": round(ks_stat, 4),
                "p_value": round(p_value, 4)
            },
            "poisson": {
                "lambda": round(lambda_poisson, 2)
            }
        }
    
    return results