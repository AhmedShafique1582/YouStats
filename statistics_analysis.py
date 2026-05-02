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

def build_regression_model(df):
    #Use days as X axis and views as Y axis
    X=df["days_since_start"].values.reshape(-1,1)
    Y=df["cumulative_views"].values

    #Split data into 80% and 20%
    split=int(len(df)*0.80)
    X_train,X_test=X[:split],X[split:]
    Y_train,Y_test=Y[:split],Y[split:]

    #Fit Linear Regression model
    model=LinearRegression()
    model.fit(X_train,Y_train)

    #Predict on test set
    Y_pred=model.predict(X_test)

    #Measure Accuracy
    r2=r2_score(Y_test,Y_pred)
    mae=mean_absolute_error(Y_test,Y_pred)

    #Prediction for next 30 days
    last_day=int(df["days_since_start"].max())
    future_days=np.array(range(last_day,last_day+365)).reshape(-1,1)
    future_predictions=model.predict(future_days)

    return {
        "model": model,
        "r2_score": round(r2, 4),
        "mae": round(mae, 2),
        "slope": round(model.coef_[0], 2),
        "intercept": round(model.intercept_, 2),
        "X_test": X_test,
        "Y_test": Y_test,
        "Y_pred": Y_pred,
        "future_days": future_days,
        "future_predictions": future_predictions
    }

def detect_outliers(df):
    outliers = {}
    
    for col in ["views", "likes", "comments"]:
        data = df[col]
        
        # IQR method
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Find outlier videos
        outlier_videos = df[
            (df[col] < lower_bound) | (df[col] > upper_bound)
        ][["title", "published_at", col]].sort_values(col, ascending=False)
        
        outliers[col] = {
            "lower_bound": round(lower_bound, 2),
            "upper_bound": round(upper_bound, 2),
            "outlier_count": len(outlier_videos),
            "outlier_videos": outlier_videos
        }
    
    return outliers

def categorize_videos(df):
    # Calculate thresholds
    mean_views = df["views"].mean()
    std_views = df["views"].std()
    
    # Categorize each video
    def get_category(views):
        if views > mean_views + 2 * std_views:
            return "Viral"
        elif views > mean_views + std_views:
            return "Hit"
        elif views > mean_views:
            return "Above Average"
        elif views > mean_views - std_views:
            return "Average"
        else:
            return "Flop"
    
    df["category"] = df["views"].apply(get_category)
    
    # Summary
    summary = df.groupby("category").agg(
        video_count = ("video_id", "count"),
        avg_views = ("views", "mean"),
        avg_likes = ("likes", "mean"),
        avg_comments = ("comments", "mean")
    ).round(2)
    
    return df, summary

