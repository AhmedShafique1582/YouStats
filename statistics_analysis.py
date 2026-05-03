import pandas as pd
import numpy as np
from youtube_api import parse_duration
from scipy import stats
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# ── Existing functions ────────────────────────────────────────────────────────

def prepare_dataframe(videos):
    df = pd.DataFrame(videos)
    df["published_at"] = pd.to_datetime(df["published_at"])
    df = df.sort_values("published_at").reset_index(drop=True)
    df["engagement_rate"] = ((df["likes"] + df["comments"]) / df["views"].replace(0,1) * 100).round(2)
    df["likes_per_view"] = (df["likes"] / df["views"].replace(0,1)).round(4)
    df["duration_seconds"] = df["duration"].apply(parse_duration)
    df["duration_minutes"] = (df["duration_seconds"] / 60).round(2)
    df["days_since_start"] = (df["published_at"] - df["published_at"].min()).dt.days
    df["cumulative_views"] = df["views"].cumsum()
    df["month_year"] = df["published_at"].dt.to_period("M")
    df["hour"] = df["published_at"].dt.hour
    df["day_of_week"] = df["published_at"].dt.dayofweek
    df["day_name"] = df["published_at"].dt.day_name()
    df["month_num"] = df["published_at"].dt.month
    df["month_name"] = df["published_at"].dt.strftime("%b")
    df["year"] = df["published_at"].dt.year
    # ── Engineered features for better ML ────────────────────────────────────
    df["is_weekend"]       = df["day_of_week"].isin([5, 6]).astype(int)
    df["quarter"]          = df["published_at"].dt.quarter
    df["title_length"]     = df["title"].str.len().fillna(50)
    df["title_words"]      = df["title"].str.split().str.len().fillna(8)
    # Rolling averages of views (channel momentum — most predictive feature)
    df["rolling_avg_5"]    = df["views"].shift(1).rolling(5,  min_periods=1).mean().fillna(df["views"].mean())
    df["rolling_avg_10"]   = df["views"].shift(1).rolling(10, min_periods=1).mean().fillna(df["views"].mean())
    df["rolling_avg_20"]   = df["views"].shift(1).rolling(20, min_periods=1).mean().fillna(df["views"].mean())
    df["log_rolling_avg5"] = np.log1p(df["rolling_avg_5"])
    # Days since last upload (upload frequency signal)
    df["days_since_last"]  = df["published_at"].diff().dt.days.fillna(7).clip(0, 365)
    # Video sequence number (channel growth proxy)
    df["video_seq"]        = np.arange(len(df))
    return df

def get_descriptive_stats(df):
    cols = ["views","likes","comments","engagement_rate","likes_per_view","duration_minutes"]
    result = {}
    for col in cols:
        data = df[col].dropna()
        result[col] = {
            "mean": round(data.mean(), 2), "median": round(data.median(), 2),
            "mode": round(data.mode()[0], 2) if len(data.mode()) > 0 else 0,
            "std_dev": round(data.std(), 2), "variance": round(data.var(), 2),
            "min": round(data.min(), 2), "max": round(data.max(), 2),
            "q1": round(data.quantile(0.25), 2), "q3": round(data.quantile(0.75), 2),
            "iqr": round(data.quantile(0.75) - data.quantile(0.25), 2),
            "skewness": round(data.skew(), 2), "kurtosis": round(data.kurtosis(), 2),
        }
    return result

def get_confidence_intervals(df, confidence_level=0.95):
    cols = ["views","likes","comments","engagement_rate","likes_per_view","duration_minutes"]
    result = {}
    for col in cols:
        data = df[col].dropna()
        n = len(data); mean = data.mean(); std_err = stats.sem(data)
        ci = stats.t.interval(confidence=confidence_level, df=n-1, loc=mean, scale=std_err)
        result[col] = {
            "mean": round(mean, 2), "confidence_level": confidence_level,
            "ci_lower": round(ci[0], 2), "ci_upper": round(ci[1], 2),
            "margin_of_error": round((ci[1]-ci[0])/2, 2),
        }
    return result

def fit_probability_distribution(df):
    results = {}
    for col in ["views","likes","comments"]:
        data = df[col].dropna()
        mu, sigma = norm.fit(data)
        lambda_p = data.mean()
        ks_stat, p_value = stats.kstest(data, 'norm', args=(mu, sigma))
        results[col] = {
            "normal": {"mu": round(mu,2), "sigma": round(sigma,2),
                       "ks_stat": round(ks_stat,4), "p_value": round(p_value,4)},
            "poisson": {"lambda": round(lambda_p, 2)}
        }
    return results

def build_regression_model(df):
    X = df["days_since_start"].values.reshape(-1,1)
    Y = df["cumulative_views"].values
    split = int(len(df)*0.80)
    X_train,X_test = X[:split],X[split:]
    Y_train,Y_test = Y[:split],Y[split:]
    model = LinearRegression()
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    r2 = r2_score(Y_test, Y_pred)
    mae = mean_absolute_error(Y_test, Y_pred)
    last_day = int(df["days_since_start"].max())
    future_days = np.array(range(last_day, last_day+365)).reshape(-1,1)
    future_predictions = model.predict(future_days)
    return {
        "model": model, "r2_score": round(r2,4), "mae": round(mae,2),
        "slope": round(model.coef_[0],2), "intercept": round(model.intercept_,2),
        "X_test": X_test, "Y_test": Y_test, "Y_pred": Y_pred,
        "future_days": future_days, "future_predictions": future_predictions
    }

def detect_outliers(df):
    outliers = {}
    for col in ["views","likes","comments"]:
        data = df[col]
        Q1=data.quantile(0.25); Q3=data.quantile(0.75); IQR=Q3-Q1
        lb=Q1-1.5*IQR; ub=Q3+1.5*IQR
        out_vids = df[(data<lb)|(data>ub)][["title","published_at",col]].sort_values(col,ascending=False)
        outliers[col] = {"lower_bound":round(lb,2),"upper_bound":round(ub,2),
                         "outlier_count":len(out_vids),"outlier_videos":out_vids}
    return outliers

def categorize_videos(df):
    mean_v=df["views"].mean(); std_v=df["views"].std()
    def cat(v):
        if v > mean_v+2*std_v: return "Viral"
        elif v > mean_v+std_v: return "Hit"
        elif v > mean_v: return "Above Average"
        elif v > mean_v-std_v: return "Average"
        else: return "Flop"
    df = df.copy()
    df["category"] = df["views"].apply(cat)
    summary = df.groupby("category").agg(
        video_count=("video_id","count"), avg_views=("views","mean"),
        avg_likes=("likes","mean"), avg_comments=("comments","mean")
    ).round(2)
    return df, summary

# ── New functions ─────────────────────────────────────────────────────────────

def get_channel_health_score(df, channel_stats):
    score = 0
    avg_eng = df["engagement_rate"].mean()
    eng_score = 30 if avg_eng>5 else 20 if avg_eng>2 else 10 if avg_eng>1 else 5
    score += eng_score

    monthly = df.groupby(df["published_at"].dt.to_period("M")).size()
    std_u = monthly.std() if len(monthly)>1 else 0
    mean_u = monthly.mean() if len(monthly)>0 else 1
    cons = 1 - min(std_u/(mean_u+1), 1)
    cons_score = round(cons*20)
    score += cons_score

    if len(df) > 10:
        growth = (df.tail(10)["views"].mean() - df.head(10)["views"].mean()) / (df.head(10)["views"].mean()+1)
        grow_score = 25 if growth>0.5 else 20 if growth>0.2 else 15 if growth>0 else 5
    else:
        grow_score = 10
    score += grow_score

    n = channel_stats.get("total_videos",0)
    vid_score = 15 if n>200 else 12 if n>100 else 8 if n>50 else 5 if n>20 else 3
    score += vid_score

    subs = channel_stats.get("subscribers",0)
    views = channel_stats.get("total_views",1)
    ratio = subs/(views+1)*100
    sub_score = 10 if ratio>10 else 8 if ratio>5 else 5 if ratio>1 else 3
    score += sub_score

    score = min(score, 100)
    status = "Excellent" if score>=85 else "Optimized" if score>=70 else "Good" if score>=55 else "Average" if score>=40 else "Needs Work"
    return {"score": score, "status": status,
            "factors": {"Engagement":eng_score,"Consistency":cons_score,"Growth":grow_score,"Volume":vid_score,"Subscriber Ratio":sub_score}}

def get_growth_metrics(df):
    df_s = df.sort_values("published_at").copy()
    now = df_s["published_at"].max()

    def pviews(start, end):
        return df_s[(df_s["published_at"]>=start)&(df_s["published_at"]<=end)]["views"].sum()

    def gpct(curr, prev):
        return round((curr-prev)/(prev+1)*100, 2)

    w_curr = pviews(now-pd.Timedelta(days=7), now)
    w_prev = pviews(now-pd.Timedelta(days=14), now-pd.Timedelta(days=7))
    m_curr = pviews(now-pd.Timedelta(days=30), now)
    m_prev = pviews(now-pd.Timedelta(days=60), now-pd.Timedelta(days=30))
    y_curr = pviews(now-pd.Timedelta(days=365), now)
    y_prev = pviews(now-pd.Timedelta(days=730), now-pd.Timedelta(days=365))

    monthly = df_s.copy()
    monthly["month"] = monthly["published_at"].dt.to_period("M")
    mdata = monthly.groupby("month").agg(
        views=("views","sum"), videos=("video_id","count"),
        likes=("likes","sum"), comments=("comments","sum")
    ).reset_index()
    mdata["month"] = mdata["month"].astype(str)
    mdata["engagement_rate"] = ((mdata["likes"]+mdata["comments"])/(mdata["views"].replace(0,1))*100).round(2)

    return {
        "weekly_pct": gpct(w_curr, w_prev), "monthly_pct": gpct(m_curr, m_prev),
        "yearly_pct": gpct(y_curr, y_prev),
        "weekly_views": int(w_curr), "monthly_views": int(m_curr), "yearly_views": int(y_curr),
        "avg_engagement": round(df_s["engagement_rate"].mean(), 2),
        "monthly_data": mdata.to_dict(orient="records"),
    }

def get_best_time_to_post(df):
    df = df.copy()
    hour_avg = df.groupby("hour")["views"].mean().round(0)
    best_hour = int(hour_avg.idxmax())
    DAYS = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    day_avg = df.groupby("day_name")["views"].mean().round(0).reindex(DAYS, fill_value=0)
    best_day = day_avg.idxmax()
    upload_by_day = df.groupby("day_name").size().reindex(DAYS, fill_value=0)
    monthly_avg = df.groupby("month_num")["views"].mean().round(0)
    heatmap = df.groupby(["day_of_week","hour"])["views"].mean().reset_index()
    heatmap.columns = ["day","hour","views"]
    return {
        "best_hour": best_hour, "best_day": best_day,
        "hour_avg": {str(k): float(v) for k,v in hour_avg.items()},
        "day_avg": {k: float(v) for k,v in day_avg.items()},
        "upload_by_day": {k: int(v) for k,v in upload_by_day.items()},
        "monthly_avg": {str(k): float(v) for k,v in monthly_avg.items()},
        "heatmap": heatmap.to_dict(orient="records"),
    }

def estimate_revenue(df, rpm=4.0):
    df = df.copy()
    df["revenue"] = (df["views"] / 1000) * rpm
    df["month"] = df["published_at"].dt.to_period("M")
    monthly = df.groupby("month").agg(revenue=("revenue","sum"), views=("views","sum")).reset_index()
    monthly["month"] = monthly["month"].astype(str)
    monthly["revenue"] = monthly["revenue"].round(2)
    mean_v = df["views"].mean(); std_v = df["views"].std()
    def cat(v):
        if v > mean_v+2*std_v: return "Viral"
        elif v > mean_v+std_v: return "Hit"
        elif v > mean_v: return "Above Average"
        else: return "Long Tail"
    df["category"] = df["views"].apply(cat)
    cat_rev = df.groupby("category")["revenue"].sum().reset_index()
    cat_rev["revenue"] = cat_rev["revenue"].round(2)
    top10 = df.nlargest(10,"revenue")[["title","views","revenue"]].copy()
    top10["revenue"] = top10["revenue"].round(2)
    return {
        "total_revenue": round(df["revenue"].sum(), 2),
        "avg_monthly_revenue": round(monthly["revenue"].mean(), 2),
        "rpm": rpm,
        "monthly_revenue": monthly.tail(24).to_dict(orient="records"),
        "category_revenue": cat_rev.to_dict(orient="records"),
        "top10_revenue": top10.to_dict(orient="records"),
    }

def get_probability_analysis(df):
    views = df["views"].values
    mean_v = float(views.mean()); std_v = float(views.std())
    z_1m  = (1_000_000  - mean_v) / (std_v+1)
    z_10m = (10_000_000 - mean_v) / (std_v+1)
    p_1m  = round((1 - norm.cdf(z_1m)) * 100, 2)
    p_10m = round((1 - norm.cdf(z_10m)) * 100, 2)
    x_range = np.linspace(mean_v - 4*std_v, mean_v + 4*std_v, 300)
    y_range = norm.pdf(x_range, mean_v, std_v)
    df2 = df.copy()
    df2["z_score"] = ((df2["views"] - mean_v) / (std_v+1)).round(3)
    top_z = df2.nlargest(20,"z_score")[["title","views","z_score"]].to_dict(orient="records")
    return {
        "mean": round(mean_v,2), "std": round(std_v,2),
        "p_1m": p_1m, "p_10m": p_10m,
        "z_1m": round(z_1m,3), "z_10m": round(z_10m,3),
        "curve_x": x_range.tolist(), "curve_y": y_range.tolist(),
        "z_scores": top_z,
    }

def build_random_forest_model(df):
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import RobustScaler
    df = df.copy().sort_values("published_at").reset_index(drop=True)
    features = [
        # Original features
        "duration_minutes", "days_since_start", "hour", "day_of_week", "month_num",
        # Engineered features (high impact)
        "rolling_avg_5", "rolling_avg_10", "rolling_avg_20", "log_rolling_avg5",
        "days_since_last", "title_length", "title_words",
        "is_weekend", "quarter", "video_seq",
    ]
    X = df[features].fillna(0)
    y = np.log1p(df["views"])

    # ── Shuffle before split so test isn't all-recent videos ──────────────────
    from sklearn.utils import shuffle as sk_shuffle
    X_s, y_s = sk_shuffle(X, y, random_state=42)

    split = int(len(df) * 0.8)
    X_train, X_test = X_s.iloc[:split], X_s.iloc[split:]
    y_train, y_test = y_s.iloc[:split], y_s.iloc[split:]

    model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred_log = model.predict(X_test)

    # R² on log scale (stable, avoids negative from extreme outliers)
    r2_log = round(r2_score(y_test, y_pred_log), 4)

    # MAE / RMSE on original scale for readability
    y_test_orig  = np.expm1(y_test)
    y_pred_orig  = np.expm1(y_pred_log)
    mae  = round(mean_absolute_error(y_test_orig, y_pred_orig), 2)
    rmse = round(np.sqrt(mean_squared_error(y_test_orig, y_pred_orig)), 2)

    importance = {f: round(float(v), 4) for f, v in zip(features, model.feature_importances_)}

    # ── Historical: ALL videos, sorted by date for chart ──────────────────────
    y_all_log = model.predict(X)
    y_all_pred = np.expm1(y_all_log)
    history = []
    for i in range(len(df)):
        orig_idx = X.index[i]
        history.append({
            "index": i,
            "date": str(df["published_at"].iloc[i])[:10],
            "actual": float(df["views"].iloc[i]),
            "predicted": float(y_all_pred[i]),
            "is_test": orig_idx in X_test.index,
        })

    # ── Future: next 30 videos ────────────────────────────────────────────────
    last_day        = float(df["days_since_start"].max())
    avg_gap         = last_day / max(len(df) - 1, 1)
    med_dur         = float(df["duration_minutes"].median())
    mode_hour       = int(df["hour"].mode()[0])
    mode_dow        = int(df["day_of_week"].mode()[0])
    last_month      = int(df["month_num"].iloc[-1])
    med_roll5       = float(df["rolling_avg_5"].median())
    med_roll10      = float(df["rolling_avg_10"].median())
    med_roll20      = float(df["rolling_avg_20"].median())
    log_roll5       = float(np.log1p(med_roll5))
    med_gap         = float(df["days_since_last"].median())
    med_title_len   = float(df["title_length"].median())
    med_title_words = float(df["title_words"].median())
    med_weekend     = int(df["is_weekend"].mode()[0])
    last_seq        = int(df["video_seq"].max())

    future = []
    for i in range(1, 31):
        fut_month   = ((last_month - 1 + i) % 12) + 1
        fut_quarter = ((fut_month - 1) // 3) + 1
        feats = np.array([[
            med_dur,                        # duration_minutes
            last_day + avg_gap * i,         # days_since_start
            mode_hour,                      # hour
            mode_dow,                       # day_of_week
            fut_month,                      # month_num
            med_roll5,                      # rolling_avg_5
            med_roll10,                     # rolling_avg_10
            med_roll20,                     # rolling_avg_20
            log_roll5,                      # log_rolling_avg5
            med_gap,                        # days_since_last
            med_title_len,                  # title_length
            med_title_words,                # title_words
            med_weekend,                    # is_weekend
            fut_quarter,                    # quarter
            last_seq + i,                   # video_seq
        ]])
        future.append({"index": len(df) + i,
                       "predicted": float(np.expm1(model.predict(feats)[0]))})


    test_chart = [{"index": i, "actual": float(a), "predicted": float(p)}
                  for i, (a, p) in enumerate(zip(y_test_orig.values, y_pred_orig))]

    return {
        "r2": r2_log, "mae": mae, "rmse": rmse,
        "feature_importance": importance,
        "chart_data": test_chart,
        "history": history,
        "future": future,
        "split_index": split,
    }

