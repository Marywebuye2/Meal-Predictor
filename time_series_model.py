import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import joblib
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

print("🚀 CREATING TIME-SERIES MEAL PREDICTION MODEL")
print("=" * 60)

# Load data
data = pd.read_excel('data.xlsx')
print(f"📊 Data loaded: {len(data)} records")

# Convert date and sort
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values(['School', 'Date'])

print(f"🏫 Schools in data: {data['School'].nunique()}")
print(f"📅 Date range: {data['Date'].min()} to {data['Date'].max()}")

# Create time-series features
def create_time_series_features(df):
    """Create features for time-series prediction"""
    
    features_list = []
    
    for school in df['School'].unique():
        school_data = df[df['School'] == school].copy()
        school_data = school_data.sort_values('Date')
        
        # Create lag features (previous days)
        for lag in [1, 2, 3, 7]:  # Previous day, 2 days ago, 3 days ago, same day last week
            school_data[f'lag_{lag}'] = school_data['Actual Meals Served'].shift(lag)
        
        # FIX 1: Fixed rolling statistics (NO DATA LEAKAGE)
        school_data['rolling_mean_3'] = school_data['Actual Meals Served'].shift(1).rolling(3, min_periods=1).mean()
        school_data['rolling_mean_7'] = school_data['Actual Meals Served'].shift(1).rolling(7, min_periods=1).mean()
        school_data['rolling_std_7'] = school_data['Actual Meals Served'].shift(1).rolling(7, min_periods=1).std()
        
        # Day of week features
        school_data['day_of_week'] = school_data['Date'].dt.dayofweek
        school_data['is_weekend'] = (school_data['Date'].dt.dayofweek >= 5).astype(int)
        
        # FIX 2: Added simple but powerful features
        school_data['day_of_month'] = school_data['Date'].dt.day
        school_data['week_of_year'] = school_data['Date'].dt.isocalendar().week
        
        # Month and season
        school_data['month'] = school_data['Date'].dt.month
        school_data['is_month_start'] = school_data['Date'].dt.is_month_start.astype(int)
        school_data['is_month_end'] = school_data['Date'].dt.is_month_end.astype(int)
        
        features_list.append(school_data)
    
    return pd.concat(features_list, ignore_index=True)

print("\n🔄 Creating time-series features...")
featured_data = create_time_series_features(data)

# Prepare for training
# FIX 3: Updated feature columns
feature_columns = [
    'lag_1', 'lag_2', 'lag_3', 'lag_7',
    'rolling_mean_3', 'rolling_mean_7', 'rolling_std_7',
    'day_of_week', 'day_of_month', 'week_of_year', 'is_weekend', 
    'month', 'is_month_start', 'is_month_end'
]

# Add menu if available
if 'Menu' in featured_data.columns:
    # Encode menu types
    menu_encoding = {menu: i for i, menu in enumerate(featured_data['Menu'].unique())}
    featured_data['menu_encoded'] = featured_data['Menu'].map(menu_encoding)
    feature_columns.append('menu_encoded')
    print(f"🍽️ Menu types found: {len(menu_encoding)}")

# Remove rows with missing values (from lag features)
model_data = featured_data.dropna(subset=feature_columns + ['Actual Meals Served'])

print(f"✅ Final training data: {len(model_data)} records")

# School encoding
school_encoding = {school: i for i, school in enumerate(model_data['School'].unique())}
model_data['school_encoded'] = model_data['School'].map(school_encoding)
feature_columns.append('school_encoded')

print(f"🎯 Features: {feature_columns}")

# Train model
X = model_data[feature_columns]
y = model_data['Actual Meals Served']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

print(f"\n🤖 Training model...")
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Predictions
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

# Calculate metrics
train_mae = mean_absolute_error(y_train, train_pred)
test_mae = mean_absolute_error(y_test, test_pred)
train_mape = mean_absolute_percentage_error(y_train, train_pred)
test_mape = mean_absolute_percentage_error(y_test, test_pred)

print(f"\n🎯 MODEL PERFORMANCE:")
print(f"Training MAE: {train_mae:.2f} meals ({train_mape:.1%} error)")
print(f"Test MAE:     {test_mae:.2f} meals ({test_mape:.1%} error)")
print(f"Average meals: {y.mean():.2f}")

# Backtesting function
def backtest_model(df, model, feature_columns, school_encoding, menu_encoding=None):
    """Backtest the model on historical data"""
    print(f"\n🧪 BACKTESTING RESULTS BY SCHOOL:")
    print("=" * 50)
    
    results = []
    
    for school in df['School'].unique():
        school_data = df[df['School'] == school].copy()
        school_data = school_data.sort_values('Date')
        
        if len(school_data) < 10:  # Skip schools with insufficient data
            continue
            
        school_predictions = []
        school_actuals = []
        
        # Predict for each day in the school's data
        for i in range(1, len(school_data)):
            current_row = school_data.iloc[i]
            previous_row = school_data.iloc[i-1]
            
            # Prepare features for prediction
            features = {}
            features['lag_1'] = previous_row['Actual Meals Served']
            features['lag_2'] = school_data.iloc[i-2]['Actual Meals Served'] if i >= 2 else previous_row['Actual Meals Served']
            features['lag_3'] = school_data.iloc[i-3]['Actual Meals Served'] if i >= 3 else previous_row['Actual Meals Served']
            features['lag_7'] = school_data.iloc[i-7]['Actual Meals Served'] if i >= 7 else previous_row['Actual Meals Served']
            
            # FIXED: Use only past data for rolling stats
            past_data = school_data.iloc[max(0, i-7):i]['Actual Meals Served']
            features['rolling_mean_3'] = past_data.tail(3).mean() if len(past_data) >= 3 else past_data.mean()
            features['rolling_mean_7'] = past_data.tail(7).mean() if len(past_data) >= 7 else past_data.mean()
            features['rolling_std_7'] = past_data.tail(7).std() if len(past_data) >= 7 else past_data.std()
            
            features['day_of_week'] = current_row['day_of_week']
            features['is_weekend'] = current_row['is_weekend']
            features['day_of_month'] = current_row['day_of_month']
            features['week_of_year'] = current_row['week_of_year']
            features['month'] = current_row['month']
            features['is_month_start'] = current_row['is_month_start']
            features['is_month_end'] = current_row['is_month_end']
            
            if 'menu_encoded' in feature_columns and menu_encoding:
                features['menu_encoded'] = menu_encoding.get(current_row['Menu'], 0)
            
            features['school_encoded'] = school_encoding[school]
            
            # Create feature array in correct order
            feature_array = [features[col] for col in feature_columns]
            
            # Make prediction
            prediction = model.predict([feature_array])[0]
            actual = current_row['Actual Meals Served']
            
            school_predictions.append(prediction)
            school_actuals.append(actual)
        
        if school_predictions:
            school_mae = mean_absolute_error(school_actuals, school_predictions)
            school_mape = mean_absolute_percentage_error(school_actuals, school_predictions)
            avg_meals = np.mean(school_actuals)
            
            results.append({
                'school': school,
                'mae': school_mae,
                'mape': school_mape,
                'avg_meals': avg_meals,
                'samples': len(school_actuals)
            })
            
            print(f"🏫 {school:<30} MAE: {school_mae:6.1f} meals ({school_mape:5.1%} error) | Avg: {avg_meals:5.1f} meals")
    
    return results

# Run backtesting
backtest_results = backtest_model(model_data, model, feature_columns, school_encoding, 
                                 menu_encoding if 'Menu' in featured_data.columns else None)

# Save model and metadata
model_info = {
    'feature_columns': feature_columns,
    'school_encoding': school_encoding,
    'menu_encoding': menu_encoding if 'Menu' in featured_data.columns else {},
    'feature_importance': dict(zip(feature_columns, model.feature_importances_)),
    'performance': {
        'test_mae': test_mae,
        'test_mape': test_mape,
        'avg_meals': y.mean()
    },
    'backtest_results': backtest_results
}

joblib.dump(model, 'time_series_model.joblib')
joblib.dump(model_info, 'time_series_model_info.joblib')

print(f"\n💾 Model saved as 'time_series_model.joblib'")
print(f"💾 Model info saved as 'time_series_model_info.joblib'")

# Show feature importance
print(f"\n📊 FEATURE IMPORTANCE:")
for feature, importance in sorted(model_info['feature_importance'].items(), key=lambda x: x[1], reverse=True):
    print(f"  {feature:<20}: {importance:.3f}")

print(f"\n🎉 TIME-SERIES MODEL TRAINING COMPLETED!")
