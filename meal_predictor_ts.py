import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime, timedelta

# Page config
st.set_page_config(
    page_title="Next-Day Meal Predictor",
    page_icon="🍽️",
    layout="wide"
)

@st.cache_resource
def load_ts_model():
    """Load the time-series model"""
    try:
        model = joblib.load('time_series_model.joblib')
        model_info = joblib.load('time_series_model_info.joblib')
        st.success("✅ Time-series model loaded successfully!")
        return model, model_info
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None

def get_school_patterns(model_info, school):
    """Get historical patterns for a school"""
    if not model_info or 'backtest_results' not in model_info:
        return None
    
    for result in model_info['backtest_results']:
        if result['school'] == school:
            return result
    return None

def generate_explanation(prediction, inputs, model_info, school_patterns):
    """Generate human-readable explanation for the prediction"""
    explanations = []
    
    # School historical performance
    if school_patterns:
        avg_meals = school_patterns['avg_meals']
        if prediction > avg_meals * 1.2:
            explanations.append(f"**Higher than usual** for {inputs['school']} (typically {avg_meals:.0f} meals)")
        elif prediction < avg_meals * 0.8:
            explanations.append(f"**Lower than usual** for {inputs['school']} (typically {avg_meals:.0f} meals)")
        else:
            explanations.append(f"**Typical demand** for {inputs['school']} (average: {avg_meals:.0f} meals)")
    
    # Previous day impact
    prev_meals = inputs['previous_day_meals']
    if prev_meals > 0:
        change_pct = ((prediction - prev_meals) / prev_meals) * 100
        if abs(change_pct) > 10:
            direction = "increase" if change_pct > 0 else "decrease"
            explanations.append(f"**{abs(change_pct):.0f}% {direction}** from yesterday's {prev_meals:.0f} meals")
    
    # Day of week pattern
    day_mapping = {
        0: "Mondays typically have steady demand after weekends",
        1: "Tuesdays often see consistent meal patterns",
        2: "Mid-week days usually maintain stable consumption",
        3: "Thursdays show typical weekly patterns",
        4: "Fridays often have slightly lower demand before weekends",
        5: "Weekends usually have significantly lower school meal demand",
        6: "Weekends typically see minimal school meal service"
    }
    day_explanation = day_mapping.get(inputs['next_day_of_week'], "Considering daily consumption patterns")
    explanations.append(day_explanation)
    
    # Menu impact (if available)
    if inputs['next_day_menu'] and inputs['next_day_menu'] != "Unknown":
        explanations.append(f"**Menu**: {inputs['next_day_menu']} - certain menus can affect participation")
    
    # Model confidence based on feature importance
    if model_info and 'feature_importance' in model_info:
        top_features = sorted(model_info['feature_importance'].items(), key=lambda x: x[1], reverse=True)[:3]
        top_feature_names = [f"**{feat}**" for feat, imp in top_features]
        explanations.append(f"Prediction based on: {', '.join(top_feature_names)}")
    
    return explanations

def main():
    st.title("🍽️ Next-Day Meal Prediction System")
    st.markdown("Predict tomorrow's meal requirements based on historical patterns")
    
    # Load model
    model, model_info = load_ts_model()
    
    if model is None:
        st.info("""
        **To get started:**
        1. Run `python time_series_model.py` to train the time-series model
        2. Refresh this page after training completes
        """)
        return
    
    # Show model performance
    with st.expander("🔍 Model Performance Overview"):
        if model_info:
            st.write(f"**Overall Accuracy:** {model_info['performance']['test_mape']:.1%} error")
            st.write(f"**Average Error:** ±{model_info['performance']['test_mae']:.0f} meals")
            st.write(f"**Features Used:** {len(model_info['feature_columns'])} time-series features")
            
            # Show school performance
            if 'backtest_results' in model_info:
                st.subheader("🏫 School-Level Accuracy")
                results_df = pd.DataFrame(model_info['backtest_results'])
                st.dataframe(results_df[['school', 'mae', 'mape', 'avg_meals', 'samples']].sort_values('mape'))
    
    # Prediction interface
    st.header("📊 Make Prediction")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            # School selection
            schools = list(model_info['school_encoding'].keys())
            school = st.selectbox("🏫 Select School", schools)
            
            # Previous day meals
            previous_day_meals = st.number_input(
                "📊 Yesterday's Actual Meals Served",
                min_value=0,
                max_value=3000,
                value=500,
                help="Enter the actual number of meals served yesterday"
            )
        
        with col2:
            # Next day menu
            menu_options = list(model_info['menu_encoding'].keys()) if model_info.get('menu_encoding') else ["Regular", "Special", "Holiday", "Unknown"]
            next_day_menu = st.selectbox("🍽️ Next Day Menu", menu_options)
            
            # Next day of week
            day_options = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            next_day_of_week = st.selectbox("📅 Next Day of Week", day_options)
            day_mapping = {day: i for i, day in enumerate(day_options)}
            next_day_dow_code = day_mapping[next_day_of_week]
            
            # Auto-calculate weekend
            is_weekend = 1 if next_day_dow_code >= 5 else 0
            st.info(f"**Weekend:** {'Yes' if is_weekend else 'No'}")
        
        submitted = st.form_submit_button("🔮 Predict Next Day Meals")
        
        if submitted:
            # Prepare features for prediction
            features = {}
            
            # Lag features (using previous day's actual)
            features['lag_1'] = previous_day_meals
            features['lag_2'] = previous_day_meals  # Simplified - in real scenario, you'd have more history
            features['lag_3'] = previous_day_meals
            features['lag_7'] = previous_day_meals
            
            # Rolling statistics (simplified)
            features['rolling_mean_3'] = previous_day_meals
            features['rolling_mean_7'] = previous_day_meals
            features['rolling_std_7'] = previous_day_meals * 0.1  # Estimate
            
            # Date features
            features['day_of_week'] = next_day_dow_code
            features['is_weekend'] = is_weekend
            features['month'] = datetime.now().month
            features['is_month_start'] = 0
            features['is_month_end'] = 0
            
            # Menu
            if model_info.get('menu_encoding'):
                features['menu_encoded'] = model_info['menu_encoding'].get(next_day_menu, 0)
            
            # School
            features['school_encoded'] = model_info['school_encoding'][school]
            
            # Create feature array in correct order
            feature_array = [features[col] for col in model_info['feature_columns']]
            
            # Make prediction
            try:
                prediction = model.predict([feature_array])[0]
                prediction = max(0, prediction)  # Ensure non-negative
                
                # Get school patterns for explanation
                school_patterns = get_school_patterns(model_info, school)
                
                # Prepare inputs for explanation
                inputs = {
                    'school': school,
                    'previous_day_meals': previous_day_meals,
                    'next_day_menu': next_day_menu,
                    'next_day_of_week': next_day_dow_code
                }
                
                # Display results
                st.success("🎯 Prediction Complete!")
                
                # Results in columns
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Predicted Meals",
                        f"{int(prediction):,}",
                        f"{((prediction - previous_day_meals) / previous_day_meals * 100):.1f}%" if previous_day_meals > 0 else "N/A"
                    )
                
                with col2:
                    st.metric("Yesterday's Meals", f"{previous_day_meals:,}")
                
                with col3:
                    if school_patterns:
                        st.metric("School Average", f"{school_patterns['avg_meals']:.0f}")
                
                # Explanation section
                st.header("📈 Prediction Explanation")
                explanations = generate_explanation(prediction, inputs, model_info, school_patterns)
                
                for explanation in explanations:
                    st.write(f"• {explanation}")
                
                # Confidence indicator
                if school_patterns:
                    error_rate = school_patterns['mape']
                    if error_rate < 0.1:
                        confidence = "High"
                        color = "green"
                    elif error_rate < 0.2:
                        confidence = "Medium"
                        color = "orange"
                    else:
                        confidence = "Low"
                        color = "red"
                    
                    st.info(f"**Confidence Level**: :{color}[{confidence}] (Based on historical accuracy for this school)")
                
            except Exception as e:
                st.error(f"❌ Prediction error: {e}")

if __name__ == "__main__":
    main()