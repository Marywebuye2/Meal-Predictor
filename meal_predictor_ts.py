import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime, timedelta
import os
try:
    import google.generativeai as genai
    AI_AVAILABLE = True
except ImportError:
    genai = None
    AI_AVAILABLE = False
    print("⚠️ google-generativeai not found. AI features will be disabled.")

import requests
import urllib.parse
import json

try:
    from dotenv import load_dotenv
    # Load environment variables
    load_dotenv('meal.env')
except ImportError:
    print("⚠️ python-dotenv not found. Environment variables must be set manually.")

# --- Google Form Integration Helpers ---

def parse_prefilled_url(url):
    """Parse a Google Form pre-filled URL to extract entry IDs."""
    try:
        parsed = urllib.parse.urlparse(url)
        params = urllib.parse.parse_qs(parsed.query)
        
        # Look for entry IDs
        entries = {}
        for key, value in params.items():
            if key.startswith('entry.'):
                # We need to map values to logical names
                # This is a bit tricky, we'll try to guess based on the value in the URL
                # The user should assume the order: School, Date, Meals, Menu, Confidence
                val_str = value[0] if isinstance(value, list) else value
                entries[key] = val_str
                
        # Return the raw text matching for the UI to confirm
        return params
    except Exception as e:
        return None

def submit_to_google_form(form_url, entries_map, data):
    """
    Submit data to Google Form.
    form_url: The 'formResponse' URL (derived from viewform URL)
    entries_map: Dict mapping logical keys ('school', 'date', etc.) to 'entry.XXXX' IDs
    data: Dict containing actual values to submit
    """
    try:
        submission_data = {}
        for logical_key, entry_id in entries_map.items():
            if logical_key in data:
                submission_data[entry_id] = data[logical_key]
        
        # Ensure we are posting to formResponse, not viewform
        post_url = form_url.replace('viewform', 'formResponse')
        if 'usp=pp_url' in post_url:
            post_url = post_url.split('?')[0]
            
        print(f"Submitting to {post_url} with data: {submission_data}")
        response = requests.post(post_url, data=submission_data)
        return response.status_code == 200
    except Exception as e:
        print(f"Error submitting form: {e}")
        return False
        
def load_form_config():
    """Load form configuration from JSON file"""
    try:
        if os.path.exists('form_config.json'):
            with open('form_config.json', 'r') as f:
                return json.load(f)
    except:
        pass
    return None

def save_form_config(config):
    """Save form configuration to JSON file"""
    with open('form_config.json', 'w') as f:
        json.dump(config, f)


GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

if GOOGLE_API_KEY and AI_AVAILABLE:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
    except Exception as e:
        print(f"⚠️ Error configuring Gemini API: {e}")
        AI_AVAILABLE = False

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
        model = joblib.load('xgboost_time_series_model.joblib')
        model_info = joblib.load('xgboost_time_series_model_info.joblib')
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
    prev_meals = inputs['today_meals']
    if prev_meals > 0:
        change_pct = ((prediction - prev_meals) / prev_meals) * 100
        if abs(change_pct) > 10:
            direction = "increase" if change_pct > 0 else "decrease"
            explanations.append(f"**{abs(change_pct):.0f}% {direction}** from today's {prev_meals:.0f} meals")
    
    # Day of week pattern
    day_mapping = {
        0: "Mondays typically have steady demand after weekends",
        1: "Tuesdays often see consistent meal patterns",
        2: "Mid-week days usually maintain stable consumption",
        3: "Thursdays show typical weekly patterns",
        4: "Fridays often have slightly lower demand before weekends"
    }
    day_explanation = day_mapping.get(inputs['tomorrow_dow_code'], "Considering daily consumption patterns")
    explanations.append(day_explanation)
    
    # Menu impact (if available)
    if inputs['tomorrow_menu'] and inputs['tomorrow_menu'] != "Unknown":
        menu_impact = {
            "Regular": "Standard menu - expected typical participation",
            "Special": "Special menu - may attract more students",
            "Holiday": "Holiday menu - variable participation patterns"
        }
        menu_note = menu_impact.get(inputs['tomorrow_menu'], "Menu type considered in prediction")
        explanations.append(f"**Menu**: {inputs['tomorrow_menu']} - {menu_note}")
    
    # Weekly trend
    if 'last_week_same_day_meals' in inputs and inputs['last_week_same_day_meals'] > 0:
        week_change = ((prediction - inputs['last_week_same_day_meals']) / inputs['last_week_same_day_meals']) * 100
        if abs(week_change) > 15:
            trend = "higher" if week_change > 0 else "lower"
            explanations.append(f"**Weekly trend**: {abs(week_change):.0f}% {trend} than same day last week")
    
    # Model confidence based on feature importance
    if model_info and 'feature_importance' in model_info:
        top_features = sorted(model_info['feature_importance'].items(), key=lambda x: x[1], reverse=True)[:3]
        top_feature_names = [f"**{feat}**" for feat, imp in top_features]
        explanations.append(f"Prediction based on: {', '.join(top_feature_names)}")
    
    return explanations

def create_tomorrow_features(today_meals, last_week_meals, tomorrow_menu, tomorrow_dow_code, school_encoding, menu_encoding, feature_columns):
    """Create features specifically for tomorrow's prediction"""
    features = {}
    
    # Today's meals (most important feature)
    features['lag_1'] = today_meals
    
    # Recent history (simplified - in practice you'd have actual data)
    features['lag_2'] = today_meals * 0.95  # Estimate yesterday
    features['lag_3'] = today_meals * 1.02  # Estimate 2 days ago
    features['lag_7'] = last_week_meals if last_week_meals > 0 else today_meals * 1.05
    
    # Rolling statistics (capturing recent trends)
    recent_meals = [today_meals, today_meals * 0.95, today_meals * 1.02]  # Current + estimates
    features['rolling_mean_3'] = np.mean(recent_meals)
    features['rolling_mean_7'] = np.mean(recent_meals + [last_week_meals] * 4) if last_week_meals > 0 else np.mean(recent_meals)
    features['rolling_std_7'] = np.std(recent_meals + [last_week_meals] * 4) if last_week_meals > 0 else np.std(recent_meals)
    
    # Day and time features
    features['day_of_week'] = tomorrow_dow_code
    features['is_weekend'] = 1 if tomorrow_dow_code >= 5 else 0
    
    # Monthly and seasonal patterns
    tomorrow_date = datetime.now() + timedelta(days=1)
    features['month'] = tomorrow_date.month
    features['day_of_month'] = tomorrow_date.day
    features['week_of_year'] = tomorrow_date.isocalendar().week
    features['is_month_start'] = 1 if tomorrow_date.day == 1 else 0
    features['is_month_end'] = 1 if tomorrow_date.day in [28, 29, 30, 31] else 0
    
    # School-specific encoding
    # This will be added after we know which school is selected
    
    # Menu encoding
    if menu_encoding and tomorrow_menu in menu_encoding:
        features['menu_encoded'] = menu_encoding[tomorrow_menu]
    elif 'menu_encoded' in feature_columns:
        features['menu_encoded'] = 0  # Default encoding
    
    # Additional trend features
    if 'daily_change' in feature_columns:
        features['daily_change'] = today_meals - (today_meals * 0.95)  # Estimate change from yesterday
    
    if 'weekly_change' in feature_columns:
        features['weekly_change'] = today_meals - last_week_meals if last_week_meals > 0 else 0
    
    # Create feature array with proper ordering and error handling
    feature_array = []
    for col in feature_columns:
        if col in features:
            feature_array.append(features[col])
        else:
            # Provide sensible defaults for missing features
            if col == 'school_encoded':
                feature_array.append(0)  # Will be set properly later
            elif col.startswith('lag_'):
                feature_array.append(today_meals)  # Default to today's meals
            elif col.startswith('rolling_'):
                feature_array.append(today_meals)  # Default to today's meals
            else:
                feature_array.append(0)  # Default for other features
    
    return feature_array

def main():
    st.title("🍽️ Tomorrow's Meal Prediction System")
    st.markdown("Predict tomorrow's meal requirements based on today's data and historical patterns")
    
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
    
    # Get tomorrow's date and check if it's a weekend
    tomorrow_date = datetime.now() + timedelta(days=1)
    tomorrow_dow = tomorrow_date.strftime("%A")
    tomorrow_dow_code = tomorrow_date.weekday()
    is_weekend = tomorrow_dow_code >= 5
    
    # Display tomorrow's info prominently
    st.header("📅 Tomorrow's Information")
    
    if is_weekend:
        st.error(f"❌ **{tomorrow_date.strftime('%A, %Y-%m-%d')} - WEEKEND**")
        st.warning("""
        ⚠️ **No meal predictions for weekends**
        
        Schools are closed on Saturdays and Sundays, so no meal predictions are available.
        Please check back on Monday for the next prediction.
        """)
        
        # Show when the next prediction will be available
        days_until_monday = (7 - tomorrow_dow_code) % 7
        next_prediction_date = tomorrow_date + timedelta(days=days_until_monday)
        st.info(f"**Next prediction available:** {next_prediction_date.strftime('%A, %Y-%m-%d')}")
        
        # Stop execution here - no predictions for weekends
        return
    
    else:
        st.success(f"✅ **{tomorrow_date.strftime('%A, %Y-%m-%d')} - WEEKDAY**")
        st.write("School is in session - meal prediction is available")
    
    # Initialize session state
    if 'prediction_result' not in st.session_state:
        st.session_state.prediction_result = None
    
    # Prediction interface (ONLY SHOW FOR WEEKDAYS)
    st.header("📊 Predict Tomorrow's Meals")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            # School selection
            schools = list(model_info['school_encoding'].keys())
            selected_school = st.selectbox("🏫 Select School", schools)
            
            # Today's meals
            today_meals = st.number_input(
                "📊 Today's Actual Meals Served",
                min_value=0,
                value=500,
                help="Enter the actual number of meals served today"
            )
            
            # Last week same day meals
            last_week_meals = st.number_input(
                "📅 Last Week Same Day Meals",
                min_value=0,
                value=0,
                help="Meals served on the same weekday last week (optional, but improves accuracy)"
            )
        
        with col2:
            # Tomorrow's menu
            menu_encoding = model_info.get('menu_encoding', {})
            menu_options = list(menu_encoding.keys()) if menu_encoding else ["Regular", "Special", "Holiday", "Unknown"]
            tomorrow_menu = st.selectbox("🍽️ Tomorrow's Menu", menu_options)
            
            # Display tomorrow's info again in the form
            st.info(f"**Tomorrow:** {tomorrow_date.strftime('%A, %Y-%m-%d')}")
            st.info(f"**Day Type:** Weekday - School in session")
        
        submitted = st.form_submit_button("🔮 Predict Tomorrow's Meals", type="primary")
        
    if submitted:
        # Double-check it's not a weekend
        if is_weekend:
            st.error("❌ No meal prediction available for weekends")
            return
        
        # Prepare features for prediction
        try:
            feature_array = create_tomorrow_features(
                today_meals=today_meals,
                last_week_meals=last_week_meals,
                tomorrow_menu=tomorrow_menu,
                tomorrow_dow_code=tomorrow_dow_code,
                school_encoding=model_info['school_encoding'],
                menu_encoding=menu_encoding,
                feature_columns=model_info['feature_columns']
            )
            
            # Set school encoding
            school_idx = model_info['feature_columns'].index('school_encoded')
            feature_array[school_idx] = model_info['school_encoding'][selected_school]
            
            # Make prediction
            prediction = model.predict([feature_array])[0]
            prediction = max(0, round(prediction))
            
            # Get school patterns
            school_patterns = get_school_patterns(model_info, selected_school)
            
            # Prepare inputs
            inputs = {
                'school': selected_school,
                'today_meals': today_meals,
                'tomorrow_menu': tomorrow_menu,
                'tomorrow_dow_code': tomorrow_dow_code,
                'last_week_same_day_meals': last_week_meals
            }
            
            # Store in session state
            st.session_state.prediction_result = {
                'prediction': prediction,
                'inputs': inputs,
                'school_patterns': school_patterns,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            st.error(f"❌ Prediction error: {e}")
            st.info("💡 Make sure all required fields are filled correctly")

    # Display Results (Outside Form)
    if st.session_state.prediction_result:
        res = st.session_state.prediction_result
        prediction = res['prediction']
        inputs = res['inputs']
        school_patterns = res['school_patterns']
        
        st.divider()
        st.success("🎯 Prediction Complete!")
        
        # Results in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            change_pct = ((prediction - inputs['today_meals']) / inputs['today_meals'] * 100) if inputs['today_meals'] > 0 else 0
            st.metric(
                "Tomorrow's Predicted Meals",
                f"{int(prediction):,}",
                f"{change_pct:+.1f}% from today"
            )
        
        with col2:
            st.metric("Today's Meals", f"{inputs['today_meals']:,}")
        
        with col3:
            if school_patterns:
                st.metric("School Average", f"{school_patterns['avg_meals']:.0f}")
            elif inputs['last_week_same_day_meals'] > 0:
                st.metric("Last Week Same Day", f"{inputs['last_week_same_day_meals']:,}")
        
        # Explanation section
        st.header("📈 Prediction Insights")
        explanations = generate_explanation(prediction, inputs, model_info, school_patterns)
        
        for explanation in explanations:
            st.write(f"• {explanation}")
        
        # Actionable recommendations
        st.header("💡 Recommended Actions")
        
        if prediction > inputs['today_meals'] * 1.15:
            st.warning("""
            **Prepare for increased demand:**
            - Increase ingredient preparation by 15-20%
            - Ensure adequate staffing
            - Check inventory levels
            """)
        elif prediction < inputs['today_meals'] * 0.85:
            st.info("""
            **Expected lower demand:**
            - Standard preparation sufficient
            - Monitor for last-minute changes
            - Consider food preservation options
            """)
        else:
            st.success("""
            **Stable demand expected:**
            - Continue with normal preparation
            - Maintain current staffing levels
            - Standard operating procedures apply
            """)
        
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
            
            st.info(f"**Confidence Level**: :{color}[{confidence}] (Based on historical accuracy for {inputs['school']})")
        
        # AI Insights Section
        st.markdown("---")
        st.subheader("🤖 AI Insights")
        
        if not AI_AVAILABLE:
            st.warning("⚠️ AI features are disabled because the `google-generativeai` library is missing.")
            st.info("Try running `pip install google-generativeai` or use `run_app.bat`.")
        elif not GOOGLE_API_KEY:
            st.warning("⚠️ to enable AI insights, please add GOOGLE_API_KEY to meal.env")
        else:
            if st.button("Generate Detailed Explanation"):
                with st.spinner("Analyzing prediction factors..."):
                    explanation = get_ai_explanation(prediction, inputs, model_info, school_patterns)
                    st.markdown(explanation)
            
            # Q&A Interface
            user_question = st.text_input("💬 Ask about this prediction:", placeholder="e.g., Why is this higher than last week?")
            if user_question:
                    with st.spinner("Thinking..."):
                        answer = get_ai_explanation(prediction, inputs, model_info, school_patterns, user_question)
                        st.markdown(f"**Answer:** {answer}")


    # --- Google Sheet / Form Configuration (Sidebar) ---
    with st.sidebar:
        st.header("⚙️ Settings")
        
        with st.expander("📝 Link Google Sheet"):
            st.markdown("""
            **To save predictions:**
            1. Create a Google Form linked to your Sheet.
            2. Add questions: **School**, **Date**, **Predicted Meals**, **Menu**, **Confidence**.
            3. Click **⋮ > Get pre-filled link**.
            4. Fill dummy data (e.g. "SCHOOL", "2025-01-01", "0", "MENU", "High").
            5. Click **Get Link**, copy it, and paste here:
            """)
            
            prefilled_link = st.text_input("Paste Pre-filled Link Here")
            
            if st.button("Save Configuration"):
                if prefilled_link:
                    parsed = urllib.parse.urlparse(prefilled_link)
                    params = urllib.parse.parse_qs(parsed.query)
                    
                    # Extract entry IDs
                    config = {'form_url': prefilled_link.split('?')[0]}
                    entries = {}
                    
                    # Heuristic mapping based on common user behavior (order of params in string might vary)
                    # We rely on the user to check mapping
                    
                    found_entries = [k for k in params.keys() if k.startswith('entry.')]
                    if len(found_entries) >= 3:
                        st.success(f"Found {len(found_entries)} fields!")
                        
                        # Let user map them manually to be safe
                        st.write("Please map the fields:")
                        
                        # We save the raw entries to session state to map in next step
                        st.session_state.temp_entries = found_entries
                        st.session_state.temp_params = params
                        st.session_state.form_url_base = prefilled_link.split('?')[0]
                        
                    else:
                        st.error("Could not find enough 'entry.XXXX' fields. Did you use the 'Get pre-filled link' option?")
            
            # Mapping UI (if entries found)
            if 'temp_entries' in st.session_state:
                entries_config = {}
                st.caption(f"Detected values: {st.session_state.temp_params}")
                
                entries_config['school'] = st.selectbox("Field for 'School'", st.session_state.temp_entries, index=0)
                entries_config['date'] = st.selectbox("Field for 'Date'", st.session_state.temp_entries, index=min(1, len(st.session_state.temp_entries)-1))
                entries_config['prediction'] = st.selectbox("Field for 'Predicted Meals'", st.session_state.temp_entries, index=min(2, len(st.session_state.temp_entries)-1))
                entries_config['menu'] = st.selectbox("Field for 'Menu'", st.session_state.temp_entries, index=min(3, len(st.session_state.temp_entries)-1))
                entries_config['confidence'] = st.selectbox("Field for 'Confidence' (Optional)", st.session_state.temp_entries + ['None'], index=min(4, len(st.session_state.temp_entries)-1) if len(st.session_state.temp_entries)>4 else 0)
                
                if st.button("Confirm & Save"):
                    final_config = {
                        'form_url': st.session_state.form_url_base,
                        'mapping': entries_config
                    }
                    save_form_config(final_config)
                    st.success("✅ Configuration saved!")
                    # Clear temp state
                    del st.session_state.temp_entries

    # --- Save to Google Sheet Logic ---
    if st.session_state.prediction_result:
        config = load_form_config()
        
        # Check if we already saved this specific prediction to ANY sheet (hacky, checking session_state)
        # Better: UI button
        
        col_save, _ = st.columns([1, 3])
        with col_save:
            if config:
                if st.button("💾 Save to Google Sheet"):
                    with st.spinner("Saving..."):
                        res = st.session_state.prediction_result
                        
                        # Prepare data
                        save_data = {
                            'school': res['inputs']['school'],
                            'date': tomorrow_date.strftime('%Y-%m-%d'),
                            'prediction': str(int(res['prediction'])),
                            'menu': res['inputs']['tomorrow_menu'],
                            'confidence': "High" # Logic to be refined
                        }
                        
                        # Add confidence actual value
                        if res['school_patterns']:
                             mape = res['school_patterns']['mape']
                             if mape < 0.1: conf = "High"
                             elif mape < 0.2: conf = "Medium"
                             else: conf = "Low"
                             save_data['confidence'] = conf
                        
                        success = submit_to_google_form(config['form_url'], config['mapping'], save_data)
                        
                        if success:
                            st.success("✅ Saved to Google Sheet!")
                        else:
                            st.error("❌ Failed to save. Check URL.")
            else:
                st.info("ℹ️ Configure Google Sheet in sidebar to save results.")

    # Additional information
    with st.expander("ℹ️ How It Works"):
        st.markdown("""
        **Prediction Methodology:**
        - **Today's Meals**: Most important indicator for tomorrow's demand
        - **Weekly Patterns**: Considers day-of-week trends (Mondays vs Fridays)
        - **Menu Impact**: Different menus attract varying participation
        - **School History**: Each school's unique consumption patterns
        - **Seasonal Trends**: Monthly and weekly variations
        
        **Important Notes:**
        - ❌ **No predictions on weekends** - schools are closed
        - ✅ **Predictions only for weekdays** (Monday-Friday)
        - 📅 **Automatically detects tomorrow's date**
        
        **For Best Results:**
        - Provide accurate today's meal count
        - Include last week's data when available
        - Select the correct menu type
        """)

def get_ai_explanation(prediction, inputs, model_info, school_patterns, user_question=None):
    """Generate AI explanation using Gemini"""
    if not AI_AVAILABLE:
        return "⚠️ AI features are not available from the server environment (missing 'google-generativeai' library)."
        
    if not GOOGLE_API_KEY:
        return "⚠️ Google API Key not found. Please add GOOGLE_API_KEY to meal.env"
        
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Construct context
        context = f"""
        You are an expert meal planning assistant for a school lunch program.
        A prediction has just been made for tomorrow's meal requirement.
        
        Context:
        - School: {inputs['school']}
        - Predicted Demand: {prediction:.0f} meals
        - Today's Demand: {inputs['today_meals']} meals
        - Tomorrow's Menu: {inputs['tomorrow_menu']}
        - Day: {inputs['tomorrow_dow_code']} (0=Mon, 4=Fri)
        
        Historical Context:
        - School Average: {school_patterns['avg_meals'] if school_patterns else 'Unknown'}
        - Average Error Rate: {school_patterns['mape']:.1%} if school_patterns else 'Unknown'
        - Important Features: {list(model_info.get('feature_importance', {}).keys())[:3]}
        """
        
        if user_question:
            prompt = f"{context}\n\nUser Question: {user_question}\n\nProvide a clear, helpful answer explaining the prediction context."
        else:
            prompt = f"{context}\n\nExplain why the model likely predicted this number. Consider the trend from today, the school's typical patterns, and the menu influence. Be concise and professional."
            
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        return f"❌ Error generating explanation: {str(e)}"

if __name__ == "__main__":
    main()
