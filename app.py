import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Aadhaar Sentinel AI", layout="wide", page_icon="🛡️")

# Load background image
import base64
import os

def get_base64_bg():
    try:
        if os.path.exists("bg.png"):
            with open("bg.png", "rb") as f:
                data = base64.b64encode(f.read()).decode()
            return f"data:image/png;base64,{data}"
        else:
            for path in ["./bg.png", "../bg.png", "bg.png"]:
                if os.path.exists(path):
                    with open(path, "rb") as f:
                        data = base64.b64encode(f.read()).decode()
                    return f"data:image/png;base64,{data}"
    except Exception as e:
        pass
    return None

bg_image = get_base64_bg()

# Aggressive custom CSS with background image
bg_style = f"""
    .stApp {{
        background-image: url('{bg_image}');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
""" if bg_image else """
    .stApp {{
        background: linear-gradient(135deg, #f5f7fa 0%, #e8eaf6 50%, #f5f7fa 100%);
    }}
"""

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap');
    
    * {{ font-family: 'Inter', sans-serif; }}
    
    {bg_style}
    
    .main {{ color: #1a1a1a; }}
    
    [data-testid="stSidebar"] {{ display: none; }}
    section[data-testid="stSidebar"] {{ display: none; }}
    
    h1, h2, h3 {{ color: #1a1a1a !important; text-shadow: none; }}
    p {{ text-shadow: none; color: #2d2d2d; }}
    
    .metric-card {{
        background: rgba(255,255,255,0.95);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(0,0,0,0.1);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        transition: all 0.3s;
    }}
    
    .metric-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.15);
    }}
    
    .critical {{ border-left: 4px solid #ff4757; }}
    .high {{ border-left: 4px solid #ffa502; }}
    .normal {{ border-left: 4px solid #2ed573; }}
    
    .big-stat {{
        font-size: 3rem;
        font-weight: 900;
        line-height: 1;
        margin: 8px 0;
    }}
    
    .stat-label {{
        font-size: 0.9rem;
        opacity: 0.7;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}
    
    .stButton > button {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 12px 32px;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s;
    }}
    
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background: rgba(255,255,255,0.6);
        padding: 10px;
        border-radius: 12px;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        background: rgba(255,255,255,0.8);
        border-radius: 8px;
        color: #1a1a1a;
        font-weight: 600;
        padding: 12px 24px;
    }}
    
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }}
    
    div[data-testid="stExpander"] {{
        background: rgba(255,255,255,0.8);
        border: 1px solid rgba(0,0,0,0.1);
        border-radius: 8px;
        margin: 8px 0;
    }}
    
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_processed' not in st.session_state:
    st.session_state.data_processed = False
if 'district_data' not in st.session_state:
    st.session_state.district_data = None
if 'ml_results' not in st.session_state:
    st.session_state.ml_results = None

def load_and_clean(file):
    if file is None:
        return None
    df = pd.read_csv(file)
    df.columns = [c.strip().lower().replace('_', ' ') for c in df.columns]
    
    for col in ['state', 'district']:
        if col in df.columns:
            df[col] = df[col].str.strip().str.title()
    
    return df

def analyze_single_dataset(df, dtype):
    """Analyze patterns in a single dataset"""
    if 'district' not in df.columns:
        return None
    
    results = []
    grouped = df.groupby('district')
    
    for district, group in grouped:
        state = group.iloc[0].get('state', 'Unknown') if 'state' in group.columns else 'Unknown'
        
        ignore_cols = ['date', 'state', 'district', 'pincode']
        total = 0
        youth_total = 0
        adult_total = 0
        
        for col in group.columns:
            if col.lower() in ignore_cols:
                continue
            try:
                col_sum = pd.to_numeric(group[col], errors='coerce').sum()
                if not pd.isna(col_sum):
                    total += col_sum
                    if '5' in col or '17' in col:
                        youth_total += col_sum
                    elif '18' in col or 'greater' in col.lower():
                        adult_total += col_sum
            except:
                continue
        
        record_count = len(group)
        avg_per_record = total / record_count if record_count > 0 else 0
        youth_ratio = (youth_total / total * 100) if total > 0 else 0
        
        volume_score = min(total / 10000, 40)
        demographic_score = 20 if youth_ratio > 60 or youth_ratio < 20 else 0
        frequency_score = 30 if record_count > 100 else (15 if record_count > 50 else 0)
        
        anomaly_score = volume_score + demographic_score + frequency_score
        risk_level = 'CRITICAL' if anomaly_score > 60 else ('HIGH' if anomaly_score > 30 else 'NORMAL')
        
        results.append({
            'district': district,
            'state': state,
            'total_updates': total,
            'youth_updates': youth_total,
            'adult_updates': adult_total,
            'record_count': record_count,
            'avg_per_record': avg_per_record,
            'youth_ratio': youth_ratio,
            'anomaly_score': anomaly_score,
            'risk_level': risk_level,
            'dataset_type': dtype
        })
    
    return pd.DataFrame(results).sort_values('anomaly_score', ascending=False)

def analyze_multiple_datasets(enrol_df, demo_df, bio_df):
    """Compare multiple datasets"""
    district_map = {}
    
    datasets = [
        (enrol_df, 'enrol'),
        (demo_df, 'demo'),
        (bio_df, 'bio')
    ]
    
    for df, dtype in datasets:
        if df is None:
            continue
        
        if 'district' not in df.columns:
            continue
            
        grouped = df.groupby('district')
        
        for district, group in grouped:
            if district not in district_map:
                state = group.iloc[0].get('state', 'Unknown') if 'state' in group.columns else 'Unknown'
                district_map[district] = {
                    'district': district,
                    'state': state,
                    'enrol_total': 0,
                    'demo_total': 0,
                    'bio_total': 0
                }
            
            ignore_cols = ['date', 'state', 'district', 'pincode']
            for col in group.columns:
                if col.lower() in ignore_cols:
                    continue
                
                try:
                    col_sum = pd.to_numeric(group[col], errors='coerce').sum()
                    if not pd.isna(col_sum):
                        if dtype == 'enrol':
                            district_map[district]['enrol_total'] += col_sum
                        elif dtype == 'demo':
                            district_map[district]['demo_total'] += col_sum
                        elif dtype == 'bio':
                            district_map[district]['bio_total'] += col_sum
                except:
                    continue
    
    results = []
    for district, data in district_map.items():
        gap = data['demo_total'] - data['bio_total']
        compliance_rate = (data['bio_total'] / data['demo_total'] * 100) if data['demo_total'] > 0 else 100
        migration_index = (data['demo_total'] / data['enrol_total']) if data['enrol_total'] > 0 else 0
        
        anomaly_score = 0
        if gap > 0:
            anomaly_score += min(gap / 1000, 50)
        if compliance_rate < 50:
            anomaly_score += 30
        elif compliance_rate < 80:
            anomaly_score += 15
        if migration_index > 2:
            anomaly_score += 20
        
        risk_level = 'CRITICAL' if anomaly_score > 60 else ('HIGH' if anomaly_score > 30 else 'NORMAL')
        
        results.append({
            'district': district,
            'state': data['state'],
            'enrol_total': data['enrol_total'],
            'demo_total': data['demo_total'],
            'bio_total': data['bio_total'],
            'gap': gap,
            'gap_abs': abs(gap),
            'compliance_rate': compliance_rate,
            'migration_index': migration_index,
            'anomaly_score': anomaly_score,
            'risk_level': risk_level
        })
    
    return pd.DataFrame(results).sort_values('anomaly_score', ascending=False)

def train_ml_models(data, files_count):
    """Train ensemble of ML models for advanced anomaly detection"""
    
    # Prepare features
    if files_count >= 2:
        feature_cols = ['enrol_total', 'demo_total', 'bio_total', 'gap_abs', 'compliance_rate', 'migration_index']
    else:
        feature_cols = ['total_updates', 'youth_updates', 'adult_updates', 'record_count', 'youth_ratio']
    
    # Filter available columns
    available_features = [col for col in feature_cols if col in data.columns]
    X = data[available_features].fillna(0)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=available_features)
    
    results = {}
    
    # Model 1: Isolation Forest (Unsupervised Anomaly Detection)
    iso_forest = IsolationForest(
        contamination=0.15,  # Expect 15% anomalies
        random_state=42,
        n_estimators=100
    )
    data['ml_anomaly_score'] = iso_forest.fit_predict(X_scaled)
    data['ml_anomaly_confidence'] = -iso_forest.score_samples(X_scaled)  # Higher = more anomalous
    
    # Normalize confidence to 0-100
    min_conf = data['ml_anomaly_confidence'].min()
    max_conf = data['ml_anomaly_confidence'].max()
    data['ml_confidence'] = ((data['ml_anomaly_confidence'] - min_conf) / (max_conf - min_conf) * 100)
    
    results['isolation_forest'] = {
        'model': iso_forest,
        'anomalies_detected': len(data[data['ml_anomaly_score'] == -1]),
        'total_samples': len(data)
    }
    
    # Model 2: Random Forest Classifier (Supervised - learns from risk levels)
    # Create labels from existing risk levels
    label_map = {'NORMAL': 0, 'HIGH': 1, 'CRITICAL': 2}
    y = data['risk_level'].map(label_map)
    
    if len(X) > 10:  # Need minimum samples
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        rf_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        rf_classifier.fit(X_train, y_train)
        
        # Predict probabilities
        rf_probs = rf_classifier.predict_proba(X_scaled)
        
        # Handle different number of classes (sometimes not all 3 classes exist in data)
        n_classes = rf_probs.shape[1]
        classes = rf_classifier.classes_
        
        # Initialize probability column
        data['ml_critical_probability'] = 0.0
        
        # Check if CRITICAL class exists and get its probability
        if 2 in classes:  # CRITICAL class exists
            critical_idx = list(classes).index(2)
            data['ml_critical_probability'] = rf_probs[:, critical_idx] * 100
        elif n_classes > 1:  # Use highest available risk class
            data['ml_critical_probability'] = rf_probs[:, -1] * 100
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': available_features,
            'importance': rf_classifier.feature_importances_
        }).sort_values('importance', ascending=False)
        
        results['random_forest'] = {
            'model': rf_classifier,
            'accuracy': rf_classifier.score(X_test, y_test),
            'feature_importance': feature_importance,
            'n_classes': n_classes
        }
    
    # Model 3: Gradient Boosting (Risk Score Prediction)
    y_risk = data['anomaly_score']
    
    if len(X) > 10:
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_risk, test_size=0.2, random_state=42)
        
        gb_regressor = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        gb_regressor.fit(X_train, y_train)
        
        # Predict risk scores
        data['ml_predicted_risk'] = gb_regressor.predict(X_scaled)
        
        results['gradient_boosting'] = {
            'model': gb_regressor,
            'r2_score': gb_regressor.score(X_test, y_test),
            'mean_error': np.mean(np.abs(y_test - gb_regressor.predict(X_test)))
        }
    
    # Ensemble score (combine all models)
    if 'ml_confidence' in data.columns and 'ml_critical_probability' in data.columns:
        data['ml_ensemble_score'] = (
            data['ml_confidence'] * 0.4 +
            data['ml_critical_probability'] * 0.4 +
            (data['ml_predicted_risk'] if 'ml_predicted_risk' in data.columns else data['anomaly_score']) * 0.2
        )
    
    # Add ML-enhanced risk level
    if 'ml_ensemble_score' in data.columns:
        data['ml_risk_level'] = pd.cut(
            data['ml_ensemble_score'],
            bins=[0, 30, 60, 100],
            labels=['NORMAL', 'HIGH', 'CRITICAL']
        )
    
    return data, results

# Header
st.markdown("""
<div style='text-align: center; padding: 20px 0;'>
    <h1 style='font-size: 4.5rem; font-weight: 900; margin: 0; 
               background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
               -webkit-background-clip: text;
               -webkit-text-fill-color: transparent;
               filter: drop-shadow(0 0 20px rgba(102,126,234,0.5));'>
        🛡️ SENTINEL AI
    </h1>
    <p style='font-size: 1.4rem; opacity: 0.95; margin-top: 10px; font-weight: 600;'>
        National Aadhaar Intelligence Platform
    </p>
</div>
""", unsafe_allow_html=True)

# File Upload Section (Centered)
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    with st.container():
        uploaded_files_list = st.file_uploader(
            "📂 Load Data",
            type=['csv'],
            accept_multiple_files=True,
            key='multi_upload'
        )
        
        if uploaded_files_list:
            if st.button("🚀 Analyze", use_container_width=True):
                with st.spinner("🧠 Processing data and training ML models..."):
                    files_count = len(uploaded_files_list)
                    df_enrol = None
                    df_demo = None
                    df_bio = None
                    
                    for file in uploaded_files_list:
                        df = load_and_clean(file)
                        filename = file.name.lower()
                        
                        if 'enrol' in filename or 'enrollment' in filename:
                            df_enrol = df
                        elif 'demo' in filename or 'demographic' in filename:
                            df_demo = df
                        elif 'bio' in filename or 'biometric' in filename:
                            df_bio = df
                        else:
                            if df_enrol is None:
                                df_enrol = df
                            elif df_demo is None:
                                df_demo = df
                            else:
                                df_bio = df
                    
                    if files_count >= 2:
                        district_data = analyze_multiple_datasets(df_enrol, df_demo, df_bio)
                    else:
                        if df_enrol is not None:
                            single_df = df_enrol
                            dtype = 'enrol'
                        elif df_demo is not None:
                            single_df = df_demo
                            dtype = 'demo'
                        else:
                            single_df = df_bio
                            dtype = 'bio'
                        district_data = analyze_single_dataset(single_df, dtype)
                    
                    # Train ML models
                    district_data, ml_results = train_ml_models(district_data, files_count)
                    
                    st.session_state.district_data = district_data
                    st.session_state.ml_results = ml_results
                    st.session_state.data_processed = True
                    st.session_state.files_count = files_count
                st.rerun()

# Main Dashboard
if st.session_state.data_processed and st.session_state.district_data is not None:
    data = st.session_state.district_data
    files_count = st.session_state.get('files_count', 1)
    
    # Determine metrics based on mode
    if files_count >= 2:
        critical_count = len(data[data['risk_level'] == 'CRITICAL'])
        total_gap = data[data['gap'] > 0]['gap'].sum() if 'gap' in data.columns else 0
        avg_compliance = data['compliance_rate'].mean() if 'compliance_rate' in data.columns else 0
        top_district = data.iloc[0]['district']
        
        metric_configs = [
            ("🚨 Critical Zones", critical_count, "#ff4757"),
            ("⚠️ Unverified", f"{int(total_gap):,}", "#ffa502"),
            ("📊 Compliance", f"{avg_compliance:.1f}%", "#5f27cd"),
            ("🎯 Top Risk", top_district, "#764ba2")
        ]
    else:
        critical_count = len(data[data['risk_level'] == 'CRITICAL'])
        total_updates = int(data['total_updates'].sum()) if 'total_updates' in data.columns else 0
        avg_youth_ratio = data['youth_ratio'].mean() if 'youth_ratio' in data.columns else 0
        top_district = data.iloc[0]['district']
        
        metric_configs = [
            ("🚨 High Activity", critical_count, "#ff4757"),
            ("📈 Total Updates", f"{total_updates:,}", "#ffa502"),
            ("👥 Youth Ratio", f"{avg_youth_ratio:.1f}%", "#5f27cd"),
            ("🎯 Hotspot", top_district, "#764ba2")
        ]
    
    # Display metrics
    cols = st.columns(4)
    for idx, (label, value, color) in enumerate(metric_configs):
        with cols[idx]:
            st.markdown(f"""
            <div class='metric-card' style='border-left: 4px solid {color};'>
                <div class='stat-label'>{label}</div>
                <div class='big-stat' style='color: {color};'>{value}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ML Model Performance Section
    if st.session_state.ml_results:
        st.markdown("### 🤖 AI-Powered Intelligence Engine")
        
        ml_results = st.session_state.ml_results
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if 'isolation_forest' in ml_results:
                anomaly_pct = (ml_results['isolation_forest']['anomalies_detected'] / 
                              ml_results['isolation_forest']['total_samples'] * 100)
                st.markdown(f"""
                <div class='metric-card' style='border-left: 4px solid #667eea; text-align: center;'>
                    <div class='stat-label'>🔍 Isolation Forest</div>
                    <div style='font-size: 1.8rem; font-weight: 700; color: #667eea; margin: 8px 0;'>
                        {anomaly_pct:.1f}%
                    </div>
                    <div style='font-size: 0.8rem; opacity: 0.7;'>Anomalies Detected</div>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            if 'random_forest' in ml_results:
                accuracy = ml_results['random_forest']['accuracy'] * 100
                st.markdown(f"""
                <div class='metric-card' style='border-left: 4px solid #2ed573; text-align: center;'>
                    <div class='stat-label'>🎯 Random Forest</div>
                    <div style='font-size: 1.8rem; font-weight: 700; color: #2ed573; margin: 8px 0;'>
                        {accuracy:.1f}%
                    </div>
                    <div style='font-size: 0.8rem; opacity: 0.7;'>Classification Accuracy</div>
                </div>
                """, unsafe_allow_html=True)
        
        with col3:
            if 'gradient_boosting' in ml_results:
                r2 = ml_results['gradient_boosting']['r2_score'] * 100
                st.markdown(f"""
                <div class='metric-card' style='border-left: 4px solid #ffa502; text-align: center;'>
                    <div class='stat-label'>📈 Gradient Boosting</div>
                    <div style='font-size: 1.8rem; font-weight: 700; color: #ffa502; margin: 8px 0;'>
                        {r2:.1f}%
                    </div>
                    <div style='font-size: 0.8rem; opacity: 0.7;'>R² Score</div>
                </div>
                """, unsafe_allow_html=True)
        
        with col4:
            # Model agreement metric
            if 'ml_ensemble_score' in data.columns:
                ml_critical = len(data[data['ml_risk_level'] == 'CRITICAL']) if 'ml_risk_level' in data.columns else 0
                st.markdown(f"""
                <div class='metric-card' style='border-left: 4px solid #764ba2; text-align: center;'>
                    <div class='stat-label'>🧠 AI Ensemble</div>
                    <div style='font-size: 1.8rem; font-weight: 700; color: #764ba2; margin: 8px 0;'>
                        {ml_critical}
                    </div>
                    <div style='font-size: 0.8rem; opacity: 0.7;'>ML-Detected Critical</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "🗺️ Geographic", "📈 Trends", "🤖 AI Models", "⚠️ Alerts"])
    
    with tab1:
        # Top row - Main visualization
        st.markdown("### 📊 Risk Intelligence Dashboard")
        
        col1, col2 = st.columns([2.5, 1.5])
        
        with col1:
            if files_count >= 2 and 'demo_total' in data.columns:
                # Enhanced scatter plot with trendline
                fig = px.scatter(data, 
                                 x='demo_total', 
                                 y='bio_total',
                                 size='gap_abs',
                                 color='anomaly_score',
                                 hover_name='district',
                                 hover_data={
                                     'state': True,
                                     'demo_total': ':,',
                                     'bio_total': ':,',
                                     'compliance_rate': ':.1f',
                                     'anomaly_score': ':.0f',
                                     'gap_abs': False
                                 },
                                 color_continuous_scale='Reds',
                                 height=450,
                                 labels={'demo_total': 'Demographic Updates', 'bio_total': 'Biometric Updates'})
                
                max_val = max(data['demo_total'].max(), data['bio_total'].max())
                fig.add_trace(go.Scatter(
                    x=[0, max_val], 
                    y=[0, max_val],
                    mode='lines',
                    name='Perfect Compliance',
                    line=dict(color='#2ed573', dash='dash', width=2),
                    showlegend=True
                ))
                
                fig.update_layout(
                    template='plotly_white', 
                    plot_bgcolor='rgba(255,255,255,0.9)', 
                    paper_bgcolor='rgba(0,0,0,0)',
                    title_font_size=16,
                    margin=dict(t=40, b=40, l=40, r=40)
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Enhanced horizontal bar chart
                fig = px.bar(data.head(20), 
                             x='anomaly_score', 
                             y='district',
                             color='risk_level',
                             orientation='h',
                             color_discrete_map={'CRITICAL': '#ff4757', 'HIGH': '#ffa502', 'NORMAL': '#2ed573'},
                             height=450,
                             labels={'anomaly_score': 'Risk Score', 'district': 'District'},
                             text='anomaly_score')
                
                fig.update_traces(texttemplate='%{text:.0f}', textposition='outside')
                fig.update_layout(
                    template='plotly_white', 
                    plot_bgcolor='rgba(255,255,255,0.9)', 
                    paper_bgcolor='rgba(0,0,0,0)',
                    title_font_size=16,
                    margin=dict(t=40, b=40, l=40, r=40)
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Risk Distribution Donut Chart
            risk_counts = data['risk_level'].value_counts()
            fig_pie = px.pie(
                values=risk_counts.values,
                names=risk_counts.index,
                color=risk_counts.index,
                color_discrete_map={'CRITICAL': '#ff4757', 'HIGH': '#ffa502', 'NORMAL': '#2ed573'},
                hole=0.5,
                height=220,
                title='Risk Level Distribution'
            )
            fig_pie.update_traces(textposition='outside', textinfo='percent+label')
            fig_pie.update_layout(
                template='plotly_white', 
                plot_bgcolor='rgba(255,255,255,0.9)', 
                paper_bgcolor='rgba(0,0,0,0)',
                showlegend=False,
                title_font_size=14,
                margin=dict(t=40, b=10, l=10, r=10)
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
            # Gauge chart for average risk
            avg_risk = data['anomaly_score'].mean()
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=avg_risk,
                title={'text': "Average Risk Score", 'font': {'size': 14}},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#667eea"},
                    'steps': [
                        {'range': [0, 30], 'color': "#e8f5e9"},
                        {'range': [30, 60], 'color': "#fff3e0"},
                        {'range': [60, 100], 'color': "#ffebee"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 60
                    }
                },
                number={'font': {'size': 32}}
            ))
            fig_gauge.update_layout(
                height=220,
                margin=dict(t=40, b=10, l=20, r=20),
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        st.markdown("---")
        
        # Second row - Detailed metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 🎯 Top 10 Risk Districts")
            top_10 = data.head(10)
            fig_top = go.Figure(go.Bar(
                x=top_10['anomaly_score'],
                y=top_10['district'],
                orientation='h',
                marker=dict(
                    color=top_10['anomaly_score'],
                    colorscale='Reds',
                    showscale=False
                ),
                text=top_10['anomaly_score'].round(0),
                textposition='outside'
            ))
            fig_top.update_layout(
                height=350,
                template='plotly_white',
                plot_bgcolor='rgba(255,255,255,0.9)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(t=10, b=10, l=10, r=40),
                xaxis_title="Risk Score",
                yaxis_title="",
                yaxis={'categoryorder': 'total ascending'}
            )
            st.plotly_chart(fig_top, use_container_width=True)
        
        with col2:
            st.markdown("#### 📈 Risk Score Distribution")
            fig_box = go.Figure()
            for risk in ['CRITICAL', 'HIGH', 'NORMAL']:
                risk_data = data[data['risk_level'] == risk]['anomaly_score']
                if len(risk_data) > 0:
                    fig_box.add_trace(go.Box(
                        y=risk_data,
                        name=risk,
                        marker_color='#ff4757' if risk == 'CRITICAL' else ('#ffa502' if risk == 'HIGH' else '#2ed573')
                    ))
            
            fig_box.update_layout(
                height=350,
                template='plotly_white',
                plot_bgcolor='rgba(255,255,255,0.9)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(t=10, b=10, l=10, r=10),
                yaxis_title="Anomaly Score",
                showlegend=True
            )
            st.plotly_chart(fig_box, use_container_width=True)
        
        with col3:
            st.markdown("#### 🗺️ State Risk Summary")
            state_risk = data.groupby('state')['anomaly_score'].mean().sort_values(ascending=False).head(10)
            
            fig_state_mini = go.Figure(go.Bar(
                x=state_risk.values,
                y=state_risk.index,
                orientation='h',
                marker=dict(
                    color=state_risk.values,
                    colorscale='YlOrRd',
                    showscale=False
                ),
                text=state_risk.values.round(1),
                textposition='outside'
            ))
            fig_state_mini.update_layout(
                height=350,
                template='plotly_white',
                plot_bgcolor='rgba(255,255,255,0.9)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(t=10, b=10, l=10, r=40),
                xaxis_title="Avg Risk Score",
                yaxis_title="",
                yaxis={'categoryorder': 'total ascending'}
            )
            st.plotly_chart(fig_state_mini, use_container_width=True)
    
    with tab2:
        st.markdown("### 🗺️ Geographic Intelligence Analysis")
        
        state_summary = data.groupby('state').agg({
            'anomaly_score': 'mean',
            'district': 'count'
        }).reset_index()
        state_summary.columns = ['State', 'Avg Risk Score', 'District Count']
        
        # Top row - Main geographic visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Treemap showing states sized by district count, colored by risk
            fig_tree = px.treemap(
                data,
                path=['state', 'district'],
                values='anomaly_score',
                color='anomaly_score',
                color_continuous_scale='RdYlGn_r',
                title='Risk Hierarchy: States → Districts',
                height=450
            )
            fig_tree.update_layout(
                template='plotly_white',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(t=40, b=10, l=10, r=10)
            )
            fig_tree.update_traces(textposition='middle center', textfont_size=11)
            st.plotly_chart(fig_tree, use_container_width=True)
        
        with col2:
            # Sunburst chart for hierarchical view
            fig_sun = px.sunburst(
                data.head(50),
                path=['risk_level', 'state', 'district'],
                values='anomaly_score',
                color='anomaly_score',
                color_continuous_scale='Reds',
                title='Hierarchical Risk View',
                height=450
            )
            fig_sun.update_layout(
                template='plotly_white',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(t=40, b=10, l=10, r=10)
            )
            st.plotly_chart(fig_sun, use_container_width=True)
        
        st.markdown("---")
        
        # Second row - Detailed state analytics
        col1, col2 = st.columns(2)
        
        with col1:
            # Enhanced state risk bar chart with gradient
            fig_state = go.Figure()
            top_states = state_summary.sort_values('Avg Risk Score', ascending=False).head(15)
            
            fig_state.add_trace(go.Bar(
                x=top_states['Avg Risk Score'],
                y=top_states['State'],
                orientation='h',
                marker=dict(
                    color=top_states['Avg Risk Score'],
                    colorscale='Reds',
                    showscale=True,
                    colorbar=dict(title="Risk Score")
                ),
                text=top_states['Avg Risk Score'].round(1),
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>Risk Score: %{x:.1f}<extra></extra>'
            ))
            
            fig_state.update_layout(
                title='Top 15 States by Average Risk Score',
                height=450,
                template='plotly_white',
                plot_bgcolor='rgba(255,255,255,0.9)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(t=50, b=40, l=10, r=40),
                xaxis_title="Average Risk Score",
                yaxis_title="",
                yaxis={'categoryorder': 'total ascending'}
            )
            st.plotly_chart(fig_state, use_container_width=True)
        
        with col2:
            # Scatter plot: District count vs Risk score
            fig_scatter_state = px.scatter(
                state_summary,
                x='District Count',
                y='Avg Risk Score',
                size='District Count',
                color='Avg Risk Score',
                text='State',
                color_continuous_scale='RdYlGn_r',
                title='State Risk vs District Density',
                height=450
            )
            fig_scatter_state.update_traces(
                textposition='top center',
                textfont_size=9
            )
            fig_scatter_state.update_layout(
                template='plotly_white',
                plot_bgcolor='rgba(255,255,255,0.9)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(t=50, b=40, l=40, r=40)
            )
            st.plotly_chart(fig_scatter_state, use_container_width=True)
        
        st.markdown("---")
        
        # Third row - Additional insights
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Districts per state bubble chart
            top_district_states = state_summary.nlargest(10, 'District Count')
            fig_bubble = px.scatter(
                top_district_states,
                x='State',
                y='District Count',
                size='District Count',
                color='Avg Risk Score',
                color_continuous_scale='Reds',
                title='Top 10 States by Districts',
                height=300
            )
            fig_bubble.update_layout(
                template='plotly_white',
                plot_bgcolor='rgba(255,255,255,0.9)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(t=40, b=40, l=40, r=10),
                xaxis={'tickangle': 45}
            )
            st.plotly_chart(fig_bubble, use_container_width=True)
        
        with col2:
            # Risk level distribution by top states
            top_5_states = state_summary.nlargest(5, 'Avg Risk Score')['State'].tolist()
            risk_state_data = data[data['state'].isin(top_5_states)]
            
            fig_risk_dist = px.histogram(
                risk_state_data,
                x='state',
                color='risk_level',
                title='Risk Distribution - Top 5 States',
                color_discrete_map={'CRITICAL': '#ff4757', 'HIGH': '#ffa502', 'NORMAL': '#2ed573'},
                barmode='group',
                height=300
            )
            fig_risk_dist.update_layout(
                template='plotly_white',
                plot_bgcolor='rgba(255,255,255,0.9)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(t=40, b=40, l=40, r=10),
                xaxis={'tickangle': 45},
                xaxis_title="State",
                yaxis_title="Count"
            )
            st.plotly_chart(fig_risk_dist, use_container_width=True)
        
        with col3:
            # State statistics table
            st.markdown("#### 📊 State Statistics")
            stats_df = state_summary.nlargest(10, 'Avg Risk Score')[['State', 'Avg Risk Score', 'District Count']]
            stats_df['Avg Risk Score'] = stats_df['Avg Risk Score'].round(1)
            
            st.dataframe(
                stats_df,
                hide_index=True,
                use_container_width=True,
                height=280
            )
    
    with tab3:
        st.markdown("### 📈 Statistical Trends & Distribution Analysis")
        
        # Top row - Main distribution charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Violin plot for anomaly score by risk level
            fig_violin = go.Figure()
            for risk in ['NORMAL', 'HIGH', 'CRITICAL']:
                risk_data = data[data['risk_level'] == risk]['anomaly_score']
                if len(risk_data) > 0:
                    fig_violin.add_trace(go.Violin(
                        y=risk_data,
                        name=risk,
                        box_visible=True,
                        meanline_visible=True,
                        fillcolor='rgba(255,71,87,0.5)' if risk == 'CRITICAL' else ('rgba(255,165,2,0.5)' if risk == 'HIGH' else 'rgba(46,213,115,0.5)'),
                        line_color='#ff4757' if risk == 'CRITICAL' else ('#ffa502' if risk == 'HIGH' else '#2ed573')
                    ))
            
            fig_violin.update_layout(
                title='Anomaly Score Distribution by Risk Level',
                height=400,
                template='plotly_white',
                plot_bgcolor='rgba(255,255,255,0.9)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(t=50, b=40, l=40, r=40),
                yaxis_title="Anomaly Score",
                showlegend=True
            )
            st.plotly_chart(fig_violin, use_container_width=True)
        
        with col2:
            # Enhanced histogram with KDE overlay
            if 'compliance_rate' in data.columns:
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(
                    x=data['compliance_rate'],
                    nbinsx=25,
                    name='Compliance Rate',
                    marker_color='#667eea',
                    opacity=0.7
                ))
                
                fig_hist.update_layout(
                    title='Compliance Rate Distribution',
                    height=400,
                    template='plotly_white',
                    plot_bgcolor='rgba(255,255,255,0.9)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    margin=dict(t=50, b=40, l=40, r=40),
                    xaxis_title="Compliance Rate (%)",
                    yaxis_title="Frequency",
                    showlegend=False
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            else:
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(
                    x=data['youth_ratio'],
                    nbinsx=25,
                    name='Youth Ratio',
                    marker_color='#667eea',
                    opacity=0.7
                ))
                
                fig_hist.update_layout(
                    title='Youth Ratio Distribution',
                    height=400,
                    template='plotly_white',
                    plot_bgcolor='rgba(255,255,255,0.9)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    margin=dict(t=50, b=40, l=40, r=40),
                    xaxis_title="Youth Ratio (%)",
                    yaxis_title="Frequency",
                    showlegend=False
                )
                st.plotly_chart(fig_hist, use_container_width=True)
        
        st.markdown("---")
        
        # Second row - Correlation and advanced analytics
        col1, col2 = st.columns(2)
        
        with col1:
            # Multi-metric box plot comparison
            if files_count >= 2 and 'compliance_rate' in data.columns:
                metrics_data = []
                for metric in ['anomaly_score', 'compliance_rate']:
                    for val in data[metric]:
                        metrics_data.append({'Metric': metric.replace('_', ' ').title(), 'Value': val})
                
                metrics_df = pd.DataFrame(metrics_data)
                fig_multi_box = px.box(
                    metrics_df,
                    x='Metric',
                    y='Value',
                    color='Metric',
                    title='Key Metrics Distribution Comparison',
                    height=400,
                    color_discrete_map={
                        'Anomaly Score': '#ff4757',
                        'Compliance Rate': '#667eea'
                    }
                )
            else:
                fig_multi_box = px.box(
                    data,
                    y='anomaly_score',
                    color='risk_level',
                    title='Anomaly Score by Risk Category',
                    height=400,
                    color_discrete_map={'CRITICAL': '#ff4757', 'HIGH': '#ffa502', 'NORMAL': '#2ed573'}
                )
            
            fig_multi_box.update_layout(
                template='plotly_white',
                plot_bgcolor='rgba(255,255,255,0.9)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(t=50, b=40, l=40, r=40)
            )
            st.plotly_chart(fig_multi_box, use_container_width=True)
        
        with col2:
            # Ridge plot style - multiple distributions
            fig_ridge = go.Figure()
            
            percentiles = [25, 50, 75]
            for i, pct in enumerate(percentiles):
                threshold = data['anomaly_score'].quantile(pct / 100)
                subset = data[data['anomaly_score'] <= threshold]
                
                fig_ridge.add_trace(go.Violin(
                    x=subset['anomaly_score'],
                    name=f'{pct}th Percentile',
                    orientation='h',
                    side='positive',
                    line_color=['#2ed573', '#ffa502', '#ff4757'][i]
                ))
            
            fig_ridge.update_layout(
                title='Anomaly Score Percentile Distribution',
                height=400,
                template='plotly_white',
                plot_bgcolor='rgba(255,255,255,0.9)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(t=50, b=40, l=40, r=40),
                xaxis_title="Anomaly Score",
                yaxis_title="Percentile Group"
            )
            st.plotly_chart(fig_ridge, use_container_width=True)
        
        # Correlation heatmap (if multiple datasets)
        if files_count >= 2:
            st.markdown("---")
            st.markdown("#### 🔗 Correlation Analysis")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                numeric_cols = ['enrol_total', 'demo_total', 'bio_total', 'compliance_rate', 'anomaly_score']
                corr_data = data[numeric_cols].corr()
                
                fig_heatmap = go.Figure(data=go.Heatmap(
                    z=corr_data.values,
                    x=[col.replace('_', ' ').title() for col in corr_data.columns],
                    y=[col.replace('_', ' ').title() for col in corr_data.index],
                    colorscale='RdBu_r',
                    zmid=0,
                    text=corr_data.values.round(2),
                    texttemplate='%{text}',
                    textfont={"size": 12},
                    colorbar=dict(title="Correlation")
                ))
                
                fig_heatmap.update_layout(
                    title='Metric Correlation Matrix',
                    height=400,
                    template='plotly_white',
                    plot_bgcolor='rgba(255,255,255,0.9)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    margin=dict(t=50, b=80, l=80, r=40)
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)
            
            with col2:
                # Statistical summary table
                st.markdown("#### 📊 Statistical Summary")
                summary_stats = data[['anomaly_score', 'compliance_rate']].describe().round(2)
                summary_stats.index = ['Count', 'Mean', 'Std Dev', 'Min', '25%', '50%', '75%', 'Max']
                st.dataframe(summary_stats, use_container_width=True, height=360)
        
        st.markdown("---")
        
        # Third row - Time-based or sequential analysis
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Cumulative distribution
            sorted_scores = np.sort(data['anomaly_score'])
            cumulative = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores) * 100
            
            fig_cdf = go.Figure()
            fig_cdf.add_trace(go.Scatter(
                x=sorted_scores,
                y=cumulative,
                mode='lines',
                fill='tozeroy',
                line=dict(color='#667eea', width=2),
                name='CDF'
            ))
            
            fig_cdf.update_layout(
                title='Cumulative Distribution',
                height=300,
                template='plotly_white',
                plot_bgcolor='rgba(255,255,255,0.9)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(t=40, b=40, l=40, r=10),
                xaxis_title="Anomaly Score",
                yaxis_title="Cumulative %"
            )
            st.plotly_chart(fig_cdf, use_container_width=True)
        
        with col2:
            # Score range analysis
            bins = [0, 30, 60, 100]
            labels = ['Low', 'Medium', 'High']
            data_copy = data.copy()
            data_copy['Score Range'] = pd.cut(data_copy['anomaly_score'], bins=bins, labels=labels)
            range_counts = data_copy['Score Range'].value_counts()
            
            fig_range = go.Figure(go.Bar(
                x=range_counts.index,
                y=range_counts.values,
                marker_color=['#2ed573', '#ffa502', '#ff4757'],
                text=range_counts.values,
                textposition='outside'
            ))
            
            fig_range.update_layout(
                title='Score Range Distribution',
                height=300,
                template='plotly_white',
                plot_bgcolor='rgba(255,255,255,0.9)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(t=40, b=40, l=40, r=10),
                xaxis_title="Score Range",
                yaxis_title="Count"
            )
            st.plotly_chart(fig_range, use_container_width=True)
        
        with col3:
            # Outlier detection visualization
            Q1 = data['anomaly_score'].quantile(0.25)
            Q3 = data['anomaly_score'].quantile(0.75)
            IQR = Q3 - Q1
            outliers = data[(data['anomaly_score'] < (Q1 - 1.5 * IQR)) | (data['anomaly_score'] > (Q3 + 1.5 * IQR))]
            
            fig_outlier = go.Figure()
            fig_outlier.add_trace(go.Box(
                y=data['anomaly_score'],
                name='All Data',
                marker_color='#667eea',
                boxpoints='outliers'
            ))
            
            fig_outlier.update_layout(
                title=f'Outlier Detection ({len(outliers)} outliers)',
                height=300,
                template='plotly_white',
                plot_bgcolor='rgba(255,255,255,0.9)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(t=40, b=40, l=40, r=10),
                yaxis_title="Anomaly Score",
                showlegend=False
            )
            st.plotly_chart(fig_outlier, use_container_width=True)
    
    with tab4:
        st.markdown("### 🤖 AI-Powered Intelligence Models")
        
        if st.session_state.ml_results:
            ml_results = st.session_state.ml_results
            
            # Model Overview Cards
            st.markdown("#### 🧠 Multi-Model Ensemble Architecture")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                <div class='metric-card' style='border-left: 4px solid #667eea;'>
                    <h4 style='color: #667eea; margin-bottom: 10px;'>🔍 Isolation Forest</h4>
                    <p style='font-size: 0.9rem; line-height: 1.6;'>
                        <b>Type:</b> Unsupervised Learning<br>
                        <b>Purpose:</b> Anomaly Detection<br>
                        <b>Method:</b> Isolates outliers in high-dimensional space<br>
                        <b>Strength:</b> Finds unknown fraud patterns
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class='metric-card' style='border-left: 4px solid #2ed573;'>
                    <h4 style='color: #2ed573; margin-bottom: 10px;'>🎯 Random Forest</h4>
                    <p style='font-size: 0.9rem; line-height: 1.6;'>
                        <b>Type:</b> Supervised Classification<br>
                        <b>Purpose:</b> Risk Level Prediction<br>
                        <b>Method:</b> Ensemble of decision trees<br>
                        <b>Strength:</b> High accuracy, interpretable
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                <div class='metric-card' style='border-left: 4px solid #ffa502;'>
                    <h4 style='color: #ffa502; margin-bottom: 10px;'>📈 Gradient Boosting</h4>
                    <p style='font-size: 0.9rem; line-height: 1.6;'>
                        <b>Type:</b> Supervised Regression<br>
                        <b>Purpose:</b> Risk Score Forecasting<br>
                        <b>Method:</b> Sequential error correction<br>
                        <b>Strength:</b> Precise numerical predictions
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Feature Importance Analysis
            if 'random_forest' in ml_results:
                st.markdown("#### 🎯 Feature Importance Analysis")
                st.markdown("*Understanding which factors drive risk predictions*")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    feat_imp = ml_results['random_forest']['feature_importance']
                    
                    fig_importance = go.Figure(go.Bar(
                        x=feat_imp['importance'],
                        y=feat_imp['feature'],
                        orientation='h',
                        marker=dict(
                            color=feat_imp['importance'],
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(title="Importance")
                        ),
                        text=feat_imp['importance'].round(3),
                        textposition='outside'
                    ))
                    
                    fig_importance.update_layout(
                        title='Random Forest Feature Importance',
                        height=400,
                        template='plotly_white',
                        plot_bgcolor='rgba(255,255,255,0.9)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        margin=dict(t=50, b=40, l=150, r=40),
                        xaxis_title="Importance Score",
                        yaxis_title="",
                        yaxis={'categoryorder': 'total ascending'}
                    )
                    st.plotly_chart(fig_importance, use_container_width=True)
                
                with col2:
                    st.markdown("**📊 Key Insights:**")
                    for idx, row in feat_imp.head(5).iterrows():
                        importance_pct = row['importance'] * 100
                        st.markdown(f"""
                        <div style='background: rgba(255,255,255,0.6); padding: 10px; border-radius: 6px; margin: 8px 0; border-left: 3px solid #667eea;'>
                            <b>{row['feature'].replace('_', ' ').title()}</b><br>
                            <span style='color: #667eea; font-size: 1.2rem; font-weight: 700;'>{importance_pct:.1f}%</span>
                        </div>
                        """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # ML vs Rule-based Comparison
            st.markdown("#### ⚖️ ML Predictions vs Rule-Based Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Scatter: Rule-based vs ML
                if 'ml_ensemble_score' in data.columns:
                    fig_comparison = px.scatter(
                        data.head(100),
                        x='anomaly_score',
                        y='ml_ensemble_score',
                        color='risk_level',
                        size='ml_confidence' if 'ml_confidence' in data.columns else None,
                        hover_name='district',
                        color_discrete_map={'CRITICAL': '#ff4757', 'HIGH': '#ffa502', 'NORMAL': '#2ed573'},
                        title='Rule-Based vs AI Ensemble Score',
                        labels={'anomaly_score': 'Rule-Based Score', 'ml_ensemble_score': 'AI Ensemble Score'},
                        height=400
                    )
                    
                    # Add diagonal line
                    fig_comparison.add_trace(go.Scatter(
                        x=[0, 100],
                        y=[0, 100],
                        mode='lines',
                        name='Perfect Agreement',
                        line=dict(color='gray', dash='dash')
                    ))
                    
                    fig_comparison.update_layout(
                        template='plotly_white',
                        plot_bgcolor='rgba(255,255,255,0.9)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        margin=dict(t=50, b=40, l=40, r=40)
                    )
                    st.plotly_chart(fig_comparison, use_container_width=True)
            
            with col2:
                # Model agreement analysis
                if 'ml_risk_level' in data.columns:
                    agreement = (data['risk_level'] == data['ml_risk_level']).sum() / len(data) * 100
                    
                    st.markdown(f"""
                    <div class='metric-card' style='border-left: 4px solid #764ba2; text-align: center;'>
                        <div class='stat-label'>Model Agreement Rate</div>
                        <div class='big-stat' style='color: #764ba2; font-size: 3rem;'>{agreement:.1f}%</div>
                        <p style='font-size: 0.9rem; margin-top: 10px;'>
                            ML models agree with rule-based classification in {agreement:.0f}% of cases
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Confusion-style comparison
                    comparison_data = []
                    for rb_level in ['NORMAL', 'HIGH', 'CRITICAL']:
                        for ml_level in ['NORMAL', 'HIGH', 'CRITICAL']:
                            count = len(data[(data['risk_level'] == rb_level) & (data['ml_risk_level'] == ml_level)])
                            comparison_data.append({
                                'Rule-Based': rb_level,
                                'ML Model': ml_level,
                                'Count': count
                            })
                    
                    comp_df = pd.DataFrame(comparison_data)
                    comp_pivot = comp_df.pivot(index='Rule-Based', columns='ML Model', values='Count')
                    
                    fig_heatmap = go.Figure(data=go.Heatmap(
                        z=comp_pivot.values,
                        x=comp_pivot.columns,
                        y=comp_pivot.index,
                        colorscale='Blues',
                        text=comp_pivot.values,
                        texttemplate='%{text}',
                        textfont={"size": 14},
                        colorbar=dict(title="Count")
                    ))
                    
                    fig_heatmap.update_layout(
                        title='Classification Agreement Matrix',
                        height=280,
                        template='plotly_white',
                        paper_bgcolor='rgba(0,0,0,0)',
                        margin=dict(t=50, b=40, l=80, r=40),
                        xaxis_title="ML Model",
                        yaxis_title="Rule-Based"
                    )
                    st.plotly_chart(fig_heatmap, use_container_width=True)
            
            st.markdown("---")
            
            # ML Confidence Analysis
            st.markdown("#### 🎲 AI Confidence Distribution")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if 'ml_confidence' in data.columns:
                    fig_conf = px.histogram(
                        data,
                        x='ml_confidence',
                        color='risk_level',
                        nbins=30,
                        title='Anomaly Detection Confidence',
                        color_discrete_map={'CRITICAL': '#ff4757', 'HIGH': '#ffa502', 'NORMAL': '#2ed573'},
                        height=320
                    )
                    fig_conf.update_layout(
                        template='plotly_white',
                        plot_bgcolor='rgba(255,255,255,0.9)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        margin=dict(t=40, b=40, l=40, r=10),
                        xaxis_title="ML Confidence Score"
                    )
                    st.plotly_chart(fig_conf, use_container_width=True)
            
            with col2:
                if 'ml_critical_probability' in data.columns:
                    fig_prob = px.box(
                        data,
                        x='risk_level',
                        y='ml_critical_probability',
                        color='risk_level',
                        title='Critical Risk Probability',
                        color_discrete_map={'CRITICAL': '#ff4757', 'HIGH': '#ffa502', 'NORMAL': '#2ed573'},
                        height=320
                    )
                    fig_prob.update_layout(
                        template='plotly_white',
                        plot_bgcolor='rgba(255,255,255,0.9)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        margin=dict(t=40, b=40, l=40, r=10),
                        yaxis_title="Probability (%)"
                    )
                    st.plotly_chart(fig_prob, use_container_width=True)
            
            with col3:
                if 'ml_predicted_risk' in data.columns:
                    # Prediction accuracy
                    actual = data['anomaly_score']
                    predicted = data['ml_predicted_risk']
                    
                    fig_pred = go.Figure()
                    fig_pred.add_trace(go.Scatter(
                        x=actual,
                        y=predicted,
                        mode='markers',
                        marker=dict(
                            size=5,
                            color=data['ml_confidence'] if 'ml_confidence' in data.columns else '#667eea',
                            colorscale='Viridis',
                            showscale=True
                        ),
                        name='Predictions'
                    ))
                    fig_pred.add_trace(go.Scatter(
                        x=[0, 100],
                        y=[0, 100],
                        mode='lines',
                        line=dict(color='red', dash='dash'),
                        name='Perfect Prediction'
                    ))
                    
                    fig_pred.update_layout(
                        title='Prediction Accuracy',
                        height=320,
                        template='plotly_white',
                        plot_bgcolor='rgba(255,255,255,0.9)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        margin=dict(t=40, b=40, l=40, r=10),
                        xaxis_title="Actual Risk",
                        yaxis_title="Predicted Risk"
                    )
                    st.plotly_chart(fig_pred, use_container_width=True)
            
            st.markdown("---")
            
            # Top ML-identified anomalies
            st.markdown("#### 🚨 Top ML-Identified Anomalies")
            
            if 'ml_confidence' in data.columns:
                top_ml = data.nlargest(10, 'ml_confidence')
                
                fig_top_ml = go.Figure()
                
                fig_top_ml.add_trace(go.Bar(
                    name='Rule-Based Score',
                    x=top_ml['district'],
                    y=top_ml['anomaly_score'],
                    marker_color='#667eea'
                ))
                
                if 'ml_ensemble_score' in top_ml.columns:
                    fig_top_ml.add_trace(go.Bar(
                        name='AI Ensemble Score',
                        x=top_ml['district'],
                        y=top_ml['ml_ensemble_score'],
                        marker_color='#ff4757'
                    ))
                
                fig_top_ml.update_layout(
                    title='Top 10 ML-Detected Anomalies: Score Comparison',
                    height=400,
                    barmode='group',
                    template='plotly_white',
                    plot_bgcolor='rgba(255,255,255,0.9)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    margin=dict(t=50, b=80, l=40, r=40),
                    xaxis={'tickangle': 45},
                    yaxis_title="Risk Score"
                )
                st.plotly_chart(fig_top_ml, use_container_width=True)
        
        else:
            st.info("ML models will be trained automatically when you analyze data. Upload datasets to see AI-powered insights!")
    
    with tab5:
        st.markdown("### ⚠️ Critical Alerts & Action Dashboard")
        
        critical_districts = data[data['risk_level'] == 'CRITICAL']
        high_districts = data[data['risk_level'] == 'HIGH']
        
        # Alert summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class='metric-card' style='border-left: 4px solid #ff4757; text-align: center;'>
                <div class='stat-label'>Critical Alerts</div>
                <div class='big-stat' style='color: #ff4757; font-size: 2.5rem;'>{len(critical_districts)}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class='metric-card' style='border-left: 4px solid #ffa502; text-align: center;'>
                <div class='stat-label'>High Priority</div>
                <div class='big-stat' style='color: #ffa502; font-size: 2.5rem;'>{len(high_districts)}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            urgent_count = len(data[data['anomaly_score'] > 80]) if len(data) > 0 else 0
            st.markdown(f"""
            <div class='metric-card' style='border-left: 4px solid #e74c3c; text-align: center;'>
                <div class='stat-label'>Urgent Action</div>
                <div class='big-stat' style='color: #e74c3c; font-size: 2.5rem;'>{urgent_count}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            total_affected = len(data[data['risk_level'].isin(['CRITICAL', 'HIGH'])])
            st.markdown(f"""
            <div class='metric-card' style='border-left: 4px solid #95a5a6; text-align: center;'>
                <div class='stat-label'>Total Affected</div>
                <div class='big-stat' style='color: #2d2d2d; font-size: 2.5rem;'>{total_affected}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Priority Matrix and Timeline
        col1, col2 = st.columns([1.5, 1])
        
        with col1:
            # Priority action matrix
            st.markdown("#### 🎯 Priority Action Matrix")
            
            priority_data = data[data['risk_level'].isin(['CRITICAL', 'HIGH'])].nlargest(20, 'anomaly_score')
            
            fig_priority = go.Figure()
            
            # Create scatter plot with urgency levels
            for risk in ['CRITICAL', 'HIGH']:
                risk_subset = priority_data[priority_data['risk_level'] == risk]
                fig_priority.add_trace(go.Scatter(
                    x=risk_subset.index,
                    y=risk_subset['anomaly_score'],
                    mode='markers+text',
                    name=risk,
                    marker=dict(
                        size=15,
                        color='#ff4757' if risk == 'CRITICAL' else '#ffa502',
                        symbol='diamond' if risk == 'CRITICAL' else 'circle',
                        line=dict(width=2, color='white')
                    ),
                    text=risk_subset['district'].str[:10],
                    textposition='top center',
                    textfont=dict(size=8),
                    hovertemplate='<b>%{text}</b><br>Score: %{y:.0f}<extra></extra>'
                ))
            
            fig_priority.update_layout(
                height=350,
                template='plotly_white',
                plot_bgcolor='rgba(255,255,255,0.9)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(t=10, b=40, l=40, r=10),
                yaxis_title="Risk Score",
                xaxis_title="Priority Rank",
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_priority, use_container_width=True)
        
        with col2:
            # Alert severity breakdown
            st.markdown("#### 📊 Severity Breakdown")
            
            severity_data = []
            for risk_level in ['CRITICAL', 'HIGH', 'NORMAL']:
                count = len(data[data['risk_level'] == risk_level])
                severity_data.append({'Level': risk_level, 'Count': count})
            
            severity_df = pd.DataFrame(severity_data)
            
            fig_severity = go.Figure(go.Funnel(
                y=severity_df['Level'],
                x=severity_df['Count'],
                textposition="inside",
                textinfo="value+percent initial",
                marker=dict(color=['#ff4757', '#ffa502', '#2ed573']),
                connector={"line": {"color": "#667eea", "dash": "dot", "width": 3}}
            ))
            
            fig_severity.update_layout(
                height=350,
                template='plotly_white',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(t=10, b=10, l=10, r=10)
            )
            st.plotly_chart(fig_severity, use_container_width=True)
        
        st.markdown("---")
        
        # Detailed alerts and visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔥 Top 10 Critical Districts - Radar View")
            
            top_critical = data.nlargest(10, 'anomaly_score')
            
            fig_radar = go.Figure()
            
            for idx, row in top_critical.head(5).iterrows():
                fig_radar.add_trace(go.Scatterpolar(
                    r=[row['anomaly_score'], row.get('youth_ratio', 50), 
                       100 - row.get('compliance_rate', 50) if 'compliance_rate' in row else 50],
                    theta=['Risk Score', 'Youth Ratio', 'Non-Compliance'],
                    fill='toself',
                    name=row['district'][:15]
                ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100])
                ),
                showlegend=True,
                height=400,
                template='plotly_white',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(t=40, b=40, l=40, r=40)
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        
        with col2:
            st.markdown("#### 📍 Alert Distribution by State")
            
            state_alerts = data[data['risk_level'].isin(['CRITICAL', 'HIGH'])].groupby(['state', 'risk_level']).size().reset_index(name='count')
            
            fig_state_alert = px.bar(
                state_alerts.nlargest(20, 'count'),
                x='count',
                y='state',
                color='risk_level',
                orientation='h',
                color_discrete_map={'CRITICAL': '#ff4757', 'HIGH': '#ffa502'},
                barmode='stack',
                height=400
            )
            
            fig_state_alert.update_layout(
                template='plotly_white',
                plot_bgcolor='rgba(255,255,255,0.9)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(t=40, b=40, l=10, r=40),
                xaxis_title="Alert Count",
                yaxis_title="",
                legend_title="Alert Level",
                yaxis={'categoryorder': 'total ascending'}
            )
            st.plotly_chart(fig_state_alert, use_container_width=True)
        
        st.markdown("---")
        st.markdown("#### 🚨 Critical Alert Details")
        
        # Detailed expandable alerts
        critical_list = data[data['risk_level'] == 'CRITICAL'].head(10)
        
        if len(critical_list) > 0:
            for idx, row in critical_list.iterrows():
                # Use simple boolean check instead of numpy bool
                is_first = (idx == critical_list.index[0])
                with st.expander(f"🔴 {row['district']}, {row['state']} - Risk Score: {row['anomaly_score']:.0f}", expanded=bool(is_first)):
                    # Visual indicator bar
                    risk_pct = min(row['anomaly_score'], 100)
                    st.markdown(f"""
                    <div style='background: linear-gradient(90deg, #ff4757 0%, #ff4757 {risk_pct}%, #e0e0e0 {risk_pct}%, #e0e0e0 100%); 
                                height: 8px; border-radius: 4px; margin-bottom: 15px;'></div>
                    """, unsafe_allow_html=True)
                    
                    if files_count >= 2:
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("📊 Demo Updates", f"{int(row.get('demo_total', 0)):,}")
                        c2.metric("🔐 Bio Updates", f"{int(row.get('bio_total', 0)):,}")
                        c3.metric("✅ Compliance", f"{row.get('compliance_rate', 0):.1f}%")
                        c4.metric("⚠️ Gap", f"{int(row.get('gap', 0)):,}")
                        
                        st.markdown("---")
                        st.markdown("**🎯 Recommended Actions:**")
                        
                        actions = []
                        if row.get('gap', 0) > 5000:
                            actions.append(f"🚨 **URGENT**: Deploy mobile biometric units - {int(row['gap']):,} pending verifications")
                        if row.get('compliance_rate', 100) < 50:
                            actions.append("📢 Launch immediate biometric awareness campaign")
                        if row.get('compliance_rate', 100) < 80:
                            actions.append("🎓 Conduct compliance training for local staff")
                        if row.get('migration_index', 0) > 2:
                            actions.append("🔍 Investigate high address change frequency patterns")
                        
                        for action in actions:
                            st.markdown(f"- {action}")
                        
                        if not actions:
                            st.markdown("- 📋 Monitor situation and maintain current protocols")
                        
                    else:
                        c1, c2, c3 = st.columns(3)
                        c1.metric("📈 Total Updates", f"{int(row.get('total_updates', 0)):,}")
                        c2.metric("👥 Youth Ratio", f"{row.get('youth_ratio', 0):.1f}%")
                        c3.metric("📝 Records", f"{int(row.get('record_count', 0)):,}")
                        
                        st.markdown("---")
                        st.markdown("**📋 Observations:**")
                        
                        if row.get('youth_ratio', 50) > 70:
                            st.markdown("- 👶 Unusually high youth enrollment detected (possible migration hub)")
                        if row.get('record_count', 0) > 100:
                            st.markdown("- 📊 High transaction frequency - requires monitoring")
                        if row.get('total_updates', 0) > 10000:
                            st.markdown("- ⚡ Exceptional activity volume - verify legitimacy")
        else:
            # Check for HIGH risk districts
            high_districts = data[data['risk_level'] == 'HIGH']
            
            if len(high_districts) > 0:
                st.warning(f"⚠️ No CRITICAL alerts, but {len(high_districts)} districts are at HIGH risk. Monitor closely.")
                
                # Show top 5 HIGH risk districts
                st.markdown("#### Top HIGH Risk Districts:")
                for idx, row in high_districts.head(5).iterrows():
                    st.markdown(f"""
                    <div style='background: rgba(255,165,2,0.1); padding: 12px; border-radius: 8px; margin: 8px 0; border-left: 4px solid #ffa502;'>
                        <b style='color: #1a1a1a;'>{row['district']}, {row['state']}</b><br>
                        <span style='color: #ffa502; font-weight: 600;'>Risk Score: {row['anomaly_score']:.0f}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("✅ No critical or high-risk alerts at this time. All districts operating within normal parameters.")
        
        # Export section
        st.markdown("---")
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown("**📥 Export Options**")
        
        with col2:
            csv = data.to_csv(index=False)
            st.download_button(
                "📄 Full Report (CSV)", 
                csv, 
                f"sentinel_full_report_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", 
                "text/csv", 
                use_container_width=True
            )
        
        with col3:
            critical_csv = data[data['risk_level'] == 'CRITICAL'].to_csv(index=False)
            st.download_button(
                "🚨 Critical Only", 
                critical_csv, 
                f"sentinel_critical_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", 
                "text/csv", 
                use_container_width=True
            )

else:
    # Welcome screen
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    features = [
        ("🤖 AI-Powered", "Multi-model ML ensemble with 3 algorithms"),
        ("⚡ Adaptive Analysis", "Automatically detects patterns in 1-3 datasets"),
        ("📊 Rich Visualizations", "20+ interactive charts and insights")
    ]
    
    for idx, (title, desc) in enumerate(features):
        with [col1, col2, col3][idx]:
            st.markdown(f"""
            <div class='metric-card' style='text-align: center; min-height: 180px;'>
                <h3 style='margin-bottom: 15px; font-size: 1.5rem;'>{title}</h3>
                <p style='opacity: 0.8; font-size: 1rem;'>{desc}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.info("👆 Upload your Aadhaar datasets using the file uploader above to begin analysis", icon="💡")