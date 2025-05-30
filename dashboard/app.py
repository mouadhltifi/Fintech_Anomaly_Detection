import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import sys
import datetime
from PIL import Image

# Add the parent directory to sys.path to import utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import utils
from dashboard.utils.model_loader import ModelManager
from dashboard.utils.visualizations import (
    create_price_prediction_chart, 
    create_model_comparison_chart, 
    create_heatmap_periods,
    create_financial_dashboard,
    create_single_day_prediction_card,
    create_probability_trend_chart
)

# Set page config
st.set_page_config(
    page_title="Financial Crisis EWS",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# After setting page config, add this custom CSS to fix white backgrounds
st.markdown("""
<style>
    /* Fix white backgrounds behind tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: transparent !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: auto;
        white-space: pre-wrap;
        background-color: transparent !important;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: rgba(255, 255, 255, 0.1) !important;
        border-bottom: 2px solid #4682B4;
    }
    
    /* Fix white backgrounds behind plotly charts */
    .js-plotly-plot .plotly {
        background-color: transparent !important;
    }
    
    /* Remove white background from all st elements */
    .stButton button, div.stButton > button:first-child,
    .stTextInput > div > div > input,
    .stDateInput > div > div > input,
    .stSelectbox > div > div,
    .stMultiselect > div > div,
    div[data-testid="stTickBarMin"], div[data-testid="stTickBarMax"], 
    div[data-testid="stTickBar"], div[data-testid="stTickBarDiv"],
    div[data-testid="stExpander"] {
        background-color: transparent !important;
    }
    
    /* Better contrast for text on dark mode */
    .dark-mode p, .dark-mode h1, .dark-mode h2, .dark-mode h3, .dark-mode h4, .dark-mode h5, .dark-mode h6 {
        color: rgba(250, 250, 250, 0.95) !important;
    }
    
    /* Make metrics transparent */
    div[data-testid="metric-container"] {
        background-color: transparent !important;
    }
    
    /* Remove background from all cards and containers */
    .element-container, .stDataFrame, 
    div[data-testid="stVerticalBlock"] > div {
        background-color: transparent !important;
    }
</style>
""", unsafe_allow_html=True)

# Function to load CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Try to load CSS file if it exists
css_path = os.path.join(os.path.dirname(__file__), 'assets/styles.css')
if os.path.exists(css_path):
    local_css(css_path)

# Initialize session state for model manager to avoid reloading on every interaction
if 'model_manager' not in st.session_state:
    st.session_state.model_manager = None
    st.session_state.datasets_loaded = False
    st.session_state.selected_tab = "Dashboard"
    st.session_state.dark_mode = False

# Header area with logo and title in a container with custom styling
header_container = st.container()
with header_container:
    col1, col2 = st.columns([1, 5])
    with col1:
        st.image(
            "https://cdn-icons-png.flaticon.com/512/6295/6295417.png", 
            width=80
        )
    with col2:
        st.markdown("""
        <div class="header-text">
            <h1>Financial Crisis Early Warning System</h1>
            <h3>Market stress prediction through machine learning</h3>
        </div>
        """, unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.sidebar.markdown("""
    <div class="sidebar-header">
        <h2>📈 EWS Controls</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Dark/Light mode toggle
    dark_mode = st.checkbox("Dark Mode", value=st.session_state.dark_mode)
    if dark_mode != st.session_state.dark_mode:
        st.session_state.dark_mode = dark_mode
        if dark_mode:
            st.markdown("""
            <style>
                :root {
                    --background-color: #121212;
                    --text-color: #f0f0f0;
                    --card-bg: #1e1e1e;
                    --sidebar-bg: #1a1a1a;
                }
                .stApp {
                    background-color: var(--background-color);
                    color: var(--text-color);
                }
                .css-1kyxreq, div.stButton > button {
                    background-color: #2c3e50;
                    color: white;
                }
                .css-1kyxreq:hover, div.stButton > button:hover {
                    background-color: #34495e;
                    color: white;
                }
                .card {
                    background-color: var(--card-bg);
                    border: 1px solid #333;
                }
                h1, h2, h3, h4 {
                    color: #3498db !important;
                }
                .sidebar .sidebar-content {
                    background-color: var(--sidebar-bg);
                }
                /* Dark mode specific styles for day selector */
                [data-testid="stDateInput"] {
                    background-color: #1e1e1e;
                    border-radius: 5px;
                    padding: 2px;
                    border: 1px solid #333;
                }
                /* Date notification banner in dark mode */
                .date-notification-dark {
                    margin-top: 34px; 
                    padding: 10px; 
                    border-left: 4px solid #38bdf8; 
                    background-color: rgba(56, 189, 248, 0.15);
                }
                /* Risk alerts in dark mode */
                .high-risk-alert-dark {
                    background-color: rgba(239, 68, 68, 0.2);
                    border-left: 5px solid #ef4444;
                }
                .medium-risk-alert-dark {
                    background-color: rgba(245, 158, 11, 0.2);
                    border-left: 5px solid #f59e0b;
                }
                .low-risk-alert-dark {
                    background-color: rgba(16, 185, 129, 0.2);
                    border-left: 5px solid #10b981;
                }
            </style>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <style>
                /* Revert to light mode - styles are in the CSS file */
            </style>
            """, unsafe_allow_html=True)
            if os.path.exists(css_path):
                local_css(css_path)

    # Model loading section with improved styling
    with st.expander("📥 Data Loading", expanded=True):
        st.markdown("""<div class="section-subheader">Load models and datasets</div>""", unsafe_allow_html=True)
        
        load_col1, load_col2 = st.columns(2)
        
        with load_col1:
            if st.button("🤖 Load Models", use_container_width=True):
                with st.spinner("Loading models... This might take a minute..."):
                    if st.session_state.model_manager is None:
                        # Create model manager
                        models_dir = '/Users/mouadh/Fintech_Projects/Business_Case_4/notebooks/models'
                        st.session_state.model_manager = ModelManager(models_dir=models_dir)
                st.success(f"✅ Loaded {len(st.session_state.model_manager.models_info)} models")
        
        with load_col2:
            if st.session_state.model_manager is not None and not st.session_state.datasets_loaded:
                if st.button("📊 Load Datasets", use_container_width=True):
                    with st.spinner("Loading datasets... Please wait..."):
                        data_dir = '/Users/mouadh/Fintech_Projects/Business_Case_4/data/processed/'
                        st.session_state.model_manager.load_datasets(data_dir=data_dir)
                        st.session_state.datasets_loaded = True
                    
                    st.success(f"✅ Loaded {len(st.session_state.model_manager.datasets)} datasets")
    
    # Define thresholds_info variable at a higher scope
    thresholds_info = ""
    if st.session_state.model_manager is not None and st.session_state.datasets_loaded:
        # Prepare thresholds info
        thresholds_info = ""
        for model_info in st.session_state.model_manager.models_info:
            threshold = model_info['metadata'].get('threshold', 0.5)
            model_type = model_info['model_type']
            metric = model_info['metadata'].get('metric', 'Unknown')
            thresholds_info += f"- {model_type} ({metric}): {threshold:.4f}\n"
        
        # Show the thresholds in an expander in the sidebar
        with st.sidebar.expander("📊 Model Thresholds"):
            st.markdown("""
            The following thresholds were automatically calculated for each model 
            based on optimizing their respective metrics on validation data:
            """)
            st.code(thresholds_info)
    
    # Only show the rest if models and datasets are loaded
    if st.session_state.model_manager is not None and st.session_state.datasets_loaded:
        # Add custom CSS to remove backgrounds from headings and metrics
        st.markdown("""
        <style>
            /* Remove backgrounds from headings */
            .stHeadingContainer {
                background-color: transparent !important;
            }
            
            /* Remove backgrounds from metrics */
            [data-testid="stMetric"] {
                background-color: transparent !important;
            }
            
            /* Remove extra padding from headings */
            .stHeadingContainer {
                padding-top: 0 !important;
                padding-bottom: 0 !important;
                margin-top: 1rem !important;
                margin-bottom: 0.5rem !important;
            }
            
            /* Style section headers more prominently */
            .section-header {
                font-size: 1.5rem;
                font-weight: 600;
                color: inherit;
                margin-top: 1rem;
                margin-bottom: 0.5rem;
            }
        </style>
        """, unsafe_allow_html=True)
        
        # Specific date selection - replaces the time period range
        with st.expander("📅 Simulation Date", expanded=True):
            st.markdown("""<div class="section-subheader">Select a specific date to analyze</div>""", unsafe_allow_html=True)
            
            # Default date (near the middle of available data)
            available_dates = pd.date_range(start='2000-01-01', end=datetime.date.today(), freq='B')
            default_date = available_dates[len(available_dates)//2].date()
            
            specific_date = st.date_input(
                "Simulation Date",
                value=default_date,
                min_value=datetime.date(2000, 1, 1),
                max_value=datetime.date.today()
            )
            
            # Use this specific date as both start (for loading data) and end (cutoff for analysis)
            # This simulates having data only up to this point
            earliest_date = specific_date - datetime.timedelta(days=365*3)  # 3 years before selected date for context
            start_date = earliest_date
            end_date = specific_date
            
            # Convert to string format for model manager
            start_date_str = start_date.strftime('%Y-%m-%d')
            end_date_str = end_date.strftime('%Y-%m-%d')
            specific_date_str = specific_date.strftime('%Y-%m-%d')
            
            st.caption(f"Analyzing data up to {specific_date_str}")
        
        # Store date info in session state
        st.session_state.selected_date_range = (start_date, end_date, start_date_str, end_date_str)
        st.session_state.specific_date = specific_date
        st.session_state.specific_date_str = specific_date_str
        
        # Model selection with improved styling
        with st.expander("🤖 Model Selection", expanded=True):
            st.markdown("""<div class="section-subheader">Select models to analyze</div>""", unsafe_allow_html=True)
            
            # Get list of available models
            model_names = st.session_state.model_manager.model_names
            
            # Group models by type for better organization
            model_types = {}
            for name in model_names:
                model_type = name.split(" - ")[0]
                if model_type not in model_types:
                    model_types[model_type] = []
                model_types[model_type].append(name)
            
            model_category = st.selectbox("Model Category", list(model_types.keys()))
            selected_model = st.selectbox(
                "Select Primary Model",
                options=model_types[model_category],
                index=0
            )
            
            st.markdown("### Comparison Models")
            # Default comparison selection includes the primary model plus others from same category
            default_models = [selected_model]
            if len(model_types[model_category]) > 1:
                default_models.append(model_types[model_category][1])
                
            compare_models = st.multiselect(
                "Select Models to Compare",
                options=model_names,
                default=default_models
            )
        
        # Visualization settings
        with st.expander("📊 Display Settings", expanded=True):
            st.markdown("""<div class="section-subheader">Customize visualization</div>""", unsafe_allow_html=True)
            
            # Market indicator selection - NEW FEATURE
            available_indicators = []
            
            # Find available market indicators across all datasets
            for dataset_name in st.session_state.model_manager.datasets:
                dataset = st.session_state.model_manager.datasets[dataset_name]
                
                # Look for common market indicators in dataset columns
                for col in dataset.columns:
                    col_lower = col.lower()
                    if any(term in col_lower for term in ['vix', 'index', 'price', 'market', 'sp500', 's&p', 'dji', 'djia', 'nasdaq']):
                        if col not in available_indicators:
                            available_indicators.append(col)
            
            # Default to VIX if available
            default_index = next((i for i, name in enumerate(available_indicators) if 'vix' in name.lower()), 0)
            
            selected_indicator = st.selectbox(
                "Market Indicator to Plot",
                options=available_indicators,
                index=min(default_index, len(available_indicators)-1) if available_indicators else 0,
                help="Select which market indicator to display in the chart"
            )
            
            show_full_dashboard = st.checkbox("Show Full Dashboard", value=True)

# Main content area with tabs
if st.session_state.model_manager is not None and st.session_state.datasets_loaded:
    # Add container to limit width
    max_width_style = """
    <style>
        .main-container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 1rem;
        }
        
        /* Make sidebar wider */
        [data-testid="stSidebar"] {
            min-width: 330px !important;
            width: 330px !important;
        }
    </style>
    """
    st.markdown(max_width_style, unsafe_allow_html=True)
    
    # Start main container
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Dashboard", "🔍 Model Details", "⚖️ Comparison", "📊 Data Explorer"])
    
    with tab1:
        # Main dashboard tab
        if selected_model:
            # Get model info
            model_info = st.session_state.model_manager.get_model_by_name(selected_model)
            
            # Extract date range from session state and specific date
            start_date, end_date, start_date_str, end_date_str = st.session_state.selected_date_range
            specific_date = st.session_state.specific_date
            specific_date_str = st.session_state.specific_date_str
            
            # Display current simulation date prominently
            st.markdown(f"""
            <div style="text-align: center; padding: 15px; background-color: rgba(2, 132, 199, 0.1); border-radius: 5px; margin-bottom: 20px; border-left: 5px solid #0284c7;">
                <h2 style="margin: 0; color: #0284c7;">Financial Crisis Early Warning Simulation</h2>
                <p style="font-size: 1.2rem; margin: 5px 0 0 0;">Date: <strong>{specific_date_str}</strong></p>
                <p style="font-size: 0.9rem; margin: 0;">Analysis based on data available up to this date</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Get range predictions (up to specific_date)
            predictions = st.session_state.model_manager.predict(model_info, start_date_str, specific_date_str)
            
            # Get specific day prediction
            specific_prediction = st.session_state.model_manager.predict_for_day(model_info, specific_date_str)
            
            # Get model-specific threshold
            model_threshold = 0.5  # Default fallback
            if specific_prediction and 'threshold' in specific_prediction:
                model_threshold = specific_prediction['threshold']
            elif model_info and 'metadata' in model_info and 'threshold' in model_info['metadata']:
                model_threshold = model_info['metadata']['threshold']
                
            # Get market indicator data
            if selected_indicator:
                market_data = st.session_state.model_manager.get_market_data(start_date_str, specific_date_str, selected_indicator)
            else:
                # Fall back to VIX data if no indicator selected
                market_data = st.session_state.model_manager.get_vix_data(start_date_str, specific_date_str)
            
            if predictions is not None and specific_prediction:
                # Create a layout with 3 vertical sections similar to Jupyter notebook
                
                # 1. Main chart with market data and crisis indicator
                st.markdown('<div class="section-header">Market Status and Crisis Prediction</div>', unsafe_allow_html=True)
                
                # Get actual values and status
                actual_value = None
                verification_text = ""
                if 'actual' in predictions.columns and specific_prediction['actual'] is not None:
                    actual_value = int(specific_prediction['actual'])
                    if specific_prediction['prediction'] == 1 and actual_value == 1:
                        verification_text = "✅ True Positive - Correctly Identified Crisis"
                    elif specific_prediction['prediction'] == 1 and actual_value == 0:
                        verification_text = "❌ False Positive - False Alarm"
                    elif specific_prediction['prediction'] == 0 and actual_value == 0:
                        verification_text = "✅ True Negative - Correctly Identified No Crisis"
                    else:  # pred == 0 and actual == 1
                        verification_text = "❌ False Negative - Missed Crisis"
                
                # Show crisis prediction status
                crisis_status_container = st.container()
                with crisis_status_container:
                    cols = st.columns([2, 3])
                    with cols[0]:
                        if specific_prediction['prediction'] == 1:
                            st.markdown("""
                            <div style="background-color: rgba(239, 68, 68, 0.2); padding: 15px; border-radius: 5px; border-left: 5px solid #ef4444;">
                                <h3 style="margin: 0; color: #ef4444;">⚠️ CRISIS WARNING!</h3>
                                <p style="margin: 5px 0 0 0;">Probability: {:.1f}% (Model Threshold: {:.0f}%)</p>
                                <p style="margin: 5px 0 0 0; font-size: 0.9em;">{}</p>
                            </div>
                            """.format(
                                specific_prediction['probability'] * 100,
                                model_threshold * 100,
                                verification_text
                            ), unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div style="background-color: rgba(16, 185, 129, 0.2); padding: 15px; border-radius: 5px; border-left: 5px solid #10b981;">
                                <h3 style="margin: 0; color: #10b981;">✓ NO CRISIS DETECTED</h3>
                                <p style="margin: 5px 0 0 0;">Probability: {:.1f}% (Model Threshold: {:.0f}%)</p>
                                <p style="margin: 5px 0 0 0; font-size: 0.9em;">{}</p>
                            </div>
                            """.format(
                                specific_prediction['probability'] * 100,
                                model_threshold * 100,
                                verification_text
                            ), unsafe_allow_html=True)
                            
                    with cols[1]:
                        # Show key metrics for the specific date
                        st.markdown("### Key Market Indicators")
                        
                        # Get data for the specific date
                        date_data = {}
                        
                        # Add selected indicator if available
                        if market_data is not None and specific_date in market_data.index:
                            for col in market_data.columns:
                                date_data[col] = market_data.loc[specific_date, col]
                        
                        # Get some additional metrics from the dataset if available
                        for feature_set_name in st.session_state.model_manager.datasets:
                            df = st.session_state.model_manager.datasets[feature_set_name]
                            if specific_date in df.index:
                                row = df.loc[specific_date]
                                
                                # Look for important indicators
                                for col in df.columns:
                                    if any(indicator in col.lower() for indicator in ['ted', 'spread', 'volatil', 'yield', 'return']):
                                        if col not in date_data and col not in ['Y', 'pre_crisis']:
                                            date_data[col] = row[col]
                        
                        # Display top indicators as metrics
                        metric_cols = st.columns(2)
                        for i, (name, value) in enumerate(sorted(date_data.items(), key=lambda x: abs(x[1]), reverse=True)[:4]):
                            with metric_cols[i % 2]:
                                # Format the value appropriately
                                if abs(value) < 0.1:
                                    formatted_value = f"{value:.4f}"
                                elif abs(value) < 1:
                                    formatted_value = f"{value:.3f}"
                                elif abs(value) < 10:
                                    formatted_value = f"{value:.2f}"
                                else:
                                    formatted_value = f"{value:.1f}"
                                    
                                st.metric(
                                    label=name,
                                    value=formatted_value
                                )
                
                # Add model predictions from all selected models
                st.markdown('<div class="section-header">All Model Predictions</div>', unsafe_allow_html=True)
                st.markdown("### Model Predictions for this Date")
                
                # Get predictions from all selected models for specific date
                model_predictions = []
                
                for model_name in compare_models:
                    curr_model_info = st.session_state.model_manager.get_model_by_name(model_name)
                    if curr_model_info:
                        # Get prediction for specific date
                        curr_prediction = st.session_state.model_manager.predict_for_day(curr_model_info, specific_date_str)
                        
                        if curr_prediction is not None:
                            # Extract model type and dataset
                            model_type = curr_model_info['model_type']
                            dataset = curr_model_info.get('dataset_name', '')
                            
                            # Get threshold from model metadata or use default
                            model_threshold = curr_model_info['metadata'].get('threshold', 0.5)
                            
                            # Create a simpler model name
                            simple_name = f"{model_type}_{dataset[:3]}"
                            
                            model_predictions.append({
                                'Model': model_name,
                                'ShortName': simple_name,
                                'Type': model_type,
                                'Dataset': dataset,
                                'Probability': curr_prediction['probability'],
                                'Prediction': curr_prediction['prediction'],
                                'Threshold': curr_prediction.get('threshold', model_threshold)
                            })
                
                # Show main chart with market data and crisis prediction
                chart_fig = create_price_prediction_chart(
                    market_data, 
                    predictions, 
                    threshold=model_threshold,  # Use model-specific threshold
                    highlight_date=specific_date, 
                    indicator_name=selected_indicator
                )
                st.plotly_chart(chart_fig, use_container_width=True, config={'displayModeBar': False})
                
                if model_predictions:
                    # Create a styled HTML table similar to Jupyter notebook
                    header = "<table style='width:100%; border-collapse:collapse;'>"
                    header += "<tr style='border-bottom:1px solid #ddd; background-color:rgba(241, 245, 249, 0.3);'>"
                    header += "<th style='text-align:left; padding:8px;'>Model</th>"
                    header += "<th style='text-align:center; padding:8px;'>Vote</th>"
                    header += "<th style='text-align:center; padding:8px;'>Probability</th>"
                    header += "<th style='text-align:center; padding:8px;'>Threshold</th>"
                    header += "<th style='text-align:center; padding:8px;'>Type</th>"
                    header += "<th style='text-align:left; padding:8px;'>Dataset</th>"
                    header += "</tr>"
                    
                    rows = ""
                    for model in model_predictions:
                        vote_icon = "⚠️" if model['Prediction'] == 1 else "✓"
                        vote_color = "#ef4444" if model['Prediction'] == 1 else "#10b981"
                        
                        # Determine color for probability
                        if model['Probability'] < 0.3:
                            prob_color = "#10b981"  # Green
                        elif model['Probability'] < 0.7:
                            prob_color = "#f59e0b"  # Yellow/Orange
                        else:
                            prob_color = "#ef4444"  # Red
                        
                        # Get model threshold
                        threshold = model['Threshold']
                        
                        rows += f"<tr style='border-bottom:1px solid #ddd;'>"
                        rows += f"<td style='padding:8px;'>{model['ShortName']}</td>"
                        rows += f"<td style='text-align:center; padding:8px; color:{vote_color};'>{vote_icon}</td>"
                        rows += f"<td style='text-align:center; padding:8px; color:{prob_color};'>{model['Probability']:.3f} ({int(model['Probability']*100)}%)</td>"
                        rows += f"<td style='text-align:center; padding:8px;'>{threshold:.3f}</td>"
                        rows += f"<td style='text-align:center; padding:8px;'>{model['Type']}</td>"
                        rows += f"<td style='padding:8px;'>{model['Dataset']}</td>"
                        rows += "</tr>"
                    
                    footer = "</table>"
                    
                    # Display the table
                    st.markdown(header + rows + footer, unsafe_allow_html=True)
                    
                    # Create a bar chart showing all model probabilities
                    st.markdown("### Model Probability Comparison")
                    
                    model_df = pd.DataFrame(model_predictions).sort_values(by='Probability', ascending=False)
                    
                    # Create horizontal bar chart for probabilities
                    prob_fig = go.Figure()
                    
                    # Add probability bars
                    prob_fig.add_trace(go.Bar(
                        y=model_df['ShortName'],
                        x=model_df['Probability'],
                        orientation='h',
                        name='Probability',
                        marker=dict(
                            color=[
                                '#10b981' if p < 0.3 else '#f59e0b' if p < 0.7 else '#ef4444' 
                                for p in model_df['Probability']
                            ]
                        ),
                        hovertemplate='%{y}: %{x:.1%}'
                    ))
                    
                    # For each model, add a threshold marker
                    for i, row in model_df.iterrows():
                        prob_fig.add_trace(go.Scatter(
                            x=[row['Threshold']],
                            y=[row['ShortName']],
                            mode='markers',
                            marker=dict(
                                symbol='line-ns',
                                size=12,
                                color='black',
                                line=dict(width=2)
                            ),
                            name=f"Threshold ({row['ShortName']})" if i == 0 else None,
                            showlegend=i == 0,  # Only show legend for first marker
                            hovertemplate=f"{row['ShortName']} Threshold: {row['Threshold']:.2f}"
                        ))
                        
                    # Update layout
                    prob_fig.update_layout(
                        title=None,
                        xaxis=dict(
                            title="Crisis Probability & Model Thresholds",
                            tickformat='.0%',
                            range=[0, 1]
                        ),
                        yaxis=dict(
                            title=None
                        ),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        height=max(250, len(model_df) * 30),  # Dynamic height based on number of models
                        margin=dict(l=10, r=10, t=10, b=10),
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1,
                        ),
                        hovermode="closest"
                    )
                    
                    st.plotly_chart(prob_fig, use_container_width=True, config={'displayModeBar': False})
                
                # Instead of the heatmap, show historical probability trend
                st.markdown('<div class="section-header">Historical Crisis Probability</div>', unsafe_allow_html=True)
                
                # Create monthly resampled probabilities for a smoother trend line
                # This will only include data up to the selected date
                monthly_probs = predictions.set_index('date')['probability'].resample('M').mean().reset_index()
                
                # Use model-specific threshold for the trend chart
                trend_threshold = model_threshold  # Use the same threshold we set earlier
                
                monthly_with_highlight = {
                    'dates': monthly_probs['date'],
                    'probabilities': monthly_probs['probability'],
                    'highlight_date': specific_date,
                    'threshold': trend_threshold
                }
                
                # Create custom probability trend chart
                trend_fig = create_probability_trend_chart(monthly_with_highlight)
                st.plotly_chart(trend_fig, use_container_width=True, config={'displayModeBar': False})
                
            else:
                st.error("No predictions available for the selected date and model. Try selecting a different date.")
    
    # Close main container
    st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        # Model details tab
        if selected_model:
            model_info = st.session_state.model_manager.get_model_by_name(selected_model)
            
            st.subheader(f"Model: {selected_model}")
            
            # Create two columns for model details
            detail_col1, detail_col2 = st.columns(2)
            
            with detail_col1:
                st.markdown("### Model Information")
                st.markdown(f"""
                - **Type:** {model_info['model_type']}
                - **Dataset:** {model_info['dataset_name']}
                - **Best Metric:** {model_info['metadata'].get('metric', 'Unknown')}
                - **Score:** {model_info['metadata'].get('score', 'Unknown'):.4f}
                """)
            
            with detail_col2:
                # Show feature importance if available
                st.markdown("### Top Features")
                
                if 'model' in model_info and hasattr(model_info['model'], 'feature_importances_'):
                    feature_importance = model_info['model'].feature_importances_
                    feature_names = model_info['features']
                    
                    if len(feature_names) > 0 and len(feature_importance) > 0:
                        # Create feature importance dataframe
                        fi_df = pd.DataFrame({
                            'Feature': feature_names[:len(feature_importance)],
                            'Importance': feature_importance
                        }).sort_values('Importance', ascending=False).head(10)
                        
                        # Plot feature importance
                        st.bar_chart(fi_df.set_index('Feature'))
                else:
                    st.write("Feature importance not available for this model type")
            
            # Show model performance metrics if available
            st.markdown("### Performance Metrics")
            
            # Create tabs for validation and test metrics
            metrics_tabs = st.tabs(["Validation Metrics", "Test Metrics"])
            
            with metrics_tabs[0]:
                st.caption("Metrics on validation data (2000-2018), used for threshold optimization")
                metrics_cols = st.columns(5)
                
                # Get validation metrics (stored at the top level of metadata)
                metrics = {
                    'accuracy': model_info['metadata'].get('accuracy', None),
                    'precision': model_info['metadata'].get('precision', None),
                    'recall': model_info['metadata'].get('recall', None),
                    'f1_score': model_info['metadata'].get('f1_score', None),
                    'roc_auc': model_info['metadata'].get('roc_auc', None)
                }
                
                # If we have a score, use it for the corresponding metric
                if 'metric' in model_info['metadata'] and 'score' in model_info['metadata']:
                    metric_name = model_info['metadata']['metric']
                    if metric_name in metrics and metrics[metric_name] is None:
                        metrics[metric_name] = model_info['metadata']['score']
                
                # Display the validation metrics
                for i, (metric_name, metric_value) in enumerate(metrics.items()):
                    with metrics_cols[i]:
                        if metric_value is not None:
                            st.metric(
                                label=metric_name.replace('_', ' ').title(),
                                value=f"{metric_value:.4f}"
                            )
                        else:
                            st.metric(
                                label=metric_name.replace('_', ' ').title(),
                                value="N/A"
                            )
                
                # Show optimized threshold
                st.info(f"Optimized threshold: {model_info['metadata'].get('threshold', 0.5):.4f}")
            
            with metrics_tabs[1]:
                st.caption("Metrics on test data (2019-2023), true out-of-sample performance")
                test_metrics_cols = st.columns(5)
                
                # Get test metrics (stored in nested test_metrics dict)
                test_metrics = model_info['metadata'].get('test_metrics', {})
                
                if test_metrics:
                    for i, metric_name in enumerate(['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']):
                        with test_metrics_cols[i]:
                            metric_value = test_metrics.get(metric_name)
                            if metric_value is not None:
                                st.metric(
                                    label=metric_name.replace('_', ' ').title(),
                                    value=f"{metric_value:.4f}"
                                )
                            else:
                                st.metric(
                                    label=metric_name.replace('_', ' ').title(),
                                    value="N/A"
                                )
                else:
                    st.warning("Test metrics are not available for this model.")
    
    with tab3:
        # Comparison tab
        st.subheader("Model Comparison")
        
        if len(compare_models) > 1:
            comparison_fig = create_model_comparison_chart(
                st.session_state.model_manager, 
                start_date_str, 
                end_date_str, 
                compare_models
            )
            st.plotly_chart(comparison_fig, use_container_width=True, config={'displayModeBar': True})
            
            # Create a metric comparison table
            st.subheader("Model Metrics Comparison")
            
            metrics_comparison_tabs = st.tabs(["Validation Metrics", "Test Metrics"])
            
            with metrics_comparison_tabs[0]:
                st.caption("Validation metrics (2000-2018), used to determine optimal thresholds")
                validation_metrics_df = []
                
                for model_name in compare_models:
                    model_info = st.session_state.model_manager.get_model_by_name(model_name)
                    if model_info:
                        # Get the model's metadata
                        metadata = model_info['metadata']
                        model_type = model_info['model_type']
                        dataset = model_info['dataset_name']
                        threshold = metadata.get('threshold', 0.5)
                        
                        # Get metrics from metadata
                        metrics_row = {
                            'Model': model_name,
                            'Type': model_type,
                            'Dataset': dataset,
                            'Threshold': threshold,
                            'Accuracy': metadata.get('accuracy'),
                            'Precision': metadata.get('precision'),
                            'Recall': metadata.get('recall'),
                            'F1 Score': metadata.get('f1_score'),
                            'ROC AUC': metadata.get('roc_auc')
                        }
                        
                        validation_metrics_df.append(metrics_row)
                
                if validation_metrics_df:
                    validation_comparison_table = pd.DataFrame(validation_metrics_df)
                    st.dataframe(validation_comparison_table, hide_index=True, use_container_width=True)
                else:
                    st.info("No validation metrics available for the selected models.")
            
            with metrics_comparison_tabs[1]:
                st.caption("Test metrics (2019-2023), representing true out-of-sample performance")
                test_metrics_df = []
                
                for model_name in compare_models:
                    model_info = st.session_state.model_manager.get_model_by_name(model_name)
                    if model_info:
                        # Get the model's metadata
                        metadata = model_info['metadata']
                        model_type = model_info['model_type']
                        dataset = model_info['dataset_name']
                        threshold = metadata.get('threshold', 0.5)
                        
                        # Get test metrics from metadata
                        test_metrics = metadata.get('test_metrics', {})
                        
                        if test_metrics:
                            metrics_row = {
                                'Model': model_name,
                                'Type': model_type,
                                'Dataset': dataset,
                                'Threshold': threshold,
                                'Accuracy': test_metrics.get('accuracy'),
                                'Precision': test_metrics.get('precision'),
                                'Recall': test_metrics.get('recall'),
                                'F1 Score': test_metrics.get('f1_score'),
                                'ROC AUC': test_metrics.get('roc_auc')
                            }
                            
                            test_metrics_df.append(metrics_row)
                
                if test_metrics_df:
                    test_comparison_table = pd.DataFrame(test_metrics_df)
                    st.dataframe(test_comparison_table, hide_index=True, use_container_width=True)
                else:
                    st.info("No test metrics available for the selected models.")
        else:
            st.info("Please select at least two models in the sidebar to compare them.")
    
    with tab4:
        # Data explorer tab
        st.subheader("Raw Data Explorer")
        
        dataset_names = list(st.session_state.model_manager.datasets.keys())
        selected_dataset = st.selectbox("Select Dataset", dataset_names)
        
        if selected_dataset:
            dataset = st.session_state.model_manager.datasets[selected_dataset]
            dataset_slice = dataset[start_date_str:end_date_str]
            
            if not dataset_slice.empty:
                # Show dataset statistics
                st.markdown(f"### Dataset: {selected_dataset}")
                st.write(f"Time period: {start_date_str} to {end_date_str}")
                st.write(f"Shape: {dataset_slice.shape[0]} rows, {dataset_slice.shape[1]} columns")
                
                # Column selection and filtering
                col_search = st.text_input("Search columns", "")
                if col_search:
                    filtered_cols = [col for col in dataset_slice.columns if col_search.lower() in col.lower()]
                else:
                    filtered_cols = dataset_slice.columns[:10].tolist()  # Default to first 10 columns
                
                selected_cols = st.multiselect(
                    "Select columns to display",
                    options=dataset_slice.columns.tolist(),
                    default=filtered_cols
                )
                
                if selected_cols:
                    st.dataframe(dataset_slice[selected_cols], use_container_width=True)
                    
                    # Download link for CSV
                    csv = dataset_slice[selected_cols].to_csv(index=True)
                    st.download_button(
                        label="Download Selected Data CSV",
                        data=csv,
                        file_name=f"{selected_dataset}_{start_date_str}_{end_date_str}.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("Please select at least one column to display data.")
            else:
                st.warning("No data available for the selected date range in this dataset.")
        else:
            st.info("Please select a dataset to explore.")

else:
    # Show welcome screen with instructions
    welcome_container = st.container()
    with welcome_container:
        st.markdown("""
        <div class="welcome-container">
            <h1>Welcome to the Financial Crisis Early Warning System</h1>
            <p class="welcome-text">This dashboard provides predictive analytics to help detect upcoming financial crises before they occur.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        ## Getting Started
        
        1. Click the **Load Models** button in the sidebar to load all trained models
        2. Next, click **Load Datasets** to load the financial datasets
        3. Select a date range and model to start analyzing
        
        ## Features
        """)
        
        feature_cols = st.columns(3)
        
        with feature_cols[0]:
            st.markdown("""
            ### 🔍 Crisis Detection
            - Early warning signals
            - Multiple model comparison
            - Historical crisis analysis
            """)
            
        with feature_cols[1]:
            st.markdown("""
            ### 📊 Data Visualization
            - Interactive charts
            - Probability heatmaps
            - Custom time periods
            """)
            
        with feature_cols[2]:
            st.markdown("""
            ### 📈 Risk Analysis
            - Current market risk level
            - Trend detection
            - Crisis probability metrics
            """)
        
        # Sample image if available
        sample_img_path = os.path.join(os.path.dirname(__file__), 'assets/sample_dashboard.png')
        if os.path.exists(sample_img_path):
            st.image(sample_img_path, caption="Sample Dashboard Preview")

# Footer
st.markdown("""
<div class="footer">
    <p>Financial Crisis Early Warning System | Fintech Final Delivery | Group 9 | © 2025</p>
</div>
""", unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    # This won't execute in Streamlit cloud environment
    pass 