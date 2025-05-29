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
    create_single_day_prediction_card
)

# Set page config
st.set_page_config(
    page_title="Financial Crisis EWS",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
        
        # Date selection with improved styling
        with st.expander("📅 Time Period", expanded=True):
            st.markdown("""<div class="section-subheader">Select analysis period</div>""", unsafe_allow_html=True)
            
            # Preset date ranges for better UX
            preset_ranges = {
                "Last 1 Year": (datetime.date.today() - datetime.timedelta(days=365), datetime.date.today()),
                "Last 3 Years": (datetime.date.today() - datetime.timedelta(days=3*365), datetime.date.today()),
                "2020 COVID Crisis": (datetime.date(2020, 1, 1), datetime.date(2020, 12, 31)),
                "2008 Financial Crisis": (datetime.date(2007, 1, 1), datetime.date(2009, 12, 31)),
                "Custom Range": None
            }
            
            selected_preset = st.selectbox("Preset Periods", list(preset_ranges.keys()))
            
            if selected_preset == "Custom Range":
                # Allow custom date selection
                start_date = st.date_input(
                    "Start Date",
                    value=datetime.date(2019, 1, 1),
                    min_value=datetime.date(2000, 1, 1),
                    max_value=datetime.date.today()
                )
                
                end_date = st.date_input(
                    "End Date",
                    value=datetime.date.today(),
                    min_value=start_date,
                    max_value=datetime.date.today()
                )
            else:
                # Use preset dates
                start_date, end_date = preset_ranges[selected_preset]
            
            # Convert to string format for model manager
            start_date_str = start_date.strftime('%Y-%m-%d')
            end_date_str = end_date.strftime('%Y-%m-%d')
            
            st.caption(f"Analyzing data from {start_date_str} to {end_date_str}")
        
        # Store date range in session state
        if 'selected_date_range' not in st.session_state:
            st.session_state.selected_date_range = (start_date, end_date, start_date_str, end_date_str)
        else:
            st.session_state.selected_date_range = (start_date, end_date, start_date_str, end_date_str)
        
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
            
            show_heatmap = st.checkbox("Show Crisis Probability Heatmap", value=True)
            
            col1, col2 = st.columns(2)
            with col1:
                heatmap_freq = st.selectbox(
                    "Heatmap Frequency",
                    options=["Monthly", "Quarterly", "Yearly"],
                    index=0
                )
            
            with col2:
                threshold = st.slider("Crisis Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
            
            freq_map = {"Monthly": "M", "Quarterly": "Q", "Yearly": "Y"}
            
            show_full_dashboard = st.checkbox("Show Full Dashboard", value=True)
        
    # Help information in accordion
    with st.expander("ℹ️ Help"):
        st.markdown("""
        ### How to use this dashboard
        
        1. Load models and datasets first
        2. Select a date range for analysis
        3. Choose models to analyze
        4. Compare predictions across models
        
        ### Interpreting Results
        
        - **Crisis Probability**: Values closer to 1.0 indicate higher risk
        - **Red Shaded Areas**: Historical crisis periods
        - **Threshold Line**: The cutoff for crisis prediction (default: 0.5)
        
        For more help, see the documentation or contact support.
        """)
    
    # About section in sidebar
    with st.expander("📝 About"):
        st.markdown("""
        ### Financial Crisis EWS
        
        This dashboard visualizes predictions from machine learning models
        trained to detect financial crisis events based on market indicators.
        
        The models analyze patterns in financial data to estimate the probability
        of market stress in the near future.
        
        Version 2.0 | Last updated: May 2025
        """)

# Main content area with tabs
if st.session_state.model_manager is not None and st.session_state.datasets_loaded:
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Dashboard", "🔍 Model Details", "⚖️ Comparison", "📊 Data Explorer"])
    
    with tab1:
        # Main dashboard tab
        if selected_model:
            # Get model info
            model_info = st.session_state.model_manager.get_model_by_name(selected_model)
            
            # Extract date range from session state
            start_date, end_date, start_date_str, end_date_str = st.session_state.selected_date_range
            
            # Specific day selector (in the main dashboard)
            st.markdown('<div class="section-header">Select Specific Day for Prediction</div>', unsafe_allow_html=True)
            date_selector_container = st.container()
            
            with date_selector_container:
                col1, col2 = st.columns([2, 2])
                with col1:
                    # Default to the middle of the selected range if possible
                    default_specific_date = start_date + (end_date - start_date) // 2
                    specific_date = st.date_input(
                        "Select Date Within Range",
                        value=default_specific_date,
                        min_value=start_date,
                        max_value=end_date,
                        key="specific_date_picker",
                        help="Select a specific date for detailed crisis prediction"
                    )
                    specific_date_str = specific_date.strftime('%Y-%m-%d')
                
                with col2:
                    st.markdown("")  # Empty space for alignment
                    # Choose style based on dark/light mode
                    if st.session_state.dark_mode:
                        notification_class = "date-notification-dark"
                    else:
                        notification_class = "date-notification-light"
                        
                    st.markdown(f"""
                    <div class="{notification_class}" style="margin-top: 34px; padding: 10px; border-left: 4px solid #0284c7; background-color: rgba(2, 132, 199, 0.1);">
                        <p style="color: {'#38bdf8' if st.session_state.dark_mode else '#0284c7'}; font-size: 15px; font-weight: bold; margin: 0;">
                            Getting detailed prediction for: <span style="color: {'#f0f0f0' if st.session_state.dark_mode else '#334155'};">{specific_date_str}</span>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Get range predictions
            predictions = st.session_state.model_manager.predict(model_info, start_date_str, end_date_str)
            
            # Get specific day prediction
            specific_prediction = st.session_state.model_manager.predict_for_day(model_info, specific_date_str)
            
            # Get VIX data
            vix_data = st.session_state.model_manager.get_vix_data(start_date_str, end_date_str)
            
            # Display specific day prediction
            if specific_prediction:
                day_prediction_fig = create_single_day_prediction_card(specific_prediction)
                st.plotly_chart(day_prediction_fig, use_container_width=True, config={'displayModeBar': False})
            else:
                st.error("No data available for the selected date. Try selecting a different date.")
            
            # Divider
            st.markdown("""
            <hr style="height:2px; border:none; background: linear-gradient(to right, transparent, #d1d5db, transparent);">
            """, unsafe_allow_html=True)
            
            # Display range predictions
            st.markdown('<div class="section-header">📈 Date Range Analysis</div>', unsafe_allow_html=True)
            if predictions is not None:
                # Show status with improved styling
                latest_pred = predictions.iloc[-1] if len(predictions) > 0 else None
                
                if latest_pred is not None:
                    latest_date = latest_pred['date'].strftime('%Y-%m-%d')
                    latest_prob = latest_pred['probability']
                    
                    # Choose alert style based on dark/light mode
                    high_risk_class = "high-risk-alert-dark" if st.session_state.dark_mode else "high-risk-alert"
                    med_risk_class = "medium-risk-alert-dark" if st.session_state.dark_mode else "medium-risk-alert"
                    low_risk_class = "low-risk-alert-dark" if st.session_state.dark_mode else "low-risk-alert"
                    
                    # Create alert based on crisis probability
                    alert_container = st.container()
                    with alert_container:
                        if latest_prob >= 0.7:
                            st.markdown(f"""
                            <div class="{high_risk_class}">
                                <h3>⚠️ HIGH RISK ALERT</h3>
                                <p>Crisis probability is at <strong>{latest_prob:.2%}</strong> as of {latest_date}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        elif latest_prob >= 0.3:
                            st.markdown(f"""
                            <div class="{med_risk_class}">
                                <h3>⚠️ ELEVATED RISK</h3>
                                <p>Crisis probability is at <strong>{latest_prob:.2%}</strong> as of {latest_date}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="{low_risk_class}">
                                <h3>✓ STABLE OUTLOOK</h3>
                                <p>Crisis probability is low at <strong>{latest_prob:.2%}</strong> as of {latest_date}</p>
                            </div>
                            """, unsafe_allow_html=True)
                
                # Display full dashboard with enhanced visuals
                if show_full_dashboard and vix_data is not None:
                    dashboard_fig = create_financial_dashboard(vix_data, predictions, highlight_date=specific_date)
                    st.plotly_chart(dashboard_fig, use_container_width=True, config={'displayModeBar': False})
                else:
                    # Show simpler chart with improved styling
                    chart_fig = create_price_prediction_chart(vix_data, predictions, threshold=threshold, highlight_date=specific_date)
                    st.plotly_chart(chart_fig, use_container_width=True, config={'displayModeBar': False})
            else:
                st.error("No predictions available for the selected date range and model. Try a different date range.")
            
            # Show heatmap below if selected (full width)
            if show_heatmap and predictions is not None:
                st.markdown(f'<div class="section-header">{heatmap_freq} Crisis Probability Heatmap</div>', unsafe_allow_html=True)
                heatmap_fig = create_heatmap_periods(predictions, freq=freq_map[heatmap_freq])
                st.plotly_chart(heatmap_fig, use_container_width=True, config={'displayModeBar': False})
            
            # Add crisis probability KPI metrics in a row (full width)
            if predictions is not None:
                st.markdown('<div class="section-header">Key Risk Metrics</div>', unsafe_allow_html=True)
                
                metric_cols = st.columns(4)
                
                with metric_cols[0]:
                    avg_prob = predictions['probability'].mean()
                    st.metric(
                        label="Average Risk",
                        value=f"{avg_prob:.2%}",
                        delta=f"{avg_prob - 0.5:.2%}" if avg_prob != 0.5 else None,
                        delta_color="inverse"
                    )
                
                with metric_cols[1]:
                    max_prob = predictions['probability'].max()
                    st.metric(
                        label="Peak Risk",
                        value=f"{max_prob:.2%}"
                    )
                
                with metric_cols[2]:
                    days_above = (predictions['probability'] > threshold).sum()
                    pct_above = days_above / len(predictions) * 100 if len(predictions) > 0 else 0
                    st.metric(
                        label=f"Days Above {threshold}",
                        value=f"{days_above}",
                        delta=f"{pct_above:.1f}% of period"
                    )
                
                with metric_cols[3]:
                    current_trend = predictions['probability'].iloc[-5:].mean() - predictions['probability'].iloc[-10:-5].mean() if len(predictions) >= 10 else 0
                    st.metric(
                        label="Recent Trend",
                        value="Increasing" if current_trend > 0.01 else "Decreasing" if current_trend < -0.01 else "Stable",
                        delta=f"{current_trend:.2%}",
                        delta_color="inverse"
                    )
    
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
            metrics_cols = st.columns(5)
            
            metrics = {
                'accuracy': model_info['metadata'].get('metrics', {}).get('accuracy', None),
                'precision': model_info['metadata'].get('metrics', {}).get('precision', None),
                'recall': model_info['metadata'].get('metrics', {}).get('recall', None),
                'f1_score': model_info['metadata'].get('metrics', {}).get('f1_score', None),
                'roc_auc': model_info['metadata'].get('metrics', {}).get('roc_auc', None)
            }
            
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
            
            metrics_df = []
            for model_name in compare_models:
                model_info = st.session_state.model_manager.get_model_by_name(model_name)
                if model_info:
                    # Get predictions for this time period
                    predictions = st.session_state.model_manager.predict(model_info, start_date_str, end_date_str)
                    
                    if predictions is not None:
                        # Calculate period metrics
                        avg_prob = predictions['probability'].mean()
                        max_prob = predictions['probability'].max()
                        days_above = (predictions['probability'] > threshold).sum()
                        pct_above = days_above / len(predictions) * 100 if len(predictions) > 0 else 0
                        
                        # Get trained metrics
                        model_type = model_info['model_type']
                        dataset = model_info['dataset_name']
                        best_metric = model_info['metadata'].get('metric', 'Unknown')
                        best_score = model_info['metadata'].get('score', 'Unknown')
                        
                        # Add to comparison dataframe
                        metrics_df.append({
                            'Model': model_name,
                            'Type': model_type,
                            'Dataset': dataset,
                            'Best Metric': best_metric,
                            'Best Score': best_score,
                            'Avg Probability': avg_prob,
                            'Max Probability': max_prob,
                            f'Days Above {threshold}': days_above,
                            '% Time Above Threshold': pct_above
                        })
            
            if metrics_df:
                comparison_table = pd.DataFrame(metrics_df)
                st.dataframe(comparison_table, hide_index=True, use_container_width=True)
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
    <p>Financial Crisis Early Warning System | Developed by Financial AI Team | © 2025</p>
</div>
""", unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    # This won't execute in Streamlit cloud environment
    pass 