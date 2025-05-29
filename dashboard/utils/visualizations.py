import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
import numpy as np

# Define a more vibrant and enhanced color scheme
CHART_COLORS = {
    'background': 'rgba(0, 0, 0, 0)',  # Transparent background
    'grid': '#f1f5f9',
    'vix': '#3b82f6',  # Brighter blue
    'probability': '#ef4444',  # Vibrant red
    'threshold': '#475569',  # Dark slate
    'crisis_area': 'rgba(239, 68, 68, 0.15)',  # Light red
    'green': '#10b981',  # Bright green
    'yellow': '#f59e0b',  # Bright orange/yellow
    'text': '#334155',  # Dark slate for text
    'highlight': '#0284c7',  # Bright blue for highlighted date
    'highlight_bg': 'rgba(2, 132, 199, 0.1)'  # Light blue background
}

def create_price_prediction_chart(vix_data, predictions_df, threshold=0.5, highlight_date=None):
    """
    Create a financial chart showing VIX values and model predictions
    
    Parameters:
    -----------
    vix_data : DataFrame
        DataFrame with VIX data
    predictions_df : DataFrame
        DataFrame with model predictions
    threshold : float
        Probability threshold for predictions
    highlight_date : str, optional
        A specific date to highlight with a vertical line
    
    Returns:
    --------
    plotly figure
    """
    # Find VIX column
    vix_col = [col for col in vix_data.columns if 'vix' in col.lower()][0] if vix_data is not None else None
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add VIX trace
    if vix_data is not None and vix_col is not None:
        fig.add_trace(
            go.Scatter(
                x=vix_data.index, 
                y=vix_data[vix_col],
                name="VIX",
                line=dict(color=CHART_COLORS['vix'], width=2.5),
                opacity=0.9
            ),
            secondary_y=False,
        )
    
    # Add prediction probability trace
    if predictions_df is not None:
        # Add shaded area beneath probability line
        fig.add_trace(
            go.Scatter(
                x=predictions_df['date'], 
                y=predictions_df['probability'],
                name="Crisis Probability",
                line=dict(color=CHART_COLORS['probability'], width=3),
                fill='tozeroy',
                fillcolor='rgba(239, 68, 68, 0.15)',
                opacity=0.95
            ),
            secondary_y=True,
        )
        
        # Add horizontal line at threshold with improved styling
        fig.add_trace(
            go.Scatter(
                x=[predictions_df['date'].min(), predictions_df['date'].max()],
                y=[threshold, threshold],
                name=f"Threshold ({threshold})",
                line=dict(color=CHART_COLORS['threshold'], width=2, dash='dash'),
                opacity=0.8
            ),
            secondary_y=True,
        )
        
        # Add vertical line for the highlighted date if provided
        if highlight_date:
            try:
                # Convert string date to datetime if needed
                if isinstance(highlight_date, str):
                    highlight_date = pd.to_datetime(highlight_date)
                
                # Add vertical line
                fig.add_shape(
                    type="line",
                    x0=highlight_date,
                    x1=highlight_date,
                    y0=0,
                    y1=1,
                    yref="paper",
                    line=dict(
                        color=CHART_COLORS['highlight'],
                        width=3,
                        dash="solid"
                    ),
                )
                
                # Add annotation for the selected date
                fig.add_annotation(
                    x=highlight_date,
                    y=1.05,
                    yref="paper",
                    text="Selected Date",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor=CHART_COLORS['highlight'],
                    font=dict(
                        family="Inter, sans-serif",
                        size=14,
                        color=CHART_COLORS['highlight']
                    ),
                    bgcolor="rgba(255, 255, 255, 0.9)",
                    bordercolor=CHART_COLORS['highlight'],
                    borderwidth=2,
                    borderpad=4,
                )
            except Exception as e:
                print(f"Error highlighting date: {e}")
    
    # Add crisis periods if actual data is available
    if predictions_df is not None and 'actual' in predictions_df.columns:
        # Convert binary series to continuous blocks for shading
        crisis_periods = []
        current_start = None
        
        for i, row in predictions_df.iterrows():
            if row['actual'] == 1 and current_start is None:
                current_start = row['date']
            elif row['actual'] == 0 and current_start is not None:
                crisis_periods.append((current_start, row['date']))
                current_start = None
        
        # Add the last period if it's still open
        if current_start is not None:
            crisis_periods.append((current_start, predictions_df['date'].iloc[-1]))
        
        # Add shaded regions for crisis periods
        for start, end in crisis_periods:
            fig.add_vrect(
                x0=start, x1=end,
                fillcolor=CHART_COLORS['crisis_area'], 
                opacity=0.4,  # Slightly more visible
                layer="below", 
                line_width=0,
                name="Crisis Period"
            )
    
    # Customize layout with improved styling
    fig.update_layout(
        title={
            'text': "Market Volatility and Crisis Probability",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 26, 'color': CHART_COLORS['text'], 'family': 'Inter, sans-serif', 'weight': 'bold'}
        },
        xaxis=dict(
            title="Date",
            gridcolor=CHART_COLORS['grid'],
            zeroline=False,
            tickfont=dict(family="Inter, sans-serif", size=13, color=CHART_COLORS['text'])
        ),
        plot_bgcolor=CHART_COLORS['background'],
        paper_bgcolor=CHART_COLORS['background'],
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(family="Inter, sans-serif", size=13, color=CHART_COLORS['text']),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="lightgrey",
            borderwidth=1
        ),
        margin=dict(l=10, r=10, t=80, b=10),
        height=600,
        hovermode="x unified"
    )
    
    # Update y-axes with improved styling
    fig.update_yaxes(
        title_text="VIX Value", 
        secondary_y=False,
        gridcolor=CHART_COLORS['grid'],
        zeroline=False,
        tickfont=dict(family="Inter, sans-serif", size=13, color=CHART_COLORS['text']),
        titlefont=dict(family="Inter, sans-serif", size=15, color=CHART_COLORS['text'])
    )
    
    fig.update_yaxes(
        title_text="Crisis Probability", 
        secondary_y=True, 
        range=[0, 1],
        gridcolor=CHART_COLORS['grid'],
        zeroline=False,
        tickformat=".0%",
        tickfont=dict(family="Inter, sans-serif", size=13, color=CHART_COLORS['text']),
        titlefont=dict(family="Inter, sans-serif", size=15, color=CHART_COLORS['text'])
    )
    
    return fig

def create_model_comparison_chart(model_manager, start_date, end_date, models_to_compare):
    """
    Create a chart comparing predictions from multiple models
    
    Parameters:
    -----------
    model_manager : ModelManager
        Model manager object
    start_date, end_date : str
        Date range
    models_to_compare : list
        List of model names to compare
        
    Returns:
    --------
    plotly figure
    """
    fig = go.Figure()
    
    # Generate a color scale for multiple models
    colors = px.colors.qualitative.Bold
    
    # Get predictions for each model
    for i, model_name in enumerate(models_to_compare):
        model_info = model_manager.get_model_by_name(model_name)
        if model_info:
            predictions = model_manager.predict(model_info, start_date, end_date)
            if predictions is not None:
                # Get color from palette, cycling if needed
                color = colors[i % len(colors)]
                
                # Shortened model name for legend
                short_name = model_name.split(" (")[0]
                
                fig.add_trace(
                    go.Scatter(
                        x=predictions['date'],
                        y=predictions['probability'],
                        name=short_name,
                        line=dict(color=color, width=2.5),
                        mode='lines',
                        hovertemplate='%{y:.2%} probability on %{x|%b %d, %Y}<extra>' + short_name + '</extra>'
                    )
                )
                
                # Add actual crisis periods if available for the first model
                if model_name == models_to_compare[0] and 'actual' in predictions.columns:
                    # Convert binary series to continuous blocks for shading
                    crisis_periods = []
                    current_start = None
                    
                    for i, row in predictions.iterrows():
                        if row['actual'] == 1 and current_start is None:
                            current_start = row['date']
                        elif row['actual'] == 0 and current_start is not None:
                            crisis_periods.append((current_start, row['date']))
                            current_start = None
                    
                    # Add the last period if it's still open
                    if current_start is not None:
                        crisis_periods.append((current_start, predictions['date'].iloc[-1]))
                    
                    # Add shaded regions for crisis periods
                    for start, end in crisis_periods:
                        fig.add_vrect(
                            x0=start, x1=end,
                            fillcolor=CHART_COLORS['crisis_area'], 
                            opacity=0.3,
                            layer="below", 
                            line_width=0,
                            name="Crisis Period"
                        )
    
    # Add threshold line
    fig.add_trace(
        go.Scatter(
            x=[pd.to_datetime(start_date), pd.to_datetime(end_date)],
            y=[0.5, 0.5],
            name="Threshold (0.5)",
            line=dict(color=CHART_COLORS['threshold'], width=1.5, dash='dash'),
            opacity=0.7
        )
    )
    
    # Customize layout with improved styling
    fig.update_layout(
        title={
            'text': "Model Prediction Comparison",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 24, 'color': CHART_COLORS['text'], 'family': 'Inter, sans-serif'}
        },
        xaxis=dict(
            title="Date",
            gridcolor=CHART_COLORS['grid'],
            zeroline=False,
            tickfont=dict(family="Inter, sans-serif", size=12, color=CHART_COLORS['text'])
        ),
        yaxis=dict(
            title="Crisis Probability",
            range=[0, 1],
            gridcolor=CHART_COLORS['grid'],
            zeroline=False,
            tickformat=".0%",
            tickfont=dict(family="Inter, sans-serif", size=12, color=CHART_COLORS['text'])
        ),
        plot_bgcolor=CHART_COLORS['background'],
        paper_bgcolor=CHART_COLORS['background'],
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(family="Inter, sans-serif", size=12, color=CHART_COLORS['text']),
            bgcolor="rgba(255,255,255,0.0)"  # Fully transparent background
        ),
        margin=dict(l=10, r=10, t=80, b=10),
        height=600,
        hovermode="x unified"
    )
    
    return fig

def create_heatmap_periods(predictions_df, freq='M'):
    """
    Create a heatmap showing crisis probabilities by time period
    
    Parameters:
    -----------
    predictions_df : DataFrame
        DataFrame with model predictions
    freq : str
        Frequency for resampling (e.g., 'M' for monthly, 'Q' for quarterly)
        
    Returns:
    --------
    plotly figure
    """
    # Resample by frequency
    resampled = predictions_df.set_index('date')['probability'].resample(freq).mean().reset_index()
    
    # Format dates based on frequency
    if freq == 'M':
        resampled['period'] = resampled['date'].dt.strftime('%b %Y')
    elif freq == 'Q':
        resampled['period'] = resampled['date'].dt.to_period('Q').astype(str)
    elif freq == 'Y':
        resampled['period'] = resampled['date'].dt.year
    else:
        resampled['period'] = resampled['date'].dt.strftime('%Y-%m-%d')
    
    # Create a custom color scale
    custom_colorscale = [
        [0.0, '#10b981'],  # Green for low probability (0%)
        [0.3, '#10b981'],  # Green up to 30%
        [0.3, '#f59e0b'],  # Yellow/Orange transition at 30%
        [0.7, '#f59e0b'],  # Yellow/Orange up to 70%
        [0.7, '#ef4444'],  # Red transition at 70%
        [1.0, '#ef4444']   # Red for high probability (100%)
    ]
    
    # Create heatmap with improved styling
    fig = px.imshow(
        np.array(resampled['probability']).reshape(1, -1),
        y=['Crisis Risk'],
        x=resampled['period'],
        color_continuous_scale=custom_colorscale,
        zmin=0, zmax=1,
        aspect="auto",
        labels=dict(color="Probability")
    )
    
    # Add annotations with percentage formatting
    for i, value in enumerate(resampled['probability']):
        text_color = "white" if value > 0.7 or value < 0.3 else "black"
        fig.add_annotation(
            text=f"{value:.0%}",
            x=i, y=0,
            showarrow=False,
            font=dict(color=text_color, size=14, family="Inter, sans-serif")
        )
    
    # Update layout with improved styling
    fig.update_layout(
        title={
            'text': f"Crisis Probability Heatmap ({freq})",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 20, 'color': CHART_COLORS['text'], 'family': 'Inter, sans-serif'}
        },
        xaxis_title="Time Period",
        coloraxis_colorbar=dict(
            title="Probability",
            tickformat=".0%",
            tickfont=dict(family="Inter, sans-serif", size=12, color=CHART_COLORS['text'])
        ),
        plot_bgcolor=CHART_COLORS['background'],
        paper_bgcolor=CHART_COLORS['background'],
        margin=dict(l=10, r=10, t=80, b=10),
        height=200,
        xaxis=dict(tickfont=dict(size=12, family="Inter, sans-serif"))
    )
    
    return fig

def create_financial_dashboard(vix_data, predictions_df, highlight_date=None):
    """
    Create a comprehensive financial dashboard combining multiple visualizations
    
    Parameters:
    -----------
    vix_data : DataFrame
        DataFrame with VIX data
    predictions_df : DataFrame
        DataFrame with model predictions
    highlight_date : str or datetime, optional
        A specific date to highlight with a vertical line
        
    Returns:
    --------
    plotly figure
    """
    # Create subplot figure with improved layout
    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.5, 0.25, 0.25],
        specs=[
            [{"type": "xy"}],
            [{"type": "indicator"}],
            [{"type": "xy"}]
        ],
        vertical_spacing=0.08,
        subplot_titles=[
            "<b>VIX and Crisis Predictions</b>", 
            "<b>Current Crisis Probability</b>", 
            "<b>Historical Probability Trend</b>"
        ]
    )
    
    # Find VIX column
    vix_col = [col for col in vix_data.columns if 'vix' in col.lower()][0] if vix_data is not None else None
    
    # 1. Add main chart (VIX and predictions)
    if vix_data is not None and vix_col is not None:
        fig.add_trace(
            go.Scatter(
                x=vix_data.index, 
                y=vix_data[vix_col],
                name="VIX",
                line=dict(color=CHART_COLORS['vix'], width=2.5),
                opacity=0.9
            ),
            row=1, col=1
        )
    
    if predictions_df is not None:
        fig.add_trace(
            go.Scatter(
                x=predictions_df['date'], 
                y=predictions_df['probability'],
                name="Crisis Probability",
                line=dict(color=CHART_COLORS['probability'], width=3),
                fill='tozeroy',
                fillcolor='rgba(239, 68, 68, 0.15)',
                opacity=0.9,
                hovertemplate='Crisis probability: %{y:.2%}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add horizontal line at threshold
        fig.add_shape(
            type="line",
            x0=predictions_df['date'].min(),
            x1=predictions_df['date'].max(),
            y0=0.5, y1=0.5,
            line=dict(color=CHART_COLORS['threshold'], width=2, dash="dash"),
            row=1, col=1
        )
        
        # Add vertical line for the highlighted date if provided
        if highlight_date:
            try:
                # Convert string date to datetime if needed
                if isinstance(highlight_date, str):
                    highlight_date = pd.to_datetime(highlight_date)
                
                # Add vertical line to main chart (row 1)
                fig.add_shape(
                    type="line",
                    x0=highlight_date,
                    x1=highlight_date,
                    y0=0,
                    y1=1,
                    yref="paper",
                    xref="x",
                    line=dict(
                        color=CHART_COLORS['highlight'],
                        width=3,
                        dash="solid"
                    ),
                    row=1, col=1
                )
                
                # Add annotation for the selected date
                fig.add_annotation(
                    x=highlight_date,
                    y=1.05,
                    yref="paper",
                    xref="x",
                    text="Selected Date",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor=CHART_COLORS['highlight'],
                    font=dict(
                        family="Inter, sans-serif",
                        size=14,
                        color=CHART_COLORS['highlight']
                    ),
                    bgcolor="rgba(255, 255, 255, 0.9)",
                    bordercolor=CHART_COLORS['highlight'],
                    borderwidth=2,
                    borderpad=4,
                    row=1, col=1
                )
                
                # Also add to monthly bar chart (row 3)
                fig.add_shape(
                    type="line",
                    x0=highlight_date,
                    x1=highlight_date,
                    y0=0,
                    y1=1,
                    yref="paper",
                    xref="x3",  # x3 refers to the x-axis of row 3
                    line=dict(
                        color=CHART_COLORS['highlight'],
                        width=3,
                        dash="solid"
                    ),
                    row=3, col=1
                )
            except Exception as e:
                print(f"Error highlighting date in dashboard: {e}")
        
        # 2. Add current probability indicator with improved styling
        latest_prob = predictions_df['probability'].iloc[-1] if len(predictions_df) > 0 else 0
        
        # Determine color based on probability
        if latest_prob < 0.3:
            color = CHART_COLORS['green']
            risk_level = "Low Risk"
        elif latest_prob < 0.7:
            color = CHART_COLORS['yellow']
            risk_level = "Medium Risk"
        else:
            color = CHART_COLORS['probability']
            risk_level = "High Risk"
            
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=latest_prob,
                domain={'row': 1, 'column': 0},
                title={
                    "text": f"Latest Crisis Probability<br><span style='font-size:0.8em;color:{color}'>{risk_level}</span>",
                    "font": {"size": 16, "family": "Inter, sans-serif"}
                },
                number={
                    "font": {"size": 40, "color": color, "family": "Inter, sans-serif"},
                    "valueformat": ".0%"
                },
                delta={
                    'reference': 0.5,
                    'relative': False,
                    'position': "bottom",
                    'valueformat': ".1%",
                    'font': {"size": 14, "family": "Inter, sans-serif"}
                },
                gauge={
                    'axis': {'range': [0, 1], 'tickformat': ".0%", 'tickfont': {"size": 14, "family": "Inter, sans-serif"}},
                    'bar': {'color': color, 'thickness': 0.7},
                    'steps': [
                        {'range': [0, 0.3], 'color': 'rgba(16, 185, 129, 0.3)'},  # Green - Low risk
                        {'range': [0.3, 0.7], 'color': 'rgba(245, 158, 11, 0.3)'},  # Yellow - Medium risk
                        {'range': [0.7, 1], 'color': 'rgba(239, 68, 68, 0.3)'},  # Red - High risk
                    ],
                    'threshold': {
                        'line': {'color': CHART_COLORS['threshold'], 'width': 4},
                        'thickness': 0.85,
                        'value': 0.5
                    },
                    'bgcolor': CHART_COLORS['background']  # Transparent background
                }
            ),
            row=2, col=1
        )
        
        # 3. Add historical trend with improved styling
        # Resample to monthly for clearer trend
        monthly_probs = predictions_df.set_index('date')['probability'].resample('M').mean().reset_index()
        
        # Set color based on risk level
        bar_colors = monthly_probs['probability'].apply(
            lambda x: CHART_COLORS['green'] if x < 0.3 else 
                      CHART_COLORS['yellow'] if x < 0.7 else 
                      CHART_COLORS['probability']
        )
        
        fig.add_trace(
            go.Bar(
                x=monthly_probs['date'],
                y=monthly_probs['probability'],
                name="Monthly Risk",
                marker_color=bar_colors,
                hovertemplate='%{y:.2%} probability<br>%{x|%b %Y}<extra></extra>'
            ),
            row=3, col=1
        )
        
        # Add horizontal line at threshold in the bar chart
        fig.add_shape(
            type="line",
            x0=monthly_probs['date'].min(),
            x1=monthly_probs['date'].max(),
            y0=0.5, y1=0.5,
            line=dict(color=CHART_COLORS['threshold'], width=1.5, dash="dash"),
            row=3, col=1
        )
        
        # Add crisis periods if actual data is available
        if 'actual' in predictions_df.columns:
            # Convert binary series to continuous blocks for shading
            crisis_periods = []
            current_start = None
            
            for i, row in predictions_df.iterrows():
                if row['actual'] == 1 and current_start is None:
                    current_start = row['date']
                elif row['actual'] == 0 and current_start is not None:
                    crisis_periods.append((current_start, row['date']))
                    current_start = None
            
            # Add the last period if it's still open
            if current_start is not None:
                crisis_periods.append((current_start, predictions_df['date'].iloc[-1]))
            
            # Add shaded regions for crisis periods - fixed for subplot compatibility
            for start, end in crisis_periods:
                # For main chart (row 1)
                fig.add_shape(
                    type="rect",
                    x0=start,
                    x1=end,
                    y0=0,
                    y1=1,
                    fillcolor=CHART_COLORS['crisis_area'],
                    opacity=0.3,
                    layer="below",
                    line_width=0,
                    xref="x",
                    yref="paper",
                    row=1, col=1
                )
                
                # For bar chart (row 3)
                fig.add_shape(
                    type="rect",
                    x0=start,
                    x1=end,
                    y0=0,
                    y1=1,
                    fillcolor=CHART_COLORS['crisis_area'],
                    opacity=0.3,
                    layer="below",
                    line_width=0,
                    xref="x3",
                    yref="paper",
                    row=3, col=1
                )
    
    # Update layout with improved styling
    fig.update_layout(
        title={
            'text': "Financial Crisis Early Warning Dashboard",
            'y':0.98,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 26, 'color': CHART_COLORS['text'], 'family': 'Inter, sans-serif', 'weight': 'bold'}
        },
        plot_bgcolor=CHART_COLORS['background'],
        paper_bgcolor=CHART_COLORS['background'],
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(family="Inter, sans-serif", size=12, color=CHART_COLORS['text']),
            bgcolor="rgba(255,255,255,0.0)"  # Fully transparent background
        ),
        margin=dict(l=10, r=10, t=100, b=10),
        height=1000,
        hovermode="x unified"
    )
    
    # Update axes styling
    fig.update_xaxes(gridcolor=CHART_COLORS['grid'], zeroline=False, row=1, col=1)
    fig.update_xaxes(gridcolor=CHART_COLORS['grid'], zeroline=False, row=3, col=1)
    
    fig.update_yaxes(
        title_text="VIX Value / Probability", 
        gridcolor=CHART_COLORS['grid'], 
        zeroline=False,
        tickfont=dict(family="Inter, sans-serif", size=12),
        row=1, col=1
    )
    
    fig.update_yaxes(
        title_text="Probability", 
        range=[0, 1], 
        gridcolor=CHART_COLORS['grid'], 
        zeroline=False,
        tickformat=".0%",
        tickfont=dict(family="Inter, sans-serif", size=12),
        row=3, col=1
    )
    
    return fig

def create_single_day_prediction_card(prediction_data):
    """
    Create a visualization for a single day prediction
    
    Parameters:
    -----------
    prediction_data : dict
        Dictionary containing prediction information for a specific date
        
    Returns:
    --------
    plotly figure
    """
    if not prediction_data:
        # Create empty figure with message if no prediction data
        fig = go.Figure()
        fig.add_annotation(
            text="No prediction data available for the selected date",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=18, color=CHART_COLORS['text'], family="Inter, sans-serif")
        )
        fig.update_layout(height=350, paper_bgcolor=CHART_COLORS['background'])
        return fig
    
    # Extract prediction data
    date_str = prediction_data.get('date', 'Unknown Date')
    probability = prediction_data.get('probability', 0)
    prediction = prediction_data.get('prediction', 0)
    actual = prediction_data.get('actual')
    vix_value = prediction_data.get('vix')
    
    # Determine risk level and color based on probability
    if probability < 0.3:
        risk_level = "LOW RISK"
        color = CHART_COLORS['green']
        risk_text = "Conditions appear stable"
    elif probability < 0.7:
        risk_level = "MEDIUM RISK"
        color = CHART_COLORS['yellow']
        risk_text = "Elevated risk signals detected"
    else:
        risk_level = "HIGH RISK"
        color = CHART_COLORS['probability']
        risk_text = "Critical risk level observed"
    
    # Create subplots for gauge and additional info with better spacing
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.65, 0.35],
        specs=[[{"type": "indicator"}, {"type": "table"}]],
        horizontal_spacing=0.03
    )
    
    # Add gauge indicator with improved styling
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=probability,
            title={
                "text": f"<b>Crisis Probability</b><br><span style='font-size:1.3em;color:{color}'>{risk_level}</span><br><span style='font-size:0.9em;color:{CHART_COLORS['text']}'>{date_str}</span>",
                "font": {"size": 20, "family": "Inter, sans-serif"}
            },
            number={
                "font": {"size": 60, "color": color, "family": "Inter, sans-serif"},
                "valueformat": ".1%"
            },
            gauge={
                'axis': {
                    'range': [0, 1], 
                    'tickformat': ".0%", 
                    'tickfont': {"size": 14, "family": "Inter, sans-serif"},
                    'tickwidth': 2
                },
                'bar': {'color': color, 'thickness': 0.8},
                'steps': [
                    {'range': [0, 0.3], 'color': 'rgba(16, 185, 129, 0.25)', 'line': {'width': 1, 'color': 'white'}},  # Green
                    {'range': [0.3, 0.7], 'color': 'rgba(245, 158, 11, 0.25)', 'line': {'width': 1, 'color': 'white'}},  # Yellow
                    {'range': [0.7, 1], 'color': 'rgba(239, 68, 68, 0.25)', 'line': {'width': 1, 'color': 'white'}}  # Red
                ],
                'threshold': {
                    'line': {'color': CHART_COLORS['threshold'], 'width': 5},
                    'thickness': 0.85,
                    'value': 0.5
                },
                'bgcolor': CHART_COLORS['background'],
                'borderwidth': 0  # Remove border for clean look
            },
            domain={'row': 0, 'column': 0}
        ),
        row=1, col=1
    )
    
    # Create a table with additional information - improved styling
    table_values = [
        ["<b>Date</b>", f"<b>{date_str}</b>"],
        ["<b>Probability</b>", f"<span style='color:{color};font-weight:bold'>{probability:.2%}</span>"],
        ["<b>Status</b>", f"<span style='color:{color}'>{risk_text}</span>"],
        ["<b>Prediction</b>", f"<span style='color:{color if prediction == 1 else CHART_COLORS['green']};font-weight:bold'>{'Crisis' if prediction == 1 else 'No Crisis'}</span>"],
    ]
    
    # Add VIX value if available
    if vix_value is not None:
        vix_color = CHART_COLORS['vix']  
        table_values.append(["<b>VIX Value</b>", f"<span style='color:{vix_color};font-weight:bold'>{vix_value:.2f}</span>"])
    
    # Add actual value if available
    if actual is not None and not np.isnan(actual):
        actual_color = CHART_COLORS['probability'] if actual == 1 else CHART_COLORS['green']
        table_values.append(["<b>Actual</b>", f"<span style='color:{actual_color};font-weight:bold'>{'Crisis' if actual == 1 else 'No Crisis'}</span>"])
    
    # Add table with improved styling
    fig.add_trace(
        go.Table(
            header=dict(
                values=["<b>Metric</b>", "<b>Value</b>"],
                line_color=CHART_COLORS['background'],
                fill_color='rgba(241, 245, 249, 0.3)',  # Very light gray with transparency
                align='center',
                font=dict(color=CHART_COLORS['text'], size=15, family="Inter, sans-serif"),
                height=35
            ),
            cells=dict(
                values=list(map(list, zip(*table_values))),
                line_color=CHART_COLORS['background'],
                fill_color=CHART_COLORS['background'],
                align=['left', 'center'],
                font=dict(color=CHART_COLORS['text'], size=14, family="Inter, sans-serif"),
                height=35
            )
        ),
        row=1, col=2
    )
    
    # Update layout with improved styling
    fig.update_layout(
        title=None,  # Remove title for cleaner look
        plot_bgcolor=CHART_COLORS['background'],
        paper_bgcolor=CHART_COLORS['background'],
        margin=dict(l=10, r=10, t=30, b=10),
        height=420,
        shapes=[
            # Add a subtle outline instead of full border
            dict(
                type="rect",
                xref="paper", yref="paper",
                x0=0, y0=0, x1=1, y1=1,
                line=dict(color=color, width=1, dash='dot'),
                layer="below"
            )
        ]
    )
    
    return fig 