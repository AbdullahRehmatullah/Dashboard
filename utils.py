import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import streamlit as st

# Color palette for consistent visualizations
COLOR_MAP = {
    'MarketingDescription': '#1f77b4',  # blue
    'PriceIntelligence': '#ff7f0e',     # orange
    'Nl2Sql': '#2ca02c',                # green
    'PointOfInterest': '#d62728',       # red
    'TextCompletion': '#9467bd'         # purple
}

def load_and_process_analytics_data(file):
    """
    Load and process AI analytics data from uploaded file
    """
    try:
        df = pd.read_csv(file)
        # Check if required columns exist
        required_columns = ['aia_id', 'aia_timestamp', 'aia_group_id', 'aia_user_id', 
                           'aia_feature', 'aia_prompt_token', 'aia_completion_token']
        
        if not all(col in df.columns for col in required_columns):
            st.error("AI Analytics file missing required columns. Please check the file format.")
            return None
        
        # Convert timestamp to datetime
        df['aia_timestamp'] = pd.to_datetime(df['aia_timestamp'])
        
        # Extract date for easier filtering
        df['date'] = df['aia_timestamp'].dt.date
        
        # Calculate total tokens
        df['total_tokens'] = df['aia_prompt_token'] + df['aia_completion_token']
        
        return df
    except Exception as e:
        st.error(f"Error processing AI analytics file: {str(e)}")
        return None

def load_group_data(file):
    """
    Load group name data from uploaded file
    """
    try:
        df = pd.read_csv(file, skiprows=1)  # Skip the empty row
        
        # Check if required columns exist
        required_columns = ['Company Name', 'Group ID']
        
        if not all(col in df.columns for col in required_columns):
            st.error("Group data file missing required columns. Please check the file format.")
            return None
            
        # Rename columns for easier merging
        df = df.rename(columns={'Group ID': 'group_id', 'Company Name': 'company_name'})
        
        return df
    except Exception as e:
        st.error(f"Error processing group data file: {str(e)}")
        return None

def merge_datasets(analytics_df, group_df):
    """
    Merge analytics and group datasets
    """
    if analytics_df is None or group_df is None:
        return None
    
    try:
        # Convert group_id to int for proper joining
        group_df['group_id'] = group_df['group_id'].astype(int)
        analytics_df['aia_group_id'] = analytics_df['aia_group_id'].astype(int)
        
        # Merge datasets
        merged_df = pd.merge(
            analytics_df,
            group_df[['group_id', 'company_name']],
            left_on='aia_group_id',
            right_on='group_id',
            how='left'
        )
        
        # Handle any missing company names
        merged_df['company_name'] = merged_df['company_name'].fillna(f"Unknown (ID: {merged_df['aia_group_id']})")
        
        return merged_df
    except Exception as e:
        st.error(f"Error merging datasets: {str(e)}")
        return None

def get_date_range(df):
    """
    Get the min and max dates from the dataset
    """
    min_date = df['date'].min()
    max_date = df['date'].max()
    return min_date, max_date

def filter_data_by_date(df, start_date, end_date):
    """
    Filter the dataframe by date range
    """
    return df[(df['date'] >= start_date) & (df['date'] <= end_date)]

def filter_data_by_company(df, selected_companies):
    """
    Filter the dataframe by selected companies
    """
    if not selected_companies or 'All Companies' in selected_companies:
        return df
    return df[df['company_name'].isin(selected_companies)]

def filter_data_by_function(df, selected_functions):
    """
    Filter the dataframe by selected functions
    """
    if not selected_functions or 'All Functions' in selected_functions:
        return df
    return df[df['aia_feature'].isin(selected_functions)]

def get_summary_stats(df):
    """
    Calculate summary statistics from the dataframe
    """
    stats = {
        'total_requests': len(df),
        'total_companies': df['aia_group_id'].nunique(),
        'total_users': df['aia_user_id'].nunique(),
        'total_prompt_tokens': df['aia_prompt_token'].sum(),
        'total_completion_tokens': df['aia_completion_token'].sum(),
        'total_tokens': df['total_tokens'].sum(),
        'avg_tokens_per_request': df['total_tokens'].mean()
    }
    return stats

def create_daily_usage_chart(df):
    """
    Create a daily usage chart
    """
    daily_df = df.groupby(['date', 'aia_feature']).size().reset_index(name='count')
    
    fig = px.line(daily_df, x='date', y='count', color='aia_feature', 
                 color_discrete_map=COLOR_MAP,
                 title='Daily AI Function Usage',
                 labels={'count': 'Number of Requests', 'date': 'Date', 'aia_feature': 'Function'})
    
    fig.update_layout(
        legend_title_text='AI Function',
        hovermode='x unified',
        plot_bgcolor='white',
        xaxis=dict(
            showgrid=True,
            gridcolor='lightgrey',
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgrey',
        )
    )
    
    return fig

def create_function_distribution_chart(df):
    """
    Create a function distribution pie chart
    """
    function_counts = df['aia_feature'].value_counts().reset_index()
    function_counts.columns = ['Function', 'Count']
    
    fig = px.pie(function_counts, values='Count', names='Function', 
                title='AI Function Distribution',
                color='Function',
                color_discrete_map=COLOR_MAP)
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(
        uniformtext_minsize=12,
        uniformtext_mode='hide',
        legend_title_text='Function'
    )
    
    return fig

def create_company_usage_chart(df):
    """
    Create a company usage chart
    """
    # Get top 15 companies by usage
    top_companies = df.groupby('company_name').size().nlargest(15).reset_index(name='count')
    
    fig = px.bar(top_companies, y='company_name', x='count', 
                orientation='h',
                title='Top 15 Companies by AI Function Usage',
                labels={'company_name': 'Company', 'count': 'Number of Requests'})
    
    fig.update_layout(
        yaxis={'categoryorder':'total ascending'},
        plot_bgcolor='white',
        xaxis=dict(
            showgrid=True,
            gridcolor='lightgrey',
        )
    )
    
    return fig

def create_company_function_heatmap(df):
    """
    Create a heatmap of company vs function usage
    """
    # Get top 10 companies by usage
    top_companies = df.groupby('company_name').size().nlargest(10).index.tolist()
    
    # Filter for those companies
    filtered_df = df[df['company_name'].isin(top_companies)]
    
    # Create pivot table
    pivot_df = filtered_df.pivot_table(
        index='company_name', 
        columns='aia_feature', 
        values='aia_id',
        aggfunc='count',
        fill_value=0
    )
    
    # Create heatmap
    fig = px.imshow(
        pivot_df,
        labels=dict(x="AI Function", y="Company", color="Usage Count"),
        x=pivot_df.columns,
        y=pivot_df.index,
        color_continuous_scale='Viridis',
        title='Top 10 Companies - Function Usage Heatmap'
    )
    
    fig.update_layout(
        xaxis={'title': 'AI Function'},
        yaxis={'title': 'Company'},
    )
    
    return fig

def create_token_usage_chart(df):
    """
    Create a token usage chart by function
    """
    token_df = df.groupby('aia_feature').agg({
        'aia_prompt_token': 'sum',
        'aia_completion_token': 'sum'
    }).reset_index()
    
    token_df = token_df.melt(
        id_vars=['aia_feature'],
        value_vars=['aia_prompt_token', 'aia_completion_token'],
        var_name='token_type',
        value_name='token_count'
    )
    
    # Rename for better display
    token_df['token_type'] = token_df['token_type'].replace({
        'aia_prompt_token': 'Prompt Tokens',
        'aia_completion_token': 'Completion Tokens'
    })
    
    fig = px.bar(token_df, x='aia_feature', y='token_count', color='token_type',
                barmode='group',
                title='Token Usage by Function',
                labels={'aia_feature': 'Function', 'token_count': 'Token Count', 'token_type': 'Token Type'})
    
    fig.update_layout(
        legend_title_text='Token Type',
        plot_bgcolor='white',
        xaxis=dict(
            showgrid=True,
            gridcolor='lightgrey',
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgrey',
        )
    )
    
    return fig

def create_user_activity_chart(df):
    """
    Create a user activity histogram
    """
    user_counts = df.groupby('aia_user_id').size().reset_index(name='request_count')
    
    fig = px.histogram(user_counts, x='request_count', 
                      title='User Activity Distribution',
                      labels={'request_count': 'Number of Requests', 'count': 'Number of Users'},
                      nbins=20)
    
    fig.update_layout(
        plot_bgcolor='white',
        xaxis=dict(
            title='Requests per User',
            showgrid=True,
            gridcolor='lightgrey',
        ),
        yaxis=dict(
            title='Number of Users',
            showgrid=True,
            gridcolor='lightgrey',
        )
    )
    
    return fig

def create_hourly_usage_chart(df):
    """
    Create an hourly usage chart
    """
    df['hour'] = df['aia_timestamp'].dt.hour
    hourly_counts = df.groupby(['hour', 'aia_feature']).size().reset_index(name='count')
    
    fig = px.line(hourly_counts, x='hour', y='count', color='aia_feature',
                 color_discrete_map=COLOR_MAP,
                 title='Hourly Usage Patterns',
                 labels={'count': 'Number of Requests', 'hour': 'Hour of Day (24hr)', 'aia_feature': 'Function'})
    
    fig.update_layout(
        legend_title_text='AI Function',
        hovermode='x unified',
        plot_bgcolor='white',
        xaxis=dict(
            showgrid=True,
            gridcolor='lightgrey',
            tickmode='linear',
            tick0=0,
            dtick=1,
            range=[-0.5, 23.5]
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgrey',
        )
    )
    
    return fig

def create_average_token_bar_chart(df):
    """
    Create an average token usage bar chart
    """
    avg_tokens = df.groupby('aia_feature').agg({
        'aia_prompt_token': 'mean',
        'aia_completion_token': 'mean',
        'total_tokens': 'mean'
    }).reset_index()
    
    avg_tokens = avg_tokens.melt(
        id_vars=['aia_feature'],
        value_vars=['aia_prompt_token', 'aia_completion_token', 'total_tokens'],
        var_name='token_type',
        value_name='avg_tokens'
    )
    
    # Rename for better display
    avg_tokens['token_type'] = avg_tokens['token_type'].replace({
        'aia_prompt_token': 'Avg Prompt Tokens',
        'aia_completion_token': 'Avg Completion Tokens',
        'total_tokens': 'Avg Total Tokens'
    })
    
    fig = px.bar(avg_tokens, x='aia_feature', y='avg_tokens', color='token_type',
                barmode='group',
                title='Average Token Usage by Function',
                labels={'aia_feature': 'Function', 'avg_tokens': 'Average Token Count', 'token_type': 'Token Type'})
    
    fig.update_layout(
        legend_title_text='Token Type',
        plot_bgcolor='white',
        xaxis=dict(
            showgrid=True,
            gridcolor='lightgrey',
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgrey',
        )
    )
    
    return fig
