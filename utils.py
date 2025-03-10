import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import streamlit as st
import io
from fpdf import FPDF
import base64
import os
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patheffects as path_effects
matplotlib.use('Agg')  # Set the backend to avoid threading issues
import seaborn as sns

# Color palette for consistent visualizations
COLOR_MAP = {
    'MarketingDescription': '#1f77b4',  # blue
    'PriceIntelligence': '#ff7f0e',     # orange
    'Nl2Sql': '#2ca02c',                # green
    'PointOfInterest': '#d62728',       # red
    'TextCompletion': '#9467bd'         # purple
}

# Stats card colors
STATS_COLORS = {
    'Total Requests': '#4361ee',
    'Total Companies': '#3a86ff',
    'Total Users': '#4cc9f0',
    'Total Tokens': '#4895ef'
}

# Internal user groups to exclude from analysis
INTERNAL_GROUPS = [2677, 5090, 5088]  # Added DB Dedicated for Trainings (5088)

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
        
        # Filter out internal user groups
        merged_df = merged_df[~merged_df['aia_group_id'].isin(INTERNAL_GROUPS)]
        
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
    Create a function distribution pie chart with improved text visibility
    """
    function_counts = df['aia_feature'].value_counts().reset_index()
    function_counts.columns = ['Function', 'Count']
    
    # Calculate percentages for more precise display
    total = function_counts['Count'].sum()
    function_counts['Percentage'] = function_counts['Count'] / total * 100
    function_counts['PercentLabel'] = function_counts['Percentage'].apply(lambda x: f"{x:.1f}%")
    
    # For hover info, create a detailed label
    function_counts['HoverInfo'] = function_counts.apply(
        lambda row: f"{row['Function']}: {row['Count']} requests ({row['Percentage']:.2f}%)", 
        axis=1
    )
    
    # Create a custom pie chart with enhanced labels
    fig = go.Figure()
    
    for i, row in function_counts.iterrows():
        fig.add_trace(go.Pie(
            labels=[row['Function']], 
            values=[row['Count']],
            name=row['Function'],
            marker=dict(
                color=COLOR_MAP.get(row['Function'], '#333333'),
                line=dict(color='black', width=2)
            ),
            text=[f"{row['Percentage']:.1f}%"],
            textinfo='text',
            hoverinfo='text',
            hovertext=[row['HoverInfo']],
            textfont=dict(color='white', size=14, family="Arial Black"),
            textposition='inside',
            showlegend=True
        ))
    
    fig.update_traces(
        hole=0.3,  # Create a donut chart for better text visibility
    )
    
    fig.update_layout(
        title='AI Function Distribution',
        legend=dict(
            title=dict(text='Function'),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        annotations=[
            dict(
                text='AI Functions',
                x=0.5,
                y=0.5,
                font_size=12,
                showarrow=False
            )
        ]
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
    
def create_top_companies_table(df):
    """
    Create a table showing top 10 companies with function usage breakdown
    """
    # Get top 10 companies by total usage
    top_companies = df.groupby('company_name').size().nlargest(10).index.tolist()
    
    # Filter for those companies
    filtered_df = df[df['company_name'].isin(top_companies)]
    
    # Create pivot table with company names as rows and functions as columns
    pivot_df = filtered_df.pivot_table(
        index='company_name',
        columns='aia_feature',
        values='aia_id',
        aggfunc='count',
        fill_value=0
    )
    
    # Add a total column
    pivot_df['Total'] = pivot_df.sum(axis=1)
    
    # Sort by the total
    pivot_df = pivot_df.sort_values('Total', ascending=False)
    
    # Format as a regular dataframe for display
    result_df = pivot_df.reset_index()
    
    return result_df

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
    
def create_total_daily_usage_chart(df):
    """
    Create a total daily usage line chart across all functions
    """
    # Group by date and count total requests per day
    daily_totals = df.groupby('date').size().reset_index(name='count')
    
    fig = px.line(daily_totals, x='date', y='count',
                 markers=True,
                 title='Daily Total AI Function Usage',
                 labels={'count': 'Total Requests', 'date': 'Date'})
    
    # Add a trend line
    fig.add_trace(
        go.Scatter(
            x=daily_totals['date'],
            y=daily_totals['count'].rolling(window=2, min_periods=1).mean(),
            mode='lines',
            line=dict(color='rgba(255, 0, 0, 0.5)', width=2, dash='dash'),
            name='Trend (2-day MA)'
        )
    )
    
    fig.update_layout(
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
    
def create_pdf_report(filtered_df, stats):
    """
    Create a PDF report with all the visualizations
    """
    # Create temporary directory for visualizations
    if not os.path.exists('temp_charts'):
        os.makedirs('temp_charts')
    
    # Create and save matplotlib visualizations for the PDF
    # 1. Function Distribution Chart
    plt.figure(figsize=(8, 5))
    function_counts = filtered_df['aia_feature'].value_counts()
    colors = [COLOR_MAP.get(func, '#333333') for func in function_counts.index]
    
    # Create pie chart with improved visibility
    wedges, texts, autotexts = plt.pie(
        function_counts, 
        labels=function_counts.index, 
        autopct='%1.1f%%', 
        startangle=140, 
        colors=colors,
        wedgeprops=dict(edgecolor='black', linewidth=1)  # Add black outline
    )
    
    # Set font properties for better visibility
    for text in texts + autotexts:
        text.set_fontweight('bold')
        text.set_color('white')  # White text
        text.set_fontsize(10)
        
    # Add shadow effect for better text contrast
    for autotext in autotexts:
        autotext.set_path_effects([
            path_effects.withStroke(linewidth=2, foreground='black')
        ])
        
    plt.axis('equal')
    plt.title('AI Function Distribution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('temp_charts/function_dist.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Daily Usage Chart
    plt.figure(figsize=(10, 5))
    daily_totals = filtered_df.groupby('date').size()
    plt.plot(daily_totals.index, daily_totals.values, marker='o', linewidth=2)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title('Daily Total AI Function Usage', fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Total Requests', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('temp_charts/daily_usage.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Company Usage Bar Chart
    plt.figure(figsize=(10, 6))
    top_companies = filtered_df.groupby('company_name').size().nlargest(10)
    ax = sns.barplot(x=top_companies.values, y=top_companies.index, palette='viridis')
    plt.title('Top 10 Companies by AI Function Usage', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Requests', fontsize=12)
    plt.ylabel('Company Name', fontsize=12)
    plt.tight_layout()
    plt.savefig('temp_charts/company_usage.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. Token Usage Chart
    plt.figure(figsize=(10, 6))
    token_data = filtered_df.groupby('aia_feature').agg({
        'aia_prompt_token': 'sum',
        'aia_completion_token': 'sum'
    })
    token_data.plot(kind='bar', figsize=(10, 6))
    plt.title('Token Usage by Function', fontsize=14, fontweight='bold')
    plt.xlabel('Function', fontsize=12)
    plt.ylabel('Token Count', fontsize=12)
    plt.legend(['Prompt Tokens', 'Completion Tokens'])
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('temp_charts/token_usage.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create PDF document using FPDF
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # Set title
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'AI Functions Analytics Report', 0, 1, 'C')
    pdf.ln(5)
    
    # Add date range
    pdf.set_font('Arial', 'B', 12)
    date_range = f"Report period: {filtered_df['date'].min()} to {filtered_df['date'].max()}"
    pdf.cell(0, 10, date_range, 0, 1)
    pdf.ln(5)
    
    # Add summary stats
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Summary Statistics', 0, 1)
    pdf.set_font('Arial', '', 11)
    pdf.cell(0, 8, f"Total Requests: {stats['total_requests']:,}", 0, 1)
    pdf.cell(0, 8, f"Total Companies: {stats['total_companies']:,}", 0, 1)
    pdf.cell(0, 8, f"Total Users: {stats['total_users']:,}", 0, 1)
    pdf.cell(0, 8, f"Total Tokens: {stats['total_tokens']:,}", 0, 1)
    pdf.cell(0, 8, f"Average Tokens per Request: {stats['avg_tokens_per_request']:.2f}", 0, 1)
    pdf.ln(10)
    
    # Add function distribution chart
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Function Distribution', 0, 1)
    pdf.image('temp_charts/function_dist.png', x=10, w=180)
    pdf.ln(5)
    
    # Add daily usage chart
    pdf.add_page()
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Daily Usage Trend', 0, 1)
    pdf.image('temp_charts/daily_usage.png', x=10, w=180)
    pdf.ln(5)
    
    # Add company usage chart
    pdf.add_page()
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Top Companies by Usage', 0, 1)
    pdf.image('temp_charts/company_usage.png', x=10, w=180)
    pdf.ln(5)
    
    # Function usage distribution table
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Function Usage Distribution', 0, 1)
    pdf.set_font('Arial', 'B', 10)
    
    # Get function counts
    function_counts = filtered_df['aia_feature'].value_counts().reset_index()
    function_counts.columns = ['Function', 'Count']
    
    # Create table header
    pdf.cell(80, 8, 'Function', 1, 0, 'C')
    pdf.cell(40, 8, 'Count', 1, 0, 'C')
    pdf.cell(40, 8, 'Percentage', 1, 1, 'C')
    
    # Create table rows
    pdf.set_font('Arial', '', 10)
    total = function_counts['Count'].sum()
    for _, row in function_counts.iterrows():
        pdf.cell(80, 8, row['Function'], 1, 0)
        pdf.cell(40, 8, str(row['Count']), 1, 0, 'C')
        pdf.cell(40, 8, f"{row['Count']/total*100:.1f}%", 1, 1, 'C')
    pdf.ln(10)
    
    # Add token usage chart and data on the same page
    pdf.add_page()
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Token Usage Analysis', 0, 1)
    pdf.image('temp_charts/token_usage.png', x=10, w=180)
    pdf.ln(5)
    
    # Token usage table
    pdf.set_font('Arial', 'B', 10)
    token_df = filtered_df.groupby('aia_feature').agg({
        'aia_prompt_token': 'sum',
        'aia_completion_token': 'sum',
        'total_tokens': 'sum'
    }).reset_index()
    
    # Create table header
    pdf.cell(60, 8, 'Function', 1, 0, 'C')
    pdf.cell(40, 8, 'Prompt Tokens', 1, 0, 'C')
    pdf.cell(40, 8, 'Completion Tokens', 1, 0, 'C')
    pdf.cell(40, 8, 'Total Tokens', 1, 1, 'C')
    
    # Create table rows
    pdf.set_font('Arial', '', 10)
    for _, row in token_df.iterrows():
        pdf.cell(60, 8, row['aia_feature'], 1, 0)
        pdf.cell(40, 8, f"{row['aia_prompt_token']:,.0f}", 1, 0, 'C')
        pdf.cell(40, 8, f"{row['aia_completion_token']:,.0f}", 1, 0, 'C')
        pdf.cell(40, 8, f"{row['total_tokens']:,.0f}", 1, 1, 'C')
    
    # Add daily stats on a new page
    pdf.add_page()
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Daily Usage Summary', 0, 1)
    pdf.set_font('Arial', 'B', 10)
    
    # Get daily counts
    daily_counts = filtered_df.groupby('date').size().reset_index(name='count')
    daily_counts = daily_counts.sort_values('date')
    
    # Create table header
    pdf.cell(60, 8, 'Date', 1, 0, 'C')
    pdf.cell(60, 8, 'Total Requests', 1, 1, 'C')
    
    # Create table rows
    pdf.set_font('Arial', '', 10)
    for _, row in daily_counts.iterrows():
        pdf.cell(60, 8, str(row['date']), 1, 0, 'C')
        pdf.cell(60, 8, str(row['count']), 1, 1, 'C')
    
    # Add footer with date generated
    pdf.ln(10)
    pdf.set_font('Arial', 'I', 8)
    pdf.cell(0, 10, f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}", 0, 0, 'C')
    
    # Generate PDF
    pdf_output = pdf.output(dest='S').encode('latin1')
    
    # Clean up the temporary charts
    import shutil
    if os.path.exists('temp_charts'):
        shutil.rmtree('temp_charts')
    
    return pdf_output

def load_group_data_from_file():
    """
    Load the embedded group data
    """
    try:
        # This is the embedded group data from the file (updated with new data)
        data = """
S. No.,Company Name,Status,Group ID
1,GoYzer Tech,Active,2677
2,Utopia Corporation,Inactive,2676
3,Yamu Corporation,Inactive,5001
4,Property Shop Investment,Active,5002
5,QA,Active,5003
6,Oryx Management,Active,5004
7,Dubai Islamic Bank,Active,5005
8,Irtikaz Property Management,Inactive,5006
9,PSI 2,Active,5007
10,Mixta,Active,5008
11,Al Forsan Real Estate,Active,5009
12,Prophunters Real Estate,Active,5011
13,Star Real Estate,Inactive,5013
14,Eastern Sands,Active,5014
15,Cherwell,Inactive,5015
16,Better Homes,Active,5016
17,Bhomes,Active,5017
18,Arab Vision,Active,5018
19,Zeus System,Inactive,5019
20,Goyzer Lite,Active,5020
21,Tamani Properties,Active,5021
22,Trx Staging System,Active,5056
23,Savoir,Active,5023
24,Amin & Wilson,Inactive,5024
25,Loka Real Estate,Inactive,5026
26,Carlton,Inactive,5027
27,M&M Real Estate Brokerage,Inactive,5028
28,Chestertons MENA,Active,5029
29,Al Asad Real Estate,Active,5030
30,Elite Estates Real Estate Broker,Active,5031
31,Habibi Properties,Active,5032
32,Powerhouse Real Estate,Active,5033
33,Leon Properties,Active,5034
34,Linda Real Estate,Active,5097
35,Ascot and Co,Active,5036
36,CRC Real Estate,Active,5037
37,Casabella Property Broker LLC,Active,5038
38,Emtelak Properties,Active,5039
39,Stars and Sands Properties,Active,5040
40,Goyzer Own Group,Active,5041
41,Home Space Realtors,Active,5042
42,Colliers,Active,5043
43,Strada UAE,Active,5044
44,EVA Real Estate LLC,Active,5045
45,Positive Properties,Active,5046
46,Luke Stays Luxury Real Estate,Active,5047
47,Jade & Co Real Estate,Active,5048
48,Emirates Post Group (EPG),Active,5049
49,United Plaza Real Estate,Active,5050
50,Zealway Real Estate,Active,5051
51,The Urban Nest Real Estate LLC,Active,5052
52,Sustainable Homes Real Estate,Active,5053
53,First Point Real Estate Brokerage,Active,5054
54,Best Luxury Properties,Active,5055
55,Palma Real Estate,Active,5057
56,National Homes Real Estate,Active,5058
57,Williams International,Active,5059
58,Homes Jordan (Hummer Contracting),Active,5060
59,Dubai Sport City,Active,5061
60,Christies International Real Estate,Active,5062
61,BlackBrick Property LLC,Active,5063
62,Irwin REal Estate,Active,5064
63,Mayfield International Real Estate,Active,5065
64,"Ports, Customs and Free Zone Corporation",Active,5066
65,Blue Coast Real Estate,Active,5067
66,Investalet,Active,5068
67,Point Blank Properties,Active,5069
68,Zed Capital Real Estate,Active,5070
69,Buy Own House Property,Active,5071
70,Moonstay Real Estate,Active,5072
71,Sino Gulf,Active,5073
72,Roya Solutions Group,Active,5074
73,Savills Middle East,Active,5075
74,Haus and Haus,Active,5076
75,City Luxe Real Estate,Active,5077
76,La Capitale Real Estate,Active,5078
77,D&B Properties,Active,5079
78,Prime Location Property Real Estate,Active,5080
79,Bramwell & Partners Real Estate,Active,5081
80,The Luxury Collection Real Estate,Active,5082
81,The Noble House Real Estate,Active,5083
82,Range International,Active,5084
83,Aqua Properties,Active,5085
84,Worldfield Real Estate,Active,5086
85,Family Homes Real Estate L.L.C,Active,5087
86,DB Dedicated for Trainings.,Active,5088
87,Ahmadyar Developments,Active,5089
88,IQ Pro Real Estate,Active,5091
89,Locke Lifestyle Properties,Active,5092
90,V serve Real Estate LLC,Active,5093
91,Luxury Escape,Active,5094
92,Luxury Invest,Active,5095
93,Apex Real Estate,Active,5096
94,Linda Real Estate,Active,5097
95,Pulse Real Estate,Active,5098
96,Horizon Vista Real Estate,Active,5099
97,Sequoia Properties,Active,5101
98,Keller Williams,Active,5102
99,Key One Realty Group,Active,5103
100,Mayak Real Estate,Active,5104
101,Property Matters,Active,5105
102,Fifty Two Real Estate,Active,5106
103,Signature Homes,Active,5107
104,Manresa Real Estate,Active,5108
105,Pangea Properties,Active,5109
106,Amaal Emirates,Active,5110
107,Henry Wiltshire International,Active,5111
108,Kevo Living,Active,5112
109,Remal Properties,Active,5113
110,Al Mamzar Real Estate And Commercial Broker,Active,5114
111,Better Homes Qatar,Active,5115
112,Wakhan Properties,Active,5116
113,AJ Gargash Real Estate Development,Active,5118
114,Mohammad Al Sayed Ibrahim,Active,5119
115,VIP Luxury Homes Real Estate,Active,5120
116,Republik Real Estate (Abu Dhabi),Active,5121
117,Stage Properties,Active,5122
118,Acres & Squares Real Estate,Active,5123
119,Al Ansari Real Estate Development LLC,Active,5124
120,Bhomes DSC,Active,5126
121,Audley's International Real Estate,Active,5125
122,Cavendish Maxwell,Active,5127
123,Al Ain Properties LLC,Active,5128
124,R H E M Properties L.L.C,Active,5129
125,SunShine Properties Real Estate Broker,Active,5130
126,Capstone Real Estate L.L.C,Active,5131
127,Phoenix Homes Real Estate,Active,5133
128,Royce International Realty,Active,5134
129,Roots Heritage,Active,5135
130,Tulip-al Thahabi,Active,5136
131,Xpotential Real Estate Brokers,Active,5137
132,Avenew Real Estate,Active,5138
133,PuraNova Properties LLC,Active,5139
134,Union Square House Real Estate,Active,5140
135,PropBay Universal Real Estate Brokerage,Active,5141
136,Lusso Real Estate,Active,5142
137,Luxbridge International Realty,Active,5143
138,MNA Properties,Active,5144
139,Elton Developers,Active,5145
140,Harbor Real Estate,Active,5146
"""
        # Write data to a temporary file
        with open("temp_group_data.csv", "w") as f:
            f.write(data)
            
        # Read the data with pandas - no need to skip rows as the data has a header now
        df = pd.read_csv("temp_group_data.csv")
        
        # Rename columns for easier merging
        df = df.rename(columns={'Group ID': 'group_id', 'Company Name': 'company_name'})
        
        # Clean up the data (remove row with duplicate company)
        df = df.drop_duplicates(subset=['group_id'])
        
        # Clean up the temp file
        if os.path.exists("temp_group_data.csv"):
            os.remove("temp_group_data.csv")
            
        return df
    except Exception as e:
        st.error(f"Error loading embedded group data: {str(e)}")
        return None
