import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io
import base64
import os
from utils import *

# Set page configuration
st.set_page_config(
    page_title="AI Functions Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS
def load_css():
    with open("styles.css") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css()

# Main title
st.markdown('<h1 class="main-title">AI Functions Analytics Dashboard</h1>', unsafe_allow_html=True)

# Initialize session state for uploaded data
if 'analytics_data' not in st.session_state:
    st.session_state.analytics_data = None
if 'group_data' not in st.session_state:
    # Load the embedded group data on first run
    st.session_state.group_data = load_group_data_from_file()
if 'merged_data' not in st.session_state:
    st.session_state.merged_data = None
if 'filtered_data' not in st.session_state:
    st.session_state.filtered_data = None

# Create two tabs for data upload and dashboard
tab1, tab2 = st.tabs(["📊 Dashboard", "📂 Data Upload"])

# Data Upload Tab
with tab2:
    st.markdown('<h2 class="section-title">Data Upload</h2>', unsafe_allow_html=True)
    st.write("Upload your AI analytics data to populate the dashboard. The company group names are already embedded.")
    
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.markdown('<h3>AI Analytics Data</h3>', unsafe_allow_html=True)
    st.write("Upload the AI functions usage data (CSV format)")
    analytics_file = st.file_uploader("Choose a file", type=['csv'], key="analytics_uploader")
    
    if analytics_file is not None:
        analytics_df = load_and_process_analytics_data(analytics_file)
        if analytics_df is not None:
            st.session_state.analytics_data = analytics_df
            st.success(f"✅ Analytics data loaded successfully! ({len(analytics_df)} records)")
            
            # Display sample data
            st.write("Sample data:")
            st.dataframe(analytics_df.head(5))
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display group data information
    if st.session_state.group_data is not None:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.markdown('<h3>Group Name Data</h3>', unsafe_allow_html=True)
        st.info(f"✅ Group data is embedded in the dashboard. ({len(st.session_state.group_data)} companies)")
        
        # Display sample data
        st.write("Sample data:")
        st.dataframe(st.session_state.group_data.head(5))
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Merge datasets if analytics data is available
    if st.session_state.analytics_data is not None and st.session_state.group_data is not None:
        merged_df = merge_datasets(st.session_state.analytics_data, st.session_state.group_data)
        if merged_df is not None:
            st.session_state.merged_data = merged_df
            st.session_state.filtered_data = merged_df.copy()
            st.success("✅ Data merged successfully! You can now view the dashboard.")
            
            # Download merged data option
            csv = merged_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="merged_ai_analytics.csv" class="custom-button">Download Merged Data</a>'
            st.markdown(href, unsafe_allow_html=True)

# Dashboard Tab
with tab1:
    if st.session_state.merged_data is not None:
        df = st.session_state.merged_data
        
        # Sidebar for filters
        st.sidebar.markdown('<h2 class="section-title">Filters</h2>', unsafe_allow_html=True)
        
        # Date range filter
        min_date, max_date = get_date_range(df)
        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        if len(date_range) == 2:
            start_date, end_date = date_range
            filtered_df = filter_data_by_date(df, start_date, end_date)
        else:
            filtered_df = df.copy()
        
        # Company filter
        all_companies = ['All Companies'] + sorted(df['company_name'].unique().tolist())
        selected_companies = st.sidebar.multiselect("Select Companies", all_companies, default='All Companies')
        filtered_df = filter_data_by_company(filtered_df, selected_companies)
        
        # Function filter
        all_functions = ['All Functions'] + sorted(df['aia_feature'].unique().tolist())
        selected_functions = st.sidebar.multiselect("Select Functions", all_functions, default='All Functions')
        filtered_df = filter_data_by_function(filtered_df, selected_functions)
        
        # Update filtered data in session state
        st.session_state.filtered_data = filtered_df
        
        # Main dashboard content
        stats = get_summary_stats(filtered_df)
        
        # Summary statistics cards with colored boxes
        st.markdown('<h2 class="section-title">Summary Statistics</h2>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            color = STATS_COLORS['Total Requests']
            st.markdown(f'''
            <div class="stats-card" style="border-left: 5px solid {color}; background: linear-gradient(to right, {color}10, white)">
                <h3>Total Requests</h3>
                <p style="color: {color};">{stats["total_requests"]:,}</p>
            </div>
            ''', unsafe_allow_html=True)
        
        with col2:
            color = STATS_COLORS['Total Companies']
            st.markdown(f'''
            <div class="stats-card" style="border-left: 5px solid {color}; background: linear-gradient(to right, {color}10, white)">
                <h3>Total Companies</h3>
                <p style="color: {color};">{stats["total_companies"]:,}</p>
            </div>
            ''', unsafe_allow_html=True)
        
        with col3:
            color = STATS_COLORS['Total Users']
            st.markdown(f'''
            <div class="stats-card" style="border-left: 5px solid {color}; background: linear-gradient(to right, {color}10, white)">
                <h3>Total Users</h3>
                <p style="color: {color};">{stats["total_users"]:,}</p>
            </div>
            ''', unsafe_allow_html=True)
        
        with col4:
            color = STATS_COLORS['Total Tokens']
            st.markdown(f'''
            <div class="stats-card" style="border-left: 5px solid {color}; background: linear-gradient(to right, {color}10, white)">
                <h3>Total Tokens</h3>
                <p style="color: {color};">{stats["total_tokens"]:,}</p>
            </div>
            ''', unsafe_allow_html=True)
        
        # New total daily usage chart
        st.markdown('<h2 class="section-title">Overall Usage Trend</h2>', unsafe_allow_html=True)
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        total_daily_chart = create_total_daily_usage_chart(filtered_df)
        st.plotly_chart(total_daily_chart, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # First row of charts
        st.markdown('<h2 class="section-title">Usage by Function</h2>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            daily_chart = create_daily_usage_chart(filtered_df)
            st.plotly_chart(daily_chart, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            dist_chart = create_function_distribution_chart(filtered_df)
            st.plotly_chart(dist_chart, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Second row of charts
        st.markdown('<h2 class="section-title">Company Analysis</h2>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            company_chart = create_company_usage_chart(filtered_df)
            st.plotly_chart(company_chart, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            heatmap_chart = create_company_function_heatmap(filtered_df)
            st.plotly_chart(heatmap_chart, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        # Top 10 Companies with function breakdown table
        st.markdown('<h3 class="subsection-title">Top 10 Companies with Function Usage Breakdown</h3>', unsafe_allow_html=True)
        companies_table = create_top_companies_table(filtered_df)
        st.dataframe(companies_table, use_container_width=True, height=400)
        
        # Third row of charts
        st.markdown('<h2 class="section-title">Token Analysis</h2>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            token_chart = create_token_usage_chart(filtered_df)
            st.plotly_chart(token_chart, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            avg_token_chart = create_average_token_bar_chart(filtered_df)
            st.plotly_chart(avg_token_chart, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Fourth row of charts
        st.markdown('<h2 class="section-title">User Patterns</h2>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            user_chart = create_user_activity_chart(filtered_df)
            st.plotly_chart(user_chart, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            hourly_chart = create_hourly_usage_chart(filtered_df)
            st.plotly_chart(hourly_chart, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Export options
        st.markdown('<h2 class="section-title">Export Options</h2>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            # Download filtered data as CSV
            csv = filtered_df.to_csv(index=False)
            b64_csv = base64.b64encode(csv.encode()).decode()
            st.markdown(
                f'<a href="data:file/csv;base64,{b64_csv}" download="ai_analytics.csv" class="custom-button">Download CSV Data</a>',
                unsafe_allow_html=True
            )
        
        with col2:
            # Generate and download PDF report
            if st.button("Generate PDF Report", type="primary"):
                with st.spinner("Generating PDF report..."):
                    try:
                        pdf_data = create_pdf_report(filtered_df, stats)
                        b64_pdf = base64.b64encode(pdf_data).decode()
                        st.markdown(
                            f'<a href="data:application/pdf;base64,{b64_pdf}" download="ai_analytics_report.pdf" class="custom-button">Download PDF Report</a>',
                            unsafe_allow_html=True
                        )
                        st.success("PDF report generated successfully!")
                    except Exception as e:
                        st.error(f"Error generating PDF report: {str(e)}")
        
        # Data table view
        with st.expander("View Data Table"):
            st.dataframe(filtered_df)
    
    else:
        st.info("👈 Please upload the AI analytics data in the Data Upload tab to view the dashboard.")
        
        # Sample visualizations using placeholder data
        st.markdown('<h2 class="section-title">Dashboard Preview</h2>', unsafe_allow_html=True)
        st.write("This is a preview of what the dashboard will look like once data is uploaded.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image("https://via.placeholder.com/600x400.png?text=Daily+Usage+Chart", use_container_width=True)
        
        with col2:
            st.image("https://via.placeholder.com/600x400.png?text=Function+Distribution+Chart", use_container_width=True)

# Footer
st.markdown("---")
st.markdown('<p style="text-align: center;">AI Functions Analytics Dashboard © 2025</p>', unsafe_allow_html=True)
