# cti_dashboard_app.py
# Import necessary libraries
import streamlit as st
import pandas as pd
import io
import matplotlib.pyplot as plt
import numpy as np

# --- Page Configuration ---
# THIS MUST BE THE FIRST STREAMLIT COMMAND IN YOUR SCRIPT.
# No other Streamlit calls or decorated function definitions should come before this.
st.set_page_config(
    page_title="Western Ghats CTI Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://docs.streamlit.io/library/api-reference/utilities/st.set_page_config',
        'Report a bug': "https://github.com/streamlit/streamlit/issues",
        'About': "# Western Ghats CTI Dashboard\nThis app visualizes Carbon Threat Index data."
    }
)

# --- Data Loading and Preparation ---

# Original CSV data as a string
# This data represents the Mean Carbon Threat Index (CTI) values for various districts.
csv_data = """DISTRICT,MEAN of CTI Values
Bharuch,0.259892591
Dhule,0.261203601
Surat,0.26806481
The Dangs,0.272367267
Valsad,0.274810943
Nashik,0.273153516
Daman,0.306524468
Dadra and Nagar Haveli,0.294713376
Thane,0.29444006
Pune,0.283631128
Greater Bombay,0.303004433
Raigarh,0.282747718
Satara,0.246013637
Ratnagiri,0.261931699
Sangli,0.213959775
Kolhapur,0.249949896
Belgaum,0.274923674
Sindhudurg,0.266855446
Dharwad,0.266535067
North Goa,0.283291686
Uttar Kannad,0.273648205
South Goa,0.282756304
Chitradurga,0.255591444
Shimoga,0.263932423
Dakshin Kannad,0.275007489
Chikmagalur,0.251801356
Hassan,0.263609686
Mandya,0.256091849
Kodagu,0.255556614
Kasaragod,0.291912428
Mysore,0.263661651
Kannur,0.306408558
Wayanad,0.284082034
Periyar,0.263816327
Kozhikode,0.298212863
Mahe,0.301569067
Nilgiri,0.257504608
Malappuram,0.29158154
Coimbatore,0.275694535
Palakkad,0.287808081
Thrissur,0.302758597
Dindigul Anna,0.263514173
Madurai,0.253523706
Idukki,0.256070388
Ernakulam,0.300454367
Alappuzha,0.298052074
Kottayam,0.304470732
Kamarajar,0.238371457
Pathanamthitta,0.268565741
Tirunelveli Kattabo,0.242296729
Kollam,0.277513973
Thiruvananthapuram,0.305039833
Kanniyakumari,0.298379038"""

@st.cache_data # Use Streamlit's caching to load data only once for better performance
def load_data():
    """
    Loads and preprocesses the CTI data.
    Returns:
        pandas.DataFrame: Processed DataFrame with District and CTI_Value.
    """
    # Use io.StringIO to treat the string data as a file
    file_content_io = io.StringIO(csv_data)
    # Read the CSV data into a pandas DataFrame
    df = pd.read_csv(file_content_io)
    # Rename columns for easier access
    df.columns = ['District', 'CTI_Value']
    # Ensure CTI_Value is numeric
    df['CTI_Value'] = pd.to_numeric(df['CTI_Value'])
    return df

# Load the data
df_data = load_data()

# --- Sidebar ---
# Add a header and descriptive text to the sidebar
st.sidebar.header("Dashboard Filters & Options")
st.sidebar.markdown("""
This dashboard visualizes the Mean Carbon Threat Index (CTI) for various districts in the Western Ghats.
A higher CTI value generally suggests a greater potential threat to carbon stocks or the carbon sequestration capacity of a district.
Use the options in the main panel to explore the data.
""")
st.sidebar.info("Data Source: Provided district-wise CTI values.")

# --- Main Page ---
st.title("📊 Western Ghats Carbon Threat Index (CTI) Dashboard")
st.markdown("An interactive overview of the mean Carbon Threat Index values across districts.")

# --- Data Analysis and Calculations ---
# Calculate basic descriptive statistics for CTI values
min_cti_val = df_data['CTI_Value'].min()
max_cti_val = df_data['CTI_Value'].max()
mean_cti_val = df_data['CTI_Value'].mean()
median_cti_val = df_data['CTI_Value'].median()
std_cti_val = df_data['CTI_Value'].std()
num_districts = len(df_data)

# Sort the DataFrame by CTI_Value in descending order for rankings and default displays
df_sorted = df_data.sort_values(by='CTI_Value', ascending=False).reset_index(drop=True)

# Create a display version of the sorted DataFrame with index starting from 1
df_sorted_display = df_sorted.copy()
df_sorted_display.index = np.arange(1, len(df_sorted_display) + 1)


# Define Threat Categories based on CTI quartiles
q1 = df_data['CTI_Value'].quantile(0.25)  # 1st Quartile
q2 = df_data['CTI_Value'].quantile(0.50)  # 2nd Quartile (Median)
q3 = df_data['CTI_Value'].quantile(0.75)  # 3rd Quartile

def categorize_cti(cti_value):
    """
    Categorizes a CTI value into a threat level.
    Args:
        cti_value (float): The CTI value.
    Returns:
        str: The threat category ('Low Threat', 'Medium-Low Threat', 'Medium-High Threat', 'High Threat').
    """
    if cti_value > q3:
        return "High Threat"
    elif cti_value > q2:
        return "Medium-High Threat"
    elif cti_value > q1:
        return "Medium-Low Threat"
    else:
        return "Low Threat"

# Apply the categorization to the display DataFrame
df_sorted_display['Threat_Category'] = df_sorted_display['CTI_Value'].apply(categorize_cti)


# --- Display Overall Statistics ---
st.header("📈 Overall CTI Statistics")
# Use columns for a cleaner layout of metrics
col1, col2, col3 = st.columns(3)
col1.metric("Total Districts Analyzed", num_districts)
col1.metric("Mean CTI Value", f"{mean_cti_val:.4f}") # Format to 4 decimal places
col2.metric("Median CTI Value", f"{median_cti_val:.4f}")
col2.metric("Standard Deviation", f"{std_cti_val:.4f}")
# Add help text to min/max metrics to show which district has that value
min_district = df_data.loc[df_data['CTI_Value'].idxmin(), 'District']
max_district = df_data.loc[df_data['CTI_Value'].idxmax(), 'District']
col3.metric("Minimum CTI Value", f"{min_cti_val:.4f}", help=f"District: {min_district}")
col3.metric("Maximum CTI Value", f"{max_cti_val:.4f}", help=f"District: {max_district}")


# --- Display Top & Bottom Districts ---
st.header("🏆 Top & Bottom Districts by CTI")
# Add a slider to control how many top/bottom districts are shown
num_top_bottom = st.slider(
    "Select number of Top/Bottom districts to display:",
    min_value=1,
    max_value=10, # Allow up to top/bottom 10
    value=5,      # Default to 5
    key="top_bottom_slider" # Unique key for the widget
)

# Get top N districts from the display-indexed DataFrame
top_n_districts = df_sorted_display.head(num_top_bottom)

# Get bottom N districts:
# 1. Take from the original sorted DataFrame (0-indexed)
# 2. Sort them ascendingly by CTI
# 3. Apply threat categorization
# 4. Reset index for display
bottom_n_districts_data = df_sorted.tail(num_top_bottom).sort_values(by='CTI_Value', ascending=True)
bottom_n_districts_data['Threat_Category'] = bottom_n_districts_data['CTI_Value'].apply(categorize_cti)
bottom_n_districts_data.index = np.arange(1, len(bottom_n_districts_data) + 1)


# Display top and bottom districts side-by-side
col1_rank, col2_rank = st.columns(2)
with col1_rank:
    st.subheader(f"Top {num_top_bottom} Districts (Highest CTI)")
    # Format CTI_Value in the DataFrame display
    st.dataframe(top_n_districts[['District', 'CTI_Value', 'Threat_Category']].style.format({'CTI_Value': "{:.4f}"}))
with col2_rank:
    st.subheader(f"Bottom {num_top_bottom} Districts (Lowest CTI)")
    st.dataframe(bottom_n_districts_data[['District', 'CTI_Value', 'Threat_Category']].style.format({'CTI_Value': "{:.4f}"}))


# --- Bar Chart of CTI Values ---
st.header("📊 CTI Values by District")

# Add an option to sort the bar chart
sort_order_chart = st.selectbox(
    "Sort chart by CTI Value:",
    ["Descending (High to Low)", "Ascending (Low to High)"],
    index=0, # Default to descending
    key="chart_sort_select"
)
# Sort the DataFrame for the chart based on user selection
df_chart_sorted = df_sorted_display.sort_values(
    by='CTI_Value',
    ascending=(sort_order_chart == "Ascending (Low to High)")
)

# Create the bar chart using Matplotlib
fig, ax = plt.subplots(figsize=(10, 14)) # Adjust figure size for better readability in Streamlit
bars_obj = ax.barh(df_chart_sorted['District'], df_chart_sorted['CTI_Value'], color='darkcyan')
ax.set_xlabel('Mean Carbon Threat Index (CTI) Value')
ax.set_ylabel('District')
ax.set_title(f'Mean CTI by District in Western Ghats ({num_districts} Districts)')

# Invert y-axis if sorting descending to show highest CTI at the top
if sort_order_chart == "Descending (High to Low)":
    ax.invert_yaxis()
ax.grid(axis='x', linestyle='--', alpha=0.7) # Add a light grid for easier value reading

# Add CTI values as text labels on the bars
for bar_item in bars_obj: # Use a different variable name for the loop item
    ax.text(
        bar_item.get_width() + 0.0005, # x-position (slightly offset from bar end)
        bar_item.get_y() + bar_item.get_height() / 2, # y-position (center of bar)
        f'{bar_item.get_width():.4f}', # Text label (formatted CTI value)
        va='center',
        ha='left',
        fontsize=7
    )
ax.set_xlim(0, max_cti_val * 1.18) # Adjust x-axis limit to make space for labels
plt.tight_layout() # Adjust plot to ensure everything fits without overlapping
st.pyplot(fig) # Display the Matplotlib figure in Streamlit


# --- Full Data Table with Threat Categories ---
st.header("📋 All Districts Data with Threat Categories")

# Add a filter for Threat Category
# Ensure 'All' is the first option and unique categories follow in a specific order
unique_threat_categories = df_sorted_display['Threat_Category'].unique().tolist()
# Define a preferred order for categories
preferred_order = ["High Threat", "Medium-High Threat", "Medium-Low Threat", "Low Threat"]
ordered_unique_categories = [cat for cat in preferred_order if cat in unique_threat_categories]
threat_categories_all = ["All"] + ordered_unique_categories

selected_category = st.selectbox(
    "Filter by Threat Category:",
    threat_categories_all,
    index=0, # Default to "All"
    key="category_filter_select"
)

# Filter the DataFrame based on selection
if selected_category == "All":
    df_display_filtered = df_sorted_display
else:
    df_display_filtered = df_sorted_display[df_sorted_display['Threat_Category'] == selected_category]

# Add options to sort the full data table
sort_column = st.selectbox(
    "Sort data by:",
    ["CTI_Value", "District", "Threat_Category"], # Columns available for sorting
    index=0, # Default to sorting by CTI_Value
    key="table_sort_column_select"
)
# Checkbox for ascending/descending sort order
# Default to descending for CTI_Value, ascending for others
default_sort_ascending = False if sort_column == "CTI_Value" else True
sort_ascending = st.checkbox(
    "Sort Ascending",
    value=default_sort_ascending,
    key="table_sort_order_checkbox"
)

# Sort the filtered DataFrame
df_display_final_sorted = df_display_filtered.sort_values(by=sort_column, ascending=sort_ascending)
# Reset index for continuous numbering after filtering and sorting for display
df_display_final_sorted.index = np.arange(1, len(df_display_final_sorted) + 1)

# Display the final sorted and filtered DataFrame
st.dataframe(
    df_display_final_sorted[['District', 'CTI_Value', 'Threat_Category']].style.format({'CTI_Value': "{:.4f}"}),
    height=600, # Set a fixed height for the scrollable table
    use_container_width=True # Make the table use the full container width
)


# --- Threat Category Definitions (Collapsible) ---
with st.expander("Threat Category Definitions (based on CTI Quartiles)"):
    st.markdown(f"- **Low Threat:** CTI <= {q1:.4f} (Lowest 25% of districts)")
    st.markdown(f"- **Medium-Low Threat:** {q1:.4f} < CTI <= {q2:.4f} (25th to 50th percentile)")
    st.markdown(f"- **Medium-High Threat:** {q2:.4f} < CTI <= {q3:.4f} (50th to 75th percentile)")
    st.markdown(f"- **High Threat:** CTI > {q3:.4f} (Highest 25% of districts)")

# --- Footer ---
st.markdown("---")
st.markdown("Dashboard created using Python and Streamlit.")
st.markdown(f"Last data refresh: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} (Note: Data is static in this script)")
