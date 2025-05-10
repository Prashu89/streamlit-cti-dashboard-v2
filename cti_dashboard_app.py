# cti_dashboard_app.py
# Import necessary libraries
import streamlit as st
import pandas as pd
import io
import matplotlib.pyplot as plt
import numpy as np
import rasterio # For reading raster files
from rasterio.plot import show as rio_show # For plotting raster data
import os # To check if the file path exists

# --- Page Configuration ---
st.set_page_config(
    page_title="Western Ghats CTI Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://docs.streamlit.io/library/api-reference/utilities/st.set_page_config',
        'Report a bug': "https://github.com/streamlit/streamlit/issues",
        'About': "# Western Ghats CTI Dashboard\nThis app visualizes Carbon Threat Index data and a pre-defined CTI map."
    }
)

# --- Define Local Raster File Path ---
# IMPORTANT: This is a hardcoded path. The Streamlit app will only find this file
# if it's run on a machine where this exact path is valid.
# Using raw string for Windows path compatibility
RASTER_FILE_PATH = r"D:\wg_cti_project\data_proc\cti\CTI MASked\cti_masked.tif"

# --- Data Loading and Preparation (CSV Data) ---
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
def load_csv_data():
    """
    Loads and preprocesses the CTI CSV data.
    Returns:
        pandas.DataFrame: Processed DataFrame with District and CTI_Value.
    """
    file_content_io = io.StringIO(csv_data)
    df = pd.read_csv(file_content_io)
    df.columns = ['District', 'CTI_Value']
    df['CTI_Value'] = pd.to_numeric(df['CTI_Value'])
    return df

# Load the CSV data
df_data = load_csv_data()

# --- Sidebar ---
st.sidebar.header("Dashboard Filters & Options")
st.sidebar.markdown("""
This dashboard visualizes the Mean Carbon Threat Index (CTI) for various districts in the Western Ghats
and displays a pre-loaded CTI raster map.
A higher CTI value generally suggests a greater potential threat to carbon stocks or the carbon sequestration capacity of a district.
""")
st.sidebar.info(f"Attempting to load raster map from: {RASTER_FILE_PATH}")

# --- Main Page ---
st.title("🌍 Western Ghats Carbon Threat Index (CTI) Dashboard")
st.markdown("An interactive overview of the mean Carbon Threat Index values across districts and a CTI map viewer.")

# --- Display CTI Raster Map ---
st.header("🗺️ CTI Raster Map")
if os.path.exists(RASTER_FILE_PATH):
    try:
        with rasterio.open(RASTER_FILE_PATH) as src:
            # Create a figure to display the raster
            fig_map, ax_map = plt.subplots(figsize=(10, 10)) # Adjust size as needed

            # Use rasterio.plot.show to display the raster
            # You can customize cmap (colormap) as needed, e.g., 'viridis', 'plasma', 'coolwarm', 'RdYlGn'
            # For CTI, a diverging colormap might be good if values represent deviation from a mean,
            # or a sequential colormap if they represent intensity.
            rio_show(src, ax=ax_map, cmap='viridis', title="CTI Raster Map")

            # Add some details about the raster
            st.caption(f"Displaying raster: {os.path.basename(RASTER_FILE_PATH)}. Bands: {src.count}. CRS: {src.crs}. Dimensions: {src.width}x{src.height}")
            st.pyplot(fig_map)

    except Exception as e:
        st.error(f"Error displaying raster file '{RASTER_FILE_PATH}': {e}")
        st.error("Please ensure the file is a valid GeoTIFF or similar raster format that rasterio can read, and the path is correct.")
else:
    st.warning(f"Raster file not found at the specified path: {RASTER_FILE_PATH}")
    st.info("Please ensure the CTI raster map file exists at the correct location for it to be displayed.")


# --- Data Analysis and Calculations (from CSV) ---
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
q1 = df_data['CTI_Value'].quantile(0.25)
q2 = df_data['CTI_Value'].quantile(0.50)
q3 = df_data['CTI_Value'].quantile(0.75)

def categorize_cti(cti_value):
    """
    Categorizes a CTI value into a threat level.
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


# --- Display Overall Statistics (from CSV) ---
st.header("📊 Overall CTI Statistics (District Data)")
col1, col2, col3 = st.columns(3)
col1.metric("Total Districts Analyzed", num_districts)
col1.metric("Mean CTI Value", f"{mean_cti_val:.4f}")
col2.metric("Median CTI Value", f"{median_cti_val:.4f}")
col2.metric("Standard Deviation", f"{std_cti_val:.4f}")
min_district = df_data.loc[df_data['CTI_Value'].idxmin(), 'District']
max_district = df_data.loc[df_data['CTI_Value'].idxmax(), 'District']
col3.metric("Minimum CTI Value", f"{min_cti_val:.4f}", help=f"District: {min_district}")
col3.metric("Maximum CTI Value", f"{max_cti_val:.4f}", help=f"District: {max_district}")


# --- Display Top & Bottom Districts (from CSV) ---
st.header("🏆 Top & Bottom Districts by CTI (District Data)")
num_top_bottom = st.slider(
    "Select number of Top/Bottom districts to display:",
    min_value=1,
    max_value=10,
    value=5,
    key="top_bottom_slider"
)

top_n_districts = df_sorted_display.head(num_top_bottom)
bottom_n_districts_data = df_sorted.tail(num_top_bottom).sort_values(by='CTI_Value', ascending=True)
bottom_n_districts_data['Threat_Category'] = bottom_n_districts_data['CTI_Value'].apply(categorize_cti)
bottom_n_districts_data.index = np.arange(1, len(bottom_n_districts_data) + 1)

col1_rank, col2_rank = st.columns(2)
with col1_rank:
    st.subheader(f"Top {num_top_bottom} Districts (Highest CTI)")
    st.dataframe(top_n_districts[['District', 'CTI_Value', 'Threat_Category']].style.format({'CTI_Value': "{:.4f}"}))
with col2_rank:
    st.subheader(f"Bottom {num_top_bottom} Districts (Lowest CTI)")
    st.dataframe(bottom_n_districts_data[['District', 'CTI_Value', 'Threat_Category']].style.format({'CTI_Value': "{:.4f}"}))


# --- Bar Chart of CTI Values (from CSV) ---
st.header("📈 CTI Values by District (District Data)")
sort_order_chart = st.selectbox(
    "Sort chart by CTI Value:",
    ["Descending (High to Low)", "Ascending (Low to High)"],
    index=0,
    key="chart_sort_select"
)
df_chart_sorted = df_sorted_display.sort_values(
    by='CTI_Value',
    ascending=(sort_order_chart == "Ascending (Low to High)")
)

fig_bar, ax_bar = plt.subplots(figsize=(10, 14))
bars_obj = ax_bar.barh(df_chart_sorted['District'], df_chart_sorted['CTI_Value'], color='darkcyan')
ax_bar.set_xlabel('Mean Carbon Threat Index (CTI) Value')
ax_bar.set_ylabel('District')
ax_bar.set_title(f'Mean CTI by District in Western Ghats ({num_districts} Districts)')
if sort_order_chart == "Descending (High to Low)":
    ax_bar.invert_yaxis()
ax_bar.grid(axis='x', linestyle='--', alpha=0.7)
for bar_item in bars_obj:
    ax_bar.text(
        bar_item.get_width() + 0.0005,
        bar_item.get_y() + bar_item.get_height() / 2,
        f'{bar_item.get_width():.4f}',
        va='center',
        ha='left',
        fontsize=7
    )
ax_bar.set_xlim(0, max_cti_val * 1.18) # Adjust x-axis limit for labels
plt.tight_layout()
st.pyplot(fig_bar)


# --- Full Data Table with Threat Categories (from CSV) ---
st.header("📋 All Districts Data with Threat Categories (District Data)")
unique_threat_categories = df_sorted_display['Threat_Category'].unique().tolist()
preferred_order = ["High Threat", "Medium-High Threat", "Medium-Low Threat", "Low Threat"]
ordered_unique_categories = [cat for cat in preferred_order if cat in unique_threat_categories]
threat_categories_all = ["All"] + ordered_unique_categories

selected_category = st.selectbox(
    "Filter by Threat Category:",
    threat_categories_all,
    index=0,
    key="category_filter_select"
)

if selected_category == "All":
    df_display_filtered = df_sorted_display
else:
    df_display_filtered = df_sorted_display[df_sorted_display['Threat_Category'] == selected_category]

sort_column = st.selectbox(
    "Sort data by:",
    ["CTI_Value", "District", "Threat_Category"],
    index=0,
    key="table_sort_column_select"
)
default_sort_ascending = False if sort_column == "CTI_Value" else True
sort_ascending = st.checkbox(
    "Sort Ascending",
    value=default_sort_ascending,
    key="table_sort_order_checkbox"
)

df_display_final_sorted = df_display_filtered.sort_values(by=sort_column, ascending=sort_ascending)
df_display_final_sorted.index = np.arange(1, len(df_display_final_sorted) + 1)

st.dataframe(
    df_display_final_sorted[['District', 'CTI_Value', 'Threat_Category']].style.format({'CTI_Value': "{:.4f}"}),
    height=600,
    use_container_width=True
)


# --- Threat Category Definitions (Collapsible) ---
with st.expander("Threat Category Definitions (based on CTI Quartiles from District Data)"):
    st.markdown(f"- **Low Threat:** CTI <= {q1:.4f} (Lowest 25% of districts)")
    st.markdown(f"- **Medium-Low Threat:** {q1:.4f} < CTI <= {q2:.4f} (25th to 50th percentile)")
    st.markdown(f"- **Medium-High Threat:** {q2:.4f} < CTI <= {q3:.4f} (50th to 75th percentile)")
    st.markdown(f"- **High Threat:** CTI > {q3:.4f} (Highest 25% of districts)")

# --- Footer ---
st.markdown("---")
st.markdown("Dashboard created using Python and Streamlit.")
st.markdown(f"CSV Data Last Refreshed: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} (Note: CSV data is static in this script)")
st.markdown(f"Raster Map Path: {RASTER_FILE_PATH}")

