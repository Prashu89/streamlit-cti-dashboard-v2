# cti_dashboard_app.py
# Import necessary libraries
import streamlit as st
import pandas as pd
import io
import matplotlib.pyplot as plt # Still used for bar chart and colormap generation
import numpy as np
import rasterio # For reading raster files
from rasterio.warp import calculate_default_transform, reproject, Resampling, transform_bounds # For reprojection
import os # To check if the file path exists
import folium # For creating interactive maps
from streamlit_folium import folium_static # To display folium maps in Streamlit
import branca.colormap as cm # For creating colorbars/legends for folium

# --- Page Configuration ---
st.set_page_config(
    page_title="Western Ghats CTI Dashboard",
    layout="wide", # Use wide layout for better map display
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://docs.streamlit.io/library/api-reference/utilities/st.set_page_config',
        'Report a bug': "https://github.com/streamlit/streamlit/issues", # Replace with your bug report link if any
        'About': "# Western Ghats CTI Dashboard\nThis app visualizes Carbon Threat Index data with an interactive CTI map."
    }
)

# --- Define Local Raster File Path ---
# IMPORTANT: This is a hardcoded path. The Streamlit app will only find this file
# if it's run on a machine where this exact path is valid.
# Using raw string for Windows path compatibility
RASTER_FILE_PATH = r"D:\wg_cti_project\streamlit_cti_app_fresh\streamlit-cti-dashboard-v2\CTI MASked\cti_masked.tif"

# --- Data Loading and Preparation (CSV Data) ---
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
def load_csv_data():
    """
    Loads and preprocesses the CTI CSV data.
    Returns:
        pandas.DataFrame: Processed DataFrame with District and CTI_Value.
    """
    file_content_io = io.StringIO(csv_data)
    df = pd.read_csv(file_content_io)
    df.columns = ['District', 'CTI_Value'] # Rename columns for easier access
    df['CTI_Value'] = pd.to_numeric(df['CTI_Value']) # Ensure CTI_Value is numeric
    return df

# Load the CSV data
df_data = load_csv_data()

# --- Sidebar ---
st.sidebar.header("Dashboard Options")
st.sidebar.markdown("""
Explore district-wise Carbon Threat Index (CTI) data and the interactive CTI map.
Adjust the raster map's opacity using the slider below.
""")
st.sidebar.info(f"Raster map: {os.path.basename(RASTER_FILE_PATH)}")

# Add a slider for raster opacity in the sidebar
raster_opacity = st.sidebar.slider("Raster Map Opacity", min_value=0.0, max_value=1.0, value=0.7, step=0.05)


# --- Main Page ---
st.title("🌍 Western Ghats Carbon Threat Index (CTI) Dashboard")
st.markdown("Interactive CTI map and district-wise statistics for the Western Ghats region.")

# --- Display CTI Raster Map with Folium ---
st.header("🗺️ Interactive CTI Raster Map")

if os.path.exists(RASTER_FILE_PATH):
    try:
        with rasterio.open(RASTER_FILE_PATH) as src:
            # Read the first band of the raster
            raster_array = src.read(1).astype(np.float32) # Ensure float for calculations
            
            # Handle NoData values by masking them (important for correct min/max and colormapping)
            nodata_value = src.nodata
            if nodata_value is not None:
                raster_array = np.ma.masked_equal(raster_array, nodata_value)
                # Replace masked values with NaN for colormap processing if not already
                raster_array = np.ma.filled(raster_array, np.nan) 
            
            # Get raster bounds in its original CRS
            bounds_orig = src.bounds
            
            # Transform bounds to Latitude/Longitude (EPSG:4326) for Folium
            # Folium's ImageOverlay expects bounds as [[lat_min, lon_min], [lat_max, lon_max]]
            bounds_latlon = transform_bounds(src.crs, {'init': 'epsg:4326'}, 
                                             bounds_orig.left, bounds_orig.bottom, 
                                             bounds_orig.right, bounds_orig.top)
            
            folium_bounds = [[bounds_latlon[1], bounds_latlon[0]], [bounds_latlon[3], bounds_latlon[2]]] # Reorder for Folium
            
            # Calculate center of the map for initial view
            map_center_lat = (bounds_latlon[1] + bounds_latlon[3]) / 2
            map_center_lon = (bounds_latlon[0] + bounds_latlon[2]) / 2

            # Create a Folium map centered on the raster
            # Adjust zoom_start as needed for appropriate initial zoom level
            m = folium.Map(location=[map_center_lat, map_center_lon], zoom_start=7, tiles="OpenStreetMap")

            # --- Prepare raster for overlay ---
            # Determine min and max values from the raster array, ignoring NaNs
            min_val = np.nanmin(raster_array)
            max_val = np.nanmax(raster_array)
            
            # Check if min_val and max_val are valid for colormap
            if np.isnan(min_val) or np.isnan(max_val) or min_val == max_val:
                st.warning("Raster data has a single value or contains all NaNs. Cannot generate a meaningful colormap or legend. Displaying raster as is.")
                # If data is flat or all NaN, ImageOverlay might not work as expected with a colormap.
                # For simplicity, we'll try to overlay it. It might appear as a solid block or not at all.
                # A more robust solution would involve specific handling for such cases.
                colored_raster_for_overlay = raster_array # Pass raw if no colormap
            else:
                # Get a matplotlib colormap
                # Options: 'viridis', 'plasma', 'inferno', 'magma', 'cividis' (sequential)
                # 'coolwarm', 'RdYlGn', 'PiYG' (diverging for data around a midpoint)
                # 'Greys', 'Blues', 'Oranges' (sequential single-hue)
                cmap_name = 'viridis' # Choose a colormap
                mpl_cmap = plt.get_cmap(cmap_name)

                # Normalize raster data for colormap (0-1 range), handling NaNs
                # Add a small epsilon to prevent division by zero if min_val is very close to max_val
                norm_array = (raster_array - min_val) / (max_val - min_val + 1e-9) 
                norm_array[np.isnan(raster_array)] = np.nan # Preserve NaNs

                # Apply colormap to the raster data to get RGBA image
                colored_raster_for_overlay = mpl_cmap(norm_array)
                # Set alpha for NaN values to 0 (transparent)
                colored_raster_for_overlay[np.isnan(norm_array), 3] = 0 


            # Add the CTI raster as an ImageOverlay
            img_overlay = folium.raster_layers.ImageOverlay(
                image=colored_raster_for_overlay, # Pass the RGBA array
                bounds=folium_bounds,
                opacity=raster_opacity, # Controlled by the sidebar slider
                interactive=True, # Allows for map interactions over the overlay
                cross_origin=False, # Typically False for local data
                name="CTI Raster Layer" # Name for Layer Control
            )
            img_overlay.add_to(m)

            # --- Add Colormap/Legend ---
            if not (np.isnan(min_val) or np.isnan(max_val) or min_val == max_val):
                # Create a Branca colormap object for the legend
                # Ensure colors are derived correctly from the chosen matplotlib colormap
                # Branca colormaps usually take a list of hex colors or (r,g,b,a) tuples
                
                # Get a list of colors from the matplotlib colormap
                # For a continuous colormap, sample it at several points
                legend_colors = [mpl_cmap(i) for i in np.linspace(0, 1, num=mpl_cmap.N if hasattr(mpl_cmap, 'N') else 256)]

                linear_cmap_legend = cm.LinearColormap(
                    colors=legend_colors, # Pass the list of RGBA colors
                    vmin=min_val,
                    vmax=max_val,
                    caption=f"CTI Value ({cmap_name} colormap)" # Legend title
                )
                linear_cmap_legend.add_to(m) # Add legend to the map
            else:
                st.info("Skipping legend generation due to invalid raster data range (NaN or flat).")


            # Add Layer Control to toggle layers (useful if you add more overlays in the future)
            folium.LayerControl().add_to(m)

            # Display the map in Streamlit
            # Adjust width and height as needed for your layout
            folium_static(m, width=None, height=600) 

            st.caption(f"Displaying raster: {os.path.basename(RASTER_FILE_PATH)}. Opacity: {raster_opacity:.2f}. "
                       f"Original CRS: {src.crs}. Min CTI: {min_val:.2f}, Max CTI: {max_val:.2f}")

    except ImportError as ie:
        st.error(f"ImportError: {ie}. One of the required libraries (rasterio, folium, streamlit-folium, branca) might not be installed correctly. Please check your 'requirements.txt' and ensure all libraries are installed in your Python environment.")
    except Exception as e:
        st.error(f"An error occurred while displaying the raster map: {e}")
        st.error("Details: Please ensure the raster file is a valid GeoTIFF, the file path is correct, and all necessary dependencies (including GDAL for rasterio.warp if reprojection is needed) are properly installed.")
else:
    st.warning(f"Raster file not found at the specified path: {RASTER_FILE_PATH}")
    st.info("Please ensure the CTI raster map file exists at the correct location for it to be displayed.")


# --- Data Analysis and Calculations (from CSV) --- (This section remains the same as before)
# Calculate basic descriptive statistics for CTI values from CSV
min_cti_val_csv = df_data['CTI_Value'].min() # Renamed to avoid conflict with raster min/max
max_cti_val_csv = df_data['CTI_Value'].max()
mean_cti_val = df_data['CTI_Value'].mean()
median_cti_val = df_data['CTI_Value'].median()
std_cti_val = df_data['CTI_Value'].std()
num_districts = len(df_data)

# Sort the DataFrame by CTI_Value in descending order for rankings and default displays
df_sorted = df_data.sort_values(by='CTI_Value', ascending=False).reset_index(drop=True)

# Create a display version of the sorted DataFrame with index starting from 1
df_sorted_display = df_sorted.copy()
df_sorted_display.index = np.arange(1, len(df_sorted_display) + 1)


# Define Threat Categories based on CTI quartiles from CSV
q1 = df_data['CTI_Value'].quantile(0.25)
q2 = df_data['CTI_Value'].quantile(0.50)
q3 = df_data['CTI_Value'].quantile(0.75)

def categorize_cti(cti_value):
    """
    Categorizes a CTI value into a threat level based on quartiles.
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
col3.metric("Minimum CTI Value (District)", f"{min_cti_val_csv:.4f}", help=f"District: {min_district}")
col3.metric("Maximum CTI Value (District)", f"{max_cti_val_csv:.4f}", help=f"District: {max_district}")


# --- Display Top & Bottom Districts (from CSV) ---
st.header("🏆 Top & Bottom Districts by CTI (District Data)")
# Use a unique key for the slider to avoid conflicts if other sliders exist
num_top_bottom = st.slider(
    "Select number of Top/Bottom districts to display:",
    min_value=1,
    max_value=10,
    value=5,
    key="top_bottom_slider_csv_data"
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
    index=0, # Default to descending
    key="chart_sort_select_csv_data"
)
df_chart_sorted = df_sorted_display.sort_values(
    by='CTI_Value',
    ascending=(sort_order_chart == "Ascending (Low to High)")
)

fig_bar, ax_bar = plt.subplots(figsize=(10, 14)) # Adjust figure size for readability
bars_obj = ax_bar.barh(df_chart_sorted['District'], df_chart_sorted['CTI_Value'], color='darkcyan')
ax_bar.set_xlabel('Mean Carbon Threat Index (CTI) Value')
ax_bar.set_ylabel('District')
ax_bar.set_title(f'Mean CTI by District in Western Ghats ({num_districts} Districts)')

if sort_order_chart == "Descending (High to Low)":
    ax_bar.invert_yaxis() # Show highest CTI at the top
ax_bar.grid(axis='x', linestyle='--', alpha=0.7) # Add a light grid

# Add CTI values as text labels on the bars
for bar_item in bars_obj:
    ax_bar.text(
        bar_item.get_width() + 0.0005, # x-position (slightly offset from bar end)
        bar_item.get_y() + bar_item.get_height() / 2, # y-position (center of bar)
        f'{bar_item.get_width():.4f}', # Text label (formatted CTI value)
        va='center', ha='left', fontsize=7
    )
ax_bar.set_xlim(0, max_cti_val_csv * 1.18) # Adjust x-axis limit to make space for labels
plt.tight_layout() # Adjust plot to ensure everything fits
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
    index=0, # Default to "All"
    key="category_filter_select_csv_data"
)

df_display_filtered = df_sorted_display if selected_category == "All" else df_sorted_display[df_sorted_display['Threat_Category'] == selected_category]

sort_column = st.selectbox(
    "Sort data by:",
    ["CTI_Value", "District", "Threat_Category"],
    index=0,
    key="table_sort_column_select_csv_data"
)
default_sort_ascending = False if sort_column == "CTI_Value" else True
sort_ascending = st.checkbox(
    "Sort Ascending",
    value=default_sort_ascending,
    key="table_sort_order_checkbox_csv_data"
)

df_display_final_sorted = df_display_filtered.sort_values(by=sort_column, ascending=sort_ascending)
# Reset index for continuous numbering after filtering and sorting for display
df_display_final_sorted.index = np.arange(1, len(df_display_final_sorted) + 1)

st.dataframe(
    df_display_final_sorted[['District', 'CTI_Value', 'Threat_Category']].style.format({'CTI_Value': "{:.4f}"}),
    height=600, # Set a fixed height for the scrollable table
    use_container_width=True # Make the table use the full container width
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
st.markdown(f"CSV Data (District-wise) is static in this script. Last script update: {pd.Timestamp.now(tz='Asia/Kolkata').strftime('%Y-%m-%d %H:%M:%S %Z')}")
st.markdown(f"Raster Map Path: {RASTER_FILE_PATH}")

