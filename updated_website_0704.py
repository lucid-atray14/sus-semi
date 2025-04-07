import pandas as pd
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, LabelSet
import streamlit as st
import numpy as np
from pymcdm.methods import PROMETHEE_II
from pymcdm.helpers import rankdata
from pymcdm.methods import TOPSIS
from pymcdm.weights import entropy_weights
from io import BytesIO
from bokeh.palettes import Category10
from bokeh.models import NumeralTickFormatter
from bokeh.models import LogScale, Range1d, LinearScale

# Custom CSS for styling
def set_custom_style():
    st.markdown("""
    <style>
        /* Lighten sidebar background */
        [data-testid="stSidebar"] {
            background-color: #f8f9fa !important;
            border-right: 1px solid #e0e0e0;
        }
        
        /* Dark text for light sidebar */
        [data-testid="stSidebar"] .stRadio > label,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] .stMarkdown {
            color: #333333 !important;
        }
        
        /* Sidebar hover effects */
        [data-testid="stSidebar"] .stRadio > div:hover {
            background-color: #e9ecef;
            border-radius: 5px;
        }
    </style>
    """, unsafe_allow_html=True)

# ===== CACHED FUNCTIONS =====
@st.cache_data
def load_final_database():
    df_original = pd.read_excel("final_database.xlsx")
    return df_original.iloc[:, 1:]

@st.cache_data
def load_bandgap_database():
    df1_original = pd.read_excel("bandgap_database.xlsx")
    return df1_original.iloc[:, 1:]

@st.cache_data
def filter_dataframe(_df, filters, selected_names=None):
    """Filter dataframe based on provided filters and optional names"""
    bandgap_range = filters.get("Bandgap", (_df["Bandgap"].min(), _df["Bandgap"].max()))
    esg_range = filters.get("ESG Score", (_df["ESG Score"].min(), _df["ESG Score"].max()))
    toxicity_range = filters.get("Toxicity", (_df["Toxicity"].min(), _df["Toxicity"].max()))
    co2_range = filters.get("CO2 footprint max (kg/kg)", 
                          (_df["CO2 footprint max (kg/kg)"].min(), 
                           _df["CO2 footprint max (kg/kg)"].max()))
    
    filtered = _df[
        _df["Bandgap"].between(bandgap_range[0], bandgap_range[1], inclusive='both') &
        _df["ESG Score"].between(esg_range[0], esg_range[1], inclusive='both') &
        _df["Toxicity"].between(toxicity_range[0], toxicity_range[1], inclusive='both') &
        _df["CO2 footprint max (kg/kg)"].between(co2_range[0], co2_range[1], inclusive='both')
    ]
    
    if selected_names is not None:
        filtered = filtered[filtered["Name"].isin(selected_names)]
    return filtered

@st.cache_data
def calculate_weights(matrix, method="entropy"):
    if method == "entropy":
        return entropy_weights(matrix)
    return None

@st.cache_data
def run_topsis(matrix, weights, criteria_types):
    topsis = TOPSIS()
    return topsis(matrix, weights, criteria_types)

@st.cache_data
def run_promethee(matrix, weights, criteria_types):
    promethee = PROMETHEE_II('usual')
    return promethee(matrix, weights, criteria_types)

@st.cache_data
def prepare_plot_data(df, x_col, y_col, log_x=False, log_y=False):
    df_plot = df.copy()
    if log_x:
        df_plot[x_col] = np.log10(df_plot[x_col].clip(lower=1e-10))
    if log_y:
        df_plot[y_col] = np.log10(df_plot[y_col].clip(lower=1e-10))
    return df_plot

@st.cache_data
def create_full_output(filtered_df, results_df, weights_df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        full_data = filtered_df.copy()
        if 'Score' in results_df.columns:
            full_data['TOPSIS_Score'] = results_df['Score']
            full_data['TOPSIS_Rank'] = results_df['Rank']
        else:
            full_data['PROMETHEE_Net_Flow'] = results_df['Net Flow']
            full_data['PROMETHEE_Rank'] = results_df['Rank']
        full_data.to_excel(writer, sheet_name='Full Data', index=False)
        results_df.to_excel(writer, sheet_name='Rankings', index=False)
        weights_df.to_excel(writer, sheet_name='Weights', index=False)
        pd.DataFrame.from_dict(st.session_state.filters, orient='index').to_excel(
            writer, sheet_name='Filter Settings'
        )
    return output.getvalue()

def create_professional_plot(df, x_col, y_col, title, x_label, y_label, log_x=False, log_y=False):
    # Create a copy to avoid modifying the original dataframe
    df_plot = df.copy()
    
    # Professional color palette
    primary_color = "#3498db"
    highlight_color = "#e74c3c"
    
    # Create the figure with dynamic axis types
    p = figure(
        title=title,
        tools="pan,wheel_zoom,box_zoom,reset,save,hover",
        x_axis_label=f"log({x_label})" if log_x else x_label,
        y_axis_label=f"log({y_label})" if log_y else y_label,
        x_axis_type="log" if log_x else "linear",
        y_axis_type="log" if log_y else "linear",
        width=800,
        height=500,
        tooltips=[("Name", "@Name"), 
                 (f"log({x_col})" if log_x else x_col, f"@{x_col}"),
                 (f"log({y_col})" if log_y else y_col, f"@{y_col}")],
        toolbar_location="above",
        sizing_mode="stretch_width"
    )
    
    # Handle negative/zero values for log scales
    if log_x:
        df_plot[x_col] = df_plot[x_col].clip(lower=1e-10)
    if log_y:
        df_plot[y_col] = df_plot[y_col].clip(lower=1e-10)
    
    # Plot all points
    source = ColumnDataSource(df_plot)
    p.circle(
        x=x_col,
        y=y_col,
        source=source,
        size=8,
        color=primary_color,
        alpha=0.6,
        legend_label="All Materials"
    )
    
    # Highlight exactly 10 random materials
    num_highlight = min(10, len(df_plot))
    highlight_df = df_plot.sample(n=num_highlight, random_state=42)
    highlight_source = ColumnDataSource(highlight_df)
    
    p.circle(
        x=x_col,
        y=y_col,
        source=highlight_source,
        size=12,
        color=highlight_color,
        alpha=1.0,
        legend_label="Highlighted Materials"
    )
    
    # Add labels to highlighted points
    labels = LabelSet(
        x=x_col,
        y=y_col,
        text="Name",
        source=highlight_source,
        text_font_size="10pt",
        text_color=highlight_color,
        y_offset=8,
        text_align='center'
    )
    p.add_layout(labels)
    
    # Professional legend styling
    p.legend.location = "top_right"
    p.legend.click_policy = "hide"
    p.legend.background_fill_alpha = 0.7
    p.legend.label_text_font_size = "12pt"
    
    # Grid and axis styling
    p.xgrid.grid_line_color = "#e0e0e0"
    p.ygrid.grid_line_color = "#e0e0e0"
    p.axis.minor_tick_line_color = None
    
    return p

def main():
    set_custom_style()
    df = load_final_database()
    df1 = load_bandgap_database()
    
    # Sidebar navigation
    st.sidebar.title("📊 Material Analysis")
    st.sidebar.markdown("---")
    selected_page = st.sidebar.radio(
        "Navigation Menu", 
        ["Home", "Bandgap Analysis", "Custom Analysis", "MCDM Analysis"],
        captions=["Welcome page", "Bandgap properties", "Custom relationships", "Decision making"]
    )
    
    # Add footer
    st.markdown("""
    <div class="footer">
        Material Analysis Platform © 2023 | v1.0 | Developed by Your Team
    </div>
    """, unsafe_allow_html=True)

    if selected_page == "Home":
        st.title("Material Properties Analysis Platform")
        
        cols = st.columns(2)
        with cols[0]:
            st.markdown("""
            ### 🔍 About This Tool
            This interactive platform enables comprehensive analysis of material properties with:
            - **Bandgap-specific** visualizations
            - **Custom relationship** exploration
            - **Multi-criteria** decision making
            - **Export capabilities** for further analysis
            """)
            
        with cols[1]:
            st.markdown("""
            ### 🚀 Getting Started
            1. Select an analysis page from the sidebar
            2. Configure your filters and parameters
            3. Visualize the relationships
            4. Download results for reporting
            
            **Pro Tip:** Use the MCDM analysis for comprehensive material evaluation.
            """)
        
        st.markdown("---")
        
        with st.expander("📚 Database Information", expanded=True):
            cols = st.columns(3)
            with cols[0]:
                st.metric("Total Materials", len(df))
            with cols[1]:
                st.metric("Bandgap Range", f"{df['Bandgap'].min():.1f} - {df['Bandgap'].max():.1f} eV")
            with cols[2]:
                st.metric("Production Range", f"{df['Production (ton)'].min():.1f} - {df['Production (ton)'].max():.1f} tons")
        
    elif selected_page == "Bandgap Analysis":
        st.title("📈 Bandgap Analysis")
        st.markdown("Analyze material properties with respect to bandgap values")

        # Filters section at the top in expandable containers
        with st.expander("🔍 Filter Settings", expanded=True):
            cols = st.columns(3)
            
            with cols[0]:
                st.markdown("**Material Properties**")
                y_col = st.selectbox(
                    "Y-Axis Property", 
                    [col for col in df1.columns if col not in ['Name', 'Bandgap']],
                    help="Select the property to plot against bandgap"
                )
                
            with cols[1]:
                st.markdown("**Range Filters**")
                esg_range = st.slider(
                    "ESG Score Range", 
                    0.0, 4.0, (0.0, 3.5), 0.1,
                    help="Filter materials by ESG score range"
                )
                toxicity_range = st.slider(
                    "Toxicity Level", 
                    0.0, 4.0, (0.0, 3.0), 1.0,
                    help="Filter materials by toxicity level"
                )
                
            with cols[2]:
                st.markdown("**Material Selection**")
                co2_range = st.slider(
                    "CO₂ Footprint (kg/kg)", 
                    0.0, 15000.0, (0.0, 5000.0),
                    help="Filter materials by CO₂ footprint"
                )
            
            specified_names = [
                "Ti1.0O2.0","Zn1.0O1.0","Mo1.0S2.0","C3.0N4.0","C1.0N1.0","Si1.0",
                "Ce1.0O2.0","Al2.0O3.0","Si1.0O2.0","Cu1.0O1.0"
            ]
            selected_names = st.multiselect(
                "Select specific materials",
                specified_names,
                default=["Ti1.0O2.0", "C3.0N4.0"],
                help="Focus on specific materials of interest"
            )

        # Apply filters
        filters = {
            "ESG Score": esg_range,
            "Toxicity": toxicity_range,
            "CO2 footprint max (kg/kg)": co2_range
        }
        
        # Process data
        name_colors = {name: Category10[len(specified_names)][i] for i, name in enumerate(specified_names)} 
        df1['color'] = df1['Name'].map(name_colors)
        filtered_df = filter_dataframe(df1, filters, selected_names if selected_names else None)
        
        # Plot section below filters
        st.markdown("---")
        st.markdown(f"**Analysis Results ({len(filtered_df)} materials)**")
        
        if len(filtered_df) == 0:
            st.warning("⚠️ No materials match your filters. Please adjust your criteria.")
        else:
            # Create the plot with maximum width
            p = figure(
                title=f"Bandgap vs {y_col}",
                tools="pan,wheel_zoom,box_zoom,reset,save",
                x_axis_label="Bandgap (eV)",
                y_axis_label=y_col,
                width=1000,
                height=600,
                sizing_mode="stretch_width"
            )
            
            # Plot data
            source = ColumnDataSource(filtered_df)
            p.circle(
                x="Bandgap", 
                y=y_col, 
                source=source, 
                size=12,
                color='color', 
                alpha=0.7,
                legend_field="Name"
            )
            
            # Calculate y-axis range
            y_min = filtered_df[y_col].min() * 0.9
            y_max = filtered_df[y_col].max() * 1.1
            
            # Add bandgap regions
            regions = [
                (0, 1.6, "#f1c40f", "Infrared (0-1.6 eV)"),
                (1.6, 3.26, "#3498db", "Visible (1.6-3.26 eV)"),
                (3.26, 20, "#2ecc71", "UV (>3.26 eV)")
            ]
            
            for left, right, color, label in regions:
                p.quad(
                    top=y_max,
                    bottom=y_min,
                    left=left,
                    right=right,
                    color=color,
                    alpha=0.1,
                    legend_label=label
                )
            
            # Add hover and legend
            hover = HoverTool(tooltips=[
                ("Name", "@Name"),
                ("Bandgap", "@Bandgap{0.00}"),
                (y_col, f"@{y_col}{{0.00}}")
            ])
            p.add_tools(hover)
            p.legend.location = "top_right"
            p.legend.click_policy = "mute"
            p.legend.label_text_font_size = "12pt"
            
            # Display plot
            st.bokeh_chart(p, use_container_width=True)
            
            # Data download
            st.download_button(
                label="📥 Download Analysis Data",
                data=filtered_df.to_csv(index=False).encode('utf-8'),
                file_name="bandgap_analysis.csv",
                mime="text/csv",
                use_container_width=True
            )
            
        # Bandgap reference (always show)
        with st.expander("ℹ️ Bandgap Reference", expanded=False):
            st.markdown("""
            **Bandgap Energy Ranges:**
            - **Infrared**: <1.6 eV (Yellow)
            - **Visible**: 1.6-3.26 eV (Blue)
            - **Ultraviolet**: >3.26 eV (Green)
            
            *The shaded regions represent these energy ranges.*
            """)

    elif selected_page == "Custom Analysis":
        st.title("🔍 Custom Analysis")
        st.markdown("Explore relationships between any material properties")
        
        # Initialize session state
        if 'filters' not in st.session_state:
            st.session_state.filters = {}
        
        with st.expander("🔧 Filter Settings", expanded=True):
            cols = st.columns(2)
            
            with cols[0]:
                st.subheader("Bandgap Selection")
                bandgap_range = st.slider(
                    "Bandgap Range (eV)", 
                    0.0, 20.0, (0.0, 3.0), 0.1,
                    key="bandgap_slider"
                )
                
            with cols[1]:
                st.subheader("Additional Filters")
                esg_range = st.slider("ESG Score", 0.0, 5.0, (0.0, 3.5), 0.1)
                toxicity_range = st.slider("Toxicity", 0.0, 4.0, (0.0, 3.0), 1.0)
                
                production_range = st.slider(
                    "Production (tons)",
                    float(df['Production (ton)'].min()),
                    float(df['Production (ton)'].max()),
                    (float(df['Production (ton)'].min()), float(df['Production (ton)'].max())),
                    step=1.0
                )
            
            if st.button("Apply Filters", key="apply_filters"):
                st.session_state.filters.update({
                    "Bandgap": bandgap_range,
                    "ESG Score": esg_range,
                    "Toxicity": toxicity_range,
                    "Production (ton)": production_range
                })
                st.success("Filters applied!")
        
        if st.session_state.filters:  # Check if any filters exist
            # Plot configuration
            st.subheader("📊 Plot Configuration")
            
            cols = st.columns(2)
            with cols[0]:
                x_col = st.selectbox("X-Axis", [col for col in df.columns if col != 'Name'])
                log_x = st.checkbox(f"Log scale X-axis")
            with cols[1]:
                y_col = st.selectbox("Y-Axis", [col for col in df.columns if col not in ['Name', x_col]])
                log_y = st.checkbox(f"Log scale Y-axis")
            
            # Apply filters and create plot
            filtered_df = filter_dataframe(df, st.session_state.filters)
            
            if not filtered_df.empty:
                st.success(f"🔄 {len(filtered_df)} materials match current filters")
                
                # Advanced options
                with st.expander("🎨 Customization Options"):
                    plot_title = st.text_input("Plot Title", f"{x_col} vs {y_col}")
                
                # Create professional plot with random highlighting
                p = create_professional_plot(
                    filtered_df, x_col, y_col, plot_title, x_col, y_col, log_x, log_y
                )
                
                st.bokeh_chart(p, use_container_width=True)
                
                # Data table
                with st.expander("📋 View Data"):
                    st.dataframe(filtered_df[[x_col, y_col, "Name"]].sort_values(y_col, ascending=False))
                
                # Download
                st.download_button(
                    label="📥 Download Analysis Data",
                    data=filtered_df.to_csv(index=False).encode('utf-8'),
                    file_name="custom_analysis.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No materials match the current filters. Please adjust your criteria.")

    elif selected_page == "MCDM Analysis":
        st.title("📊 Multi-Criteria Decision Making")
        st.markdown("Evaluate materials using TOPSIS or PROMETHEE methods")
        
        # Initialize session state
        if 'filters' not in st.session_state:
            st.session_state.filters = {}
            st.session_state.bandgap_selected = False
        
        with st.expander("🔧 Filter Settings", expanded=True):
            cols = st.columns(2)
            
            with cols[0]:
                st.subheader("Bandgap Selection")
                fixed_width = st.checkbox("Fixed Width (±0.1 eV)", True, key="mcdm_bandgap")
                
                if fixed_width:
                    center = st.number_input(
                        "Center Value (eV)", 
                        min_value=0.0, max_value=20.0, value=2.0, step=0.1, 
                        key="mcdm_center"
                    )
                    bandgap_range = (round(center - 0.1, 1), round(center + 0.1, 1))
                else:
                    bandgap_range = st.slider("Custom Range (eV)", 0.0, 20.0, (0.0, 3.0), 0.1, key="mcdm_range")
                
                if st.button("Apply Bandgap Filter", key="mcdm_bandgap_filter"):
                    st.session_state.filters["Bandgap"] = bandgap_range
                    st.session_state.bandgap_selected = True
                    st.success("Bandgap filter applied!")
            
            with cols[1]:
                if st.session_state.get('bandgap_selected', False):
                    st.subheader("Additional Filters")
                    esg_range = st.slider("ESG Score", 0.0, 5.0, (0.0, 3.5), 0.1, key="mcdm_esg")
                    toxicity_range = st.slider("Toxicity", 0.0, 4.0, (0.0, 3.0), 1.0, key="mcdm_toxicity")
                    
                    production_range = st.slider(
                        "Production (tons)",
                        float(df['Production (ton)'].min()),
                        float(df['Production (ton)'].max()),
                        (float(df['Production (ton)'].min()), float(df['Production (ton)'].max())),
                        step=1.0,
                        key="mcdm_production"
                    )
                    
                    st.session_state.filters.update({
                        "ESG Score": esg_range,
                        "Toxicity": toxicity_range,
                        "Production (ton)": production_range
                    })
        
        if st.session_state.get('bandgap_selected', False):
            # MCDM configuration
            st.subheader("⚖️ Analysis Configuration")
            
            cols = st.columns(2)
            with cols[0]:
                mcdm_method = st.selectbox(
                    "Method",
                    ["TOPSIS", "PROMETHEE"],
                    help="TOPSIS: Technique for Order Preference by Similarity to Ideal Solution\nPROMETHEE: Preference Ranking Organization Method for Enrichment Evaluation"
                )
            with cols[1]:
                if mcdm_method == "TOPSIS":
                    weighting_method = st.radio(
                        "Weighting",
                        ["Entropy Weighting", "Manual Weights"],
                        horizontal=True
                    )
            
            # Get filtered data
            filtered_df = filter_dataframe(df, st.session_state.filters)
            
            if not filtered_df.empty:
                st.success(f"🔄 {len(filtered_df)} materials available for analysis")
                
                # Criteria selection
                criteria_options = {
                    'Reserve (ton)': 1, 'Production (ton)': 1, 'HHI (USGS)': -1,
                    'ESG Score': -1, 'CO2 footprint max (kg/kg)': -1,
                    'Embodied energy max (MJ/kg)': -1, 'Water usage max (l/kg)': -1,
                    'Toxicity': -1, 'Companionality': -1
                }
                available_criteria = {k: v for k, v in criteria_options.items() if k in filtered_df.columns}
                
                # Weight assignment
                if mcdm_method == "TOPSIS" and weighting_method == "Entropy Weighting":
                    weights = entropy_weights(filtered_df[list(available_criteria.keys())].values)
                else:
                    st.subheader("📊 Criteria Weights")
                    st.markdown("Assign importance to each criterion (0-5 scale):")
                    
                    weights = []
                    cols = st.columns(len(available_criteria))
                    for i, (col, direction) in enumerate(available_criteria.items()):
                        with cols[i]:
                            weight = st.slider(
                                f"{col} ({'Max' if direction == 1 else 'Min'})",
                                0, 5, 3,
                                key=f"weight_{col}"
                            )
                            weights.append(weight)
                    
                    # Normalize weights
                    if sum(weights) == 0:
                        st.warning("All weights set to 0 - using equal weights instead")
                        weights = np.ones(len(weights)) / len(weights)
                    else:
                        weights = np.array(weights) / sum(weights)
                
                # Display weights
                weights_df = pd.DataFrame({
                    'Criterion': list(available_criteria.keys()),
                    'Weight': weights,
                    'Direction': ['Maximize' if d == 1 else 'Minimize' for d in available_criteria.values()]
                }).sort_values('Weight', ascending=False)
                
                st.dataframe(
                    weights_df.style.format({'Weight': '{:.2%}'}),
                    use_container_width=True
                )
                
                # Run analysis
                if st.button("🚀 Run Analysis", type="primary"):
                    with st.spinner("Performing analysis..."):
                        matrix = filtered_df[list(available_criteria.keys())].values
                        types = np.array([available_criteria[k] for k in available_criteria])
                        
                        if mcdm_method == "TOPSIS":
                            scores = run_topsis(matrix, weights, types)
                            ranks = rankdata(scores, reverse=True)
                            results = pd.DataFrame({
                                'Material': filtered_df['Name'],
                                'Score': scores,
                                'Rank': ranks
                            }).sort_values('Rank')
                        else:
                            flows = run_promethee(matrix, weights, types)
                            ranks = rankdata(flows, reverse=True)
                            results = pd.DataFrame({
                                'Material': filtered_df['Name'],
                                'Net Flow': flows,
                                'Rank': ranks
                            }).sort_values('Rank')
                        
                        # Display results
                        st.subheader("📋 Results")
                        st.dataframe(
                            results.style.format({'Score': '{:.4f}', 'Net Flow': '{:.4f}'}),
                            use_container_width=True
                        )
                        
                        # Visualize top materials
                        st.subheader("🏆 Top Materials")
                        top_n = min(3, len(results))
                        top_materials = results.head(top_n)['Material'].tolist()
                        
                        cols = st.columns(top_n)
                        for i, material in enumerate(top_materials):
                            with cols[i]:
                                st.metric(
                                    label=f"Rank #{i+1}",
                                    value=material,
                                    help=f"Score: {results.iloc[i]['Score'] if 'Score' in results.columns else results.iloc[i]['Net Flow']:.4f}"
                                )
                        
                        # Download results
                        excel_data = create_full_output(filtered_df, results, weights_df)
                        st.download_button(
                            label="📥 Download Full Report",
                            data=excel_data,
                            file_name=f"material_analysis_{mcdm_method}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
            else:
                st.warning("No materials match the current filters. Please adjust your criteria.")

if __name__ == "__main__":
    main()
