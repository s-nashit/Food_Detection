import streamlit as st
import os
import datetime
import pandas as pd
from collections import defaultdict
import calendar
import glob
from yolo_detector import count_unique_objects
import gdown
import os

model_path = "model.pt"


# Download the model if it doesn't exist
if not os.path.exists(model_path):
    with st.spinner("Loading model..."):
        gdown.download(model_path, quiet=False)

RAW_MATERIALS_USAGE = {
     'biryani': {'rice': 0.15, 'meat': 0.1, 'onions': 0.05, 'spices': 0.02, 'oil': 0.02},
    'chapathi': {'flour': 0.05, 'oil': 0.005, 'salt': 0.001},
    'chole_bature': {'chickpeas': 0.1, 'flour': 0.08, 'oil': 0.03, 'spices': 0.01},
    'dahl': {'lentils': 0.08, 'onions': 0.02, 'spices': 0.01, 'oil': 0.01},
    'dosa': {'rice': 0.06, 'lentils': 0.02, 'oil': 0.01},
    'gulab_jamun': {'milk_powder': 0.05, 'flour': 0.02, 'sugar': 0.03, 'oil': 0.02},
    'idly': {'rice': 0.04, 'lentils': 0.01, 'salt': 0.001},
    'jalebi': {'flour': 0.03, 'sugar': 0.04, 'oil': 0.02, 'saffron': 0.0001},
    'kadai_paneer': {'paneer': 0.1, 'onions': 0.03, 'tomatoes': 0.03, 'spices': 0.01, 'oil': 0.02},
    'naan': {'flour': 0.06, 'yogurt': 0.02, 'oil': 0.01, 'salt': 0.001},
    'pakoda': {'flour': 0.04, 'vegetables': 0.03, 'oil': 0.05, 'spices': 0.005},
    'pancakes': {'flour': 0.05, 'milk': 0.06, 'eggs': 0.02, 'sugar': 0.01, 'butter': 0.01},
    'pani_puri': {'flour': 0.02, 'potato': 0.03, 'tamarind': 0.01, 'spices': 0.005},
    'pav_bhaji': {'vegetables': 0.1, 'bread': 0.05, 'butter': 0.02, 'spices': 0.01},
    'rolls': {'flour': 0.04, 'vegetables': 0.05, 'oil': 0.01, 'spices': 0.005},
    'samosa': {'flour': 0.03, 'potato': 0.04, 'peas': 0.01, 'oil': 0.03, 'spices': 0.005},
    'vada_pav': {'potato': 0.08, 'flour': 0.03, 'bread': 0.05, 'oil': 0.04, 'spices': 0.01},
    'hamburger': {'bun': 1, 'patty': 0.15, 'lettuce': 0.02, 'cheese': 0.02, 'tomato': 0.03},
    'ice_cream': {'milk': 0.1, 'sugar': 0.02, 'cream': 0.05},
    'pizza': {'flour': 0.15, 'cheese': 0.08, 'tomato_sauce': 0.04, 'toppings': 0.08},
    # Newly added items
    'badammilk': {'milk': 0.2, 'almonds': 0.03, 'sugar': 0.02, 'cardamom': 0.001},
    'cholekulcha': {'chickpeas': 0.1, 'flour': 0.08, 'spices': 0.01, 'oil': 0.02},
    'coldcoffee': {'milk': 0.2, 'coffee': 0.01, 'sugar': 0.02, 'ice': 0.05},
    'lassi': {'yogurt': 0.25, 'sugar': 0.02, 'cardamom': 0.001},
    'makhnakheer': {'milk': 0.25, 'makhana': 0.05, 'sugar': 0.03, 'cardamom': 0.001, 'nuts': 0.01},
    'matarkachori': {'flour': 0.05, 'peas': 0.04, 'spices': 0.01, 'oil': 0.03},
    'momos': {'flour': 0.06, 'vegetables': 0.05, 'oil': 0.01, 'spices': 0.005},
    'pasta': {'pasta': 0.12, 'sauce': 0.05, 'vegetables': 0.04, 'cheese': 0.02},
    'poha': {'poha': 0.08, 'onions': 0.02, 'potato': 0.03, 'spices': 0.005, 'oil': 0.01},
    'sandwich': {'bread': 0.06, 'vegetables': 0.04, 'butter': 0.01, 'cheese': 0.02},
    'sattu': {'sattu_flour': 0.1, 'onions': 0.02, 'spices': 0.01, 'water': 0.05},
    'vada': {'lentils': 0.06, 'spices': 0.01, 'oil': 0.03, 'onions': 0.01},
    'littichoka': {'sattu_flour': 0.1, 'spices': 0.02, 'potato': 0.05, 'tomatoes': 0.03, 'oil': 0.02}
}

# ---- RAW MATERIAL MONTHLY PROJECTION ----
def calculate_monthly_breakdown(detection_df, year, month):
    item_counts = dict(zip(
        detection_df['Class'].str.lower().str.replace(" ", "_"), 
        detection_df['Count']
    ))
    days = calendar.monthrange(year, month)[1]
    monthly_item = {k: v * days for k, v in item_counts.items()}

    raw_totals = defaultdict(float)
    breakdown_rows = []
    for item, month_count in monthly_item.items():
        if item not in RAW_MATERIALS_USAGE:
            continue
        for rmat, peritem_qty in RAW_MATERIALS_USAGE[item].items():
            tot_qty = peritem_qty * month_count
            raw_totals[rmat] += tot_qty
            breakdown_rows.append({
                "Food Item": item,
                "Monthly Count": month_count,
                "Raw Material": rmat,
                "Qty per Item": peritem_qty,
                "Total Qty Month": tot_qty,
            })
    return (
        pd.DataFrame({
            "Food Item": list(monthly_item.keys()),
            "Monthly Count": list(monthly_item.values())
        }),
        pd.DataFrame({
            "Raw Material": list(raw_totals.keys()),
            "Total Qty Month": list(raw_totals.values())
        }),
        pd.DataFrame(breakdown_rows)
    )


def build_procurement_markdown(df, year, month):
    days = calendar.monthrange(year, month)[1]
    items = []
    monthly_counts = {}
    for _, row in df.iterrows():
        item_name = row["Class"]
        count = row["Count"]
        monthly = count * days
        items.append((item_name, count, days, monthly))
        monthly_counts[item_name.lower().replace(" ", "_")] = (monthly, count)
    orig_name = {k.lower().replace("_"," "):k for k in monthly_counts}
    raw_material_totals = {}
    calculation_rows = []
    for item_key, (monthly_count, daily_count) in monthly_counts.items():
        rm_dict = RAW_MATERIALS_USAGE.get(item_key, {})
        for mat, per_item_qty in rm_dict.items():
            total_qty = per_item_qty * monthly_count
            raw_material_totals.setdefault(mat, [0.0, []])[0] += total_qty
            raw_material_totals[mat][1].append(
                f"{orig_name.get(item_key, item_key)}: {monthly_count}×{per_item_qty:.3f}"
            )
            calculation_rows.append(
                (orig_name.get(item_key, item_key), monthly_count, mat, per_item_qty, total_qty)
            )
    md1 = '## 1. Estimated Monthly Consumption of Food Items\n\n'
    md1 += "| Food Item | Daily Count | Days in Month | Estimated Monthly Count |\n"
    md1 += "|-----------|:-----------:|:-------------:|:----------------------:|\n"
    for item, daily, days, monthly in items:
        md1 += f"| {item} | {daily} | {days} | {monthly} |\n"
    md2 = f'## 2. Estimated Raw Material Requirement for {calendar.month_name[month]} {year}\n\n'
    md2 += "| Raw Material | Calculation Details | Total Quantity (kg/units) |\n"
    md2 += "|--------------|--------------------|:------------------------:|\n"
    for mat, (total, calcs) in raw_material_totals.items():
        md2 += f"| {mat} | {' + '.join(calcs)} | {total:.2f} |\n"
    md3 = '## 3. Detailed Ingredient-wise Breakdown\n\n'
    md3 += "| Item | Monthly Count | Raw Material | Per Item Qty | Total Qty for Month (kg/units) |\n"
    md3 += "|------|:-------------:|:------------:|:------------:|:------------------------------:|\n"
    for row in calculation_rows:
        md3 += f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]:.2f} |\n"
   
    return md1 + '\n' + md2 + '\n' + md3

st.title("Food Tracker")


st.header("Analysis")
uploaded_video = st.file_uploader("Upload video/ image", type=["mp4", "mov", "jpg", "png", "jpeg"])

model_path = "model.pt"

if uploaded_video:
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_video.read())
    st.video("temp_video.mp4")

    if st.button("Detect Items"):
        try:
            with st.spinner("Processing video and detecting items..."):
                output_video, csv_path, df = count_unique_objects(
                    "temp_video.mp4", "annotated_output.mp4", model_path
                )

            st.subheader("Current Detection Results")
            st.dataframe(df)
            
            # Show CSV download button here, only after detection
            with open(csv_path, "rb") as f:
                st.download_button("Download Report", f, file_name=os.path.basename(csv_path))

        except Exception as e:
            st.error(f"An error occurred during detection: {e}")

st.divider()

# ---- MONTHLY PROCUREMENT SECTION (ALWAYS AVAILABLE) ----
st.header("Monthly Procurement Report")
st.write("Click the button below to view cumulative requirements")

if st.button("Monthly Procurement"):
    try:
        # Load all existing CSV files
        all_csvs = glob.glob("counts_*.csv")
        
        if all_csvs:
            st.info(f"Found {len(all_csvs)} detection report...")
            
            df_list = []
            valid_files = 0
            
            for csv_file in all_csvs:
                try:
                    if os.path.getsize(csv_file) > 0:
                        temp_df = pd.read_csv(csv_file)
                        if not temp_df.empty:
                            df_list.append(temp_df)
                            valid_files += 1
                            st.write(f"Loaded: {csv_file}")
                        else:
                            st.write(f"Empty file: {csv_file}")
                    else:
                        st.write(f"Zero-size file: {csv_file}")
                except Exception as e:
                    st.write(f"Error reading {csv_file}: {e}")

            if df_list:
                # Combine all dataframes and sum up counts for same items
                full_df = pd.concat(df_list, ignore_index=True)
                summary_df = full_df.groupby("Class")['Count'].sum().reset_index()
                
                st.success(f"Successfully processed {valid_files} files with detection data!")
                
                # Show current accumulated totals
                st.subheader("Cumulative Daily Totals")
                st.dataframe(summary_df)
                
                # Generate and display procurement report
                today = datetime.date.today()
                procurement_markdown = build_procurement_markdown(
                    summary_df, 
                    year=today.year, 
                    month=today.month
                )
                
                st.markdown(procurement_markdown)
                
                # Additional summary metrics
                st.subheader("Summary")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Items/Day", summary_df['Count'].sum())
                with col2:
                    st.metric("Unique Food Types", len(summary_df))
                with col3:
                    days_in_month = calendar.monthrange(today.year, today.month)[1]
                    monthly_total = summary_df['Count'].sum() * days_in_month
                    st.metric("Projected Monthly Total", monthly_total)
                    
            else:
                st.warning("Empty or invalid files.")
        else:
            st.info("Invalid File. Upload again.")
            # st.markdown("""
            # ### How to generate procurement data:
            # 1. Upload a kitchen video above
            # 2. Click "Detect and Count Items" 
            # 3. Once detection is complete, come back and click this button
            # 4. The system will show accumulated data from all your detections
            # """)
            
    except Exception as e:
        st.error(f"Error processing files: {e}")
        st.write("Upload again.")

# ---- FOOTER INFO ----
st.divider()
st.subheader("Files.")

# Show information about existing files
existing_files = glob.glob("counts_*.csv")
if existing_files:
    st.write(f"**Available files:** {len(existing_files)}")
    
    # Show file details in an expandable section
    with st.expander("View details"):
        for file in existing_files:
            try:
                file_size = os.path.getsize(file)
                if file_size > 0:
                    temp_df = pd.read_csv(file)
                    st.write(f"• **{file}** - {len(temp_df)} items detected")
                else:
                    st.write(f"• **{file}** - Empty file")
            except:
                st.write(f"• **{file}** - Error reading file")
else:

    st.write("No detection files found yet.")
