import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Page config
st.set_page_config(
    page_title="Supplier Classification using Kraljic Model",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📦 Supplier Classification using Kraljic Portfolio Model")
st.markdown("""
This interactive app allows you to analyze and visualize suppliers based on **Supply Risk** and **Profit Impact**  
using the **Kraljic Portfolio Matrix**.
""")

# Sidebar
st.sidebar.header("📁 Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

# Default data / sample display
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📋 Uploaded Dataset")
    st.dataframe(df.head())
else:
    st.info("👆 Upload a CSV file to get started (must contain columns like `Supplier`, `Supply_Risk`, `Profit_Impact`).")

# Proceed if dataset is available
if uploaded_file is not None:
    # Check for required columns
    required_cols = ['Supplier', 'Supply_Risk', 'Profit_Impact']
    if all(col in df.columns for col in required_cols):

        # Normalize the scores (0–100 scale)
        df['Supply_Risk_Score'] = (df['Supply_Risk'] - df['Supply_Risk'].min()) / (df['Supply_Risk'].max() - df['Supply_Risk'].min()) * 100
        df['Profit_Impact_Score'] = (df['Profit_Impact'] - df['Profit_Impact'].min()) / (df['Profit_Impact'].max() - df['Profit_Impact'].min()) * 100

        # Define classification logic
        def classify_supplier(row):
            if row['Supply_Risk_Score'] > 50 and row['Profit_Impact_Score'] > 50:
                return 'Strategic'
            elif row['Supply_Risk_Score'] <= 50 and row['Profit_Impact_Score'] > 50:
                return 'Leverage'
            elif row['Supply_Risk_Score'] > 50 and row['Profit_Impact_Score'] <= 50:
                return 'Bottleneck'
            else:
                return 'Non-Critical'

        df['Category'] = df.apply(classify_supplier, axis=1)

        # Display classification result
        st.subheader("🏷️ Supplier Classification Results")
        st.dataframe(df[['Supplier', 'Supply_Risk', 'Profit_Impact', 'Category']])

        # Summary count
        st.subheader("📊 Category Distribution")
        st.bar_chart(df['Category'].value_counts())

        # Visualization: Kraljic Matrix
        st.subheader("📈 Kraljic Portfolio Matrix")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(
            x='Supply_Risk_Score',
            y='Profit_Impact_Score',
            hue='Category',
            style='Category',
            s=100,
            data=df,
            palette='Set2'
        )
        plt.axhline(50, color='gray', linestyle='--')
        plt.axvline(50, color='gray', linestyle='--')
        plt.title("Kraljic Matrix: Supplier Segmentation")
        plt.xlabel("Supply Risk (0–100)")
        plt.ylabel("Profit Impact (0–100)")
        plt.grid(True, alpha=0.3)
        st.pyplot(fig)

        # Download results
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="💾 Download Classified Data as CSV",
            data=csv,
            file_name='classified_suppliers.csv',
            mime='text/csv'
        )

    else:
        st.error(f"❌ The dataset must contain the following columns: {', '.join(required_cols)}")
