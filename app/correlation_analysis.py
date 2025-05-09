import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from datetime import datetime, date
import seaborn as sns
from scipy import stats

# Add the project root directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import necessary functions from other modules
from src.drug_events import (
    get_top_drug_reactions,
    get_drug_events_by_patient_sex,
    get_drug_events_by_patient_weight,
    adverse_events_by_patient_age_group_within_data_range,
    get_drug_therapeutic_response,
    get_drug_manufacturer_distribution,
    get_drug_indications
)
from src.food_endpoints import (
    get_food_recalls_by_classification,
    get_food_recalls_by_reason,
    get_food_events_by_product,
    get_food_events_by_symptom
)
from src.tobacco_endpoints import (
    get_tobacco_reports_by_health_effect,
    get_tobacco_reports_by_product,
    get_tobacco_reports_by_problem_type
)
from src.device_endpoints import (
    get_device_events_by_medical_specialty,
    get_device_events_by_type,
    get_device_recalls_by_class
)

@st.cache_data(ttl=3600)
def get_cross_category_recalls():
    """Get recall data across drug, food, and device categories for comparison."""
    # Get recall data from different categories
    drug_recalls = get_drug_manufacturer_distribution(
        st.session_state.start_date,
        st.session_state.end_date,
        100
    )

    food_recalls = get_food_recalls_by_classification(
        st.session_state.start_date,
        st.session_state.end_date,
        100
    )

    device_recalls = get_device_recalls_by_class(
        st.session_state.start_date,
        st.session_state.end_date,
        100
    )

    # Format data for comparison
    # Handle drug recalls - could be DataFrame or dict with categorized field
    if isinstance(drug_recalls, dict) and "categorized" in drug_recalls:
        drug_df = drug_recalls["categorized"]
        drug_recalls_sum = drug_df['Count'].sum() if not drug_df.empty else 0
    elif isinstance(drug_recalls, pd.DataFrame) and not drug_recalls.empty:
        drug_recalls_sum = drug_recalls['Count'].sum()
    else:
        drug_recalls_sum = 0

    # Handle food recalls - could be DataFrame or dict with categorized field
    if isinstance(food_recalls, dict) and "categorized" in food_recalls:
        food_df = food_recalls["categorized"]
        food_recalls_sum = food_df['Count'].sum() if not food_df.empty else 0
    elif isinstance(food_recalls, pd.DataFrame) and not food_recalls.empty:
        food_recalls_sum = food_recalls['Count'].sum()
    else:
        food_recalls_sum = 0

    # Handle device recalls - could be DataFrame or dict with categorized field
    if isinstance(device_recalls, dict) and "categorized" in device_recalls:
        device_df = device_recalls["categorized"]
        device_recalls_sum = device_df['Count'].sum() if not device_df.empty else 0
    elif isinstance(device_recalls, pd.DataFrame) and not device_recalls.empty:
        device_recalls_sum = device_recalls['Count'].sum()
    else:
        device_recalls_sum = 0

    # Create a unified DataFrame for comparison
    recalls_df = pd.DataFrame({
        'Category': ['Drug', 'Food', 'Device'],
        'Recall Count': [drug_recalls_sum, food_recalls_sum, device_recalls_sum]
    })

    return recalls_df

@st.cache_data(ttl=3600)
def get_demographic_vs_adverse_events():
    """Analyze correlation between patient demographics and adverse events."""
    # Get demographic data
    age_data = adverse_events_by_patient_age_group_within_data_range(
        st.session_state.start_date.strftime('%Y-%m-%d'),
        st.session_state.end_date.strftime('%Y-%m-%d')
    )

    # Extract age data - could be DataFrame or dict
    if isinstance(age_data, dict) and "categorized" in age_data:
        age_data = age_data["categorized"]
    elif not isinstance(age_data, pd.DataFrame):
        age_data = pd.DataFrame()

    sex_data = get_drug_events_by_patient_sex(
        st.session_state.start_date,
        st.session_state.end_date
    )

    # Extract sex data - could be DataFrame or dict
    if isinstance(sex_data, dict) and "categorized" in sex_data:
        sex_data = sex_data["categorized"]
    elif not isinstance(sex_data, pd.DataFrame):
        sex_data = pd.DataFrame()

    weight_data = get_drug_events_by_patient_weight()

    # Extract weight data - could be DataFrame or dict
    if isinstance(weight_data, dict) and "categorized" in weight_data:
        weight_data = weight_data["categorized"]
    elif not isinstance(weight_data, pd.DataFrame):
        weight_data = pd.DataFrame()

    # Get adverse reaction data
    reaction_data = get_top_drug_reactions(
        st.session_state.start_date,
        st.session_state.end_date
    )

    # Extract reaction data - could be DataFrame or dict
    if isinstance(reaction_data, dict) and "categorized" in reaction_data:
        reaction_data = reaction_data["categorized"]
    elif not isinstance(reaction_data, pd.DataFrame):
        reaction_data = pd.DataFrame()

    # Prepare processed results
    results = {
        "age_data": age_data,
        "sex_data": sex_data,
        "weight_data": weight_data,
        "reaction_data": reaction_data
    }

    return results

@st.cache_data(ttl=3600)
def analyze_health_effects_across_categories():
    """Analyze health effects across different product categories."""
    # Get health effects data from different categories
    drug_reactions = get_top_drug_reactions(
        st.session_state.start_date,
        st.session_state.end_date,
        50
    )

    tobacco_effects = get_tobacco_reports_by_health_effect(
        st.session_state.start_date,
        st.session_state.end_date,
        50
    )

    food_symptoms = get_food_events_by_symptom(
        st.session_state.start_date,
        st.session_state.end_date,
        50
    )

    # Process and return data
    health_effects = {
        "drug_reactions": drug_reactions if isinstance(drug_reactions, pd.DataFrame)
                          else drug_reactions.get("categorized", pd.DataFrame()) if isinstance(drug_reactions, dict)
                          else pd.DataFrame(),
        "tobacco_effects": tobacco_effects.get("categorized", pd.DataFrame()) if isinstance(tobacco_effects, dict)
                          else tobacco_effects if isinstance(tobacco_effects, pd.DataFrame)
                          else pd.DataFrame(),
        "food_symptoms": food_symptoms.get("categorized", pd.DataFrame()) if isinstance(food_symptoms, dict)
                         else food_symptoms if isinstance(food_symptoms, pd.DataFrame)
                         else pd.DataFrame()
    }

    return health_effects

@st.cache_data(ttl=3600)
def get_product_vs_problem_correlation():
    """Get correlation between product types and reported problems."""
    # Get product and problem data
    tobacco_products = get_tobacco_reports_by_product(
        st.session_state.start_date,
        st.session_state.end_date
    )

    tobacco_problems = get_tobacco_reports_by_problem_type(
        st.session_state.start_date,
        st.session_state.end_date
    )

    device_types = get_device_events_by_type(
        st.session_state.start_date,
        st.session_state.end_date
    )

    device_problems = get_device_events_by_medical_specialty(
        st.session_state.start_date,
        st.session_state.end_date
    )

    # Process data for correlation analysis
    results = {
        "tobacco_products": tobacco_products.get("categorized", pd.DataFrame()) if isinstance(tobacco_products, dict)
                            else tobacco_products if isinstance(tobacco_products, pd.DataFrame)
                            else pd.DataFrame(),
        "tobacco_problems": tobacco_problems.get("categorized", pd.DataFrame()) if isinstance(tobacco_problems, dict)
                           else tobacco_problems if isinstance(tobacco_problems, pd.DataFrame)
                           else pd.DataFrame(),
        "device_types": device_types.get("categorized", pd.DataFrame()) if isinstance(device_types, dict)
                       else device_types if isinstance(device_types, pd.DataFrame)
                       else pd.DataFrame(),
        "device_problems": device_problems.get("categorized", pd.DataFrame()) if isinstance(device_problems, dict)
                          else device_problems if isinstance(device_problems, pd.DataFrame)
                          else pd.DataFrame()
    }

    return results

def calculate_correlation_between_categories(df1, df2, col1, col2):
    """Calculate correlation between two data categories."""
    if df1.empty or df2.empty:
        return None, None

    # Merge data if possible or create a correlation analysis
    try:
        merged_df = pd.merge(df1, df2, left_index=True, right_index=True)
        if merged_df.empty:
            return None, None

        # Calculate correlation
        correlation, p_value = stats.pearsonr(
            merged_df[col1].astype(float),
            merged_df[col2].astype(float)
        )
        return correlation, p_value
    except:
        return None, None

def display_correlation_analysis():
    """Display correlation analysis between different FDA data points."""
    st.title("Correlation Analysis")

    st.write("""
    This section analyzes correlations between different FDA data points to identify patterns and relationships
    that might not be apparent when looking at individual categories.
    """)

    # Create tabs for different correlation analyses
    tab1, tab2, tab3, tab4 = st.tabs([
        "Cross-Category Recalls",
        "Demographics vs. Adverse Events",
        "Health Effects Comparison",
        "Product-Problem Correlation"
    ])

    # Tab 1: Cross-Category Recalls
    with tab1:
        st.header("Cross-Category Recall Analysis")

        try:
            with st.spinner("Loading cross-category recall data..."):
                recalls_df = get_cross_category_recalls()

                if recalls_df.empty or recalls_df["Recall Count"].sum() == 0:
                    st.warning("No recall data available for the selected time period.")
                else:
                    # Create visualizations
                    col1, col2 = st.columns(2)

                    with col1:
                        # Bar chart for comparing recalls across categories
                        fig = px.bar(
                            recalls_df,
                            x="Category",
                            y="Recall Count",
                            title="Recall Comparison Across Categories",
                            color="Category",
                            text="Recall Count"
                        )
                        fig.update_traces(textposition='outside')
                        st.plotly_chart(fig, use_container_width=True)

                    with col2:
                        # Pie chart for distribution
                        fig = px.pie(
                            recalls_df,
                            values="Recall Count",
                            names="Category",
                            title="Distribution of Recalls by Category"
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    # Calculate relative risk metrics
                    total_recalls = recalls_df["Recall Count"].sum()
                    recalls_df["Percentage"] = (recalls_df["Recall Count"] / total_recalls * 100).round(2)
                    recalls_df["Percentage"] = recalls_df["Percentage"].apply(lambda x: f"{x}%")

                    st.subheader("Recall Distribution Analysis")
                    st.dataframe(recalls_df, use_container_width=True)

                    # Add interpretation
                    st.subheader("Interpretation")

                    # Calculate which category has the most recalls
                    if total_recalls > 0:
                        max_category = recalls_df.loc[recalls_df["Recall Count"].idxmax()]
                        st.markdown(f"""
                        - **{max_category['Category']}** has the highest number of recalls ({max_category['Recall Count']}),
                          representing {max_category['Percentage']} of all recalls.
                        - This cross-category comparison helps identify which product categories might need more
                          regulatory attention or quality control measures.
                        """)
        except Exception as e:
            st.error(f"An error occurred while analyzing recall data: {str(e)}")
            st.info("This could be due to network connectivity issues or API limitations. Please try adjusting the date range or sample size.")

    # Tab 2: Demographics vs. Adverse Events
    with tab2:
        st.header("Demographics vs. Adverse Events Analysis")

        try:
            with st.spinner("Loading demographic and adverse event data..."):
                demo_vs_events = get_demographic_vs_adverse_events()

                if all(df.empty for df in demo_vs_events.values()):
                    st.warning("No demographic or adverse event data available for the selected time period.")
                else:
                    # Age vs. Adverse Events section removed as requested

                    # Sex vs. Adverse Events
                    st.subheader("Sex vs. Adverse Events")
                    if not demo_vs_events["sex_data"].empty:
                        # Remove percentage sign and convert to float for analysis
                        try:
                            demo_vs_events["sex_data"]["Percentage_Value"] = demo_vs_events["sex_data"]["Percentage"].apply(
                                lambda x: float(x.replace("%", "")) if isinstance(x, str) else x
                            )
                        except Exception:
                            # Handle case where Percentage isn't a string or doesn't contain '%'
                            pass

                        fig = px.pie(
                            demo_vs_events["sex_data"],
                            values="Count",
                            names="Sex",
                            title="Adverse Events by Sex"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No sex data available.")

                    # Weight vs. Adverse Events
                    st.subheader("Weight Group vs. Adverse Events")
                    if not demo_vs_events["weight_data"].empty:
                        fig = px.bar(
                            demo_vs_events["weight_data"],
                            x="Weight Group",
                            y="Count",
                            title="Adverse Events by Weight Group",
                            color="Weight Group"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No weight data available.")

                    # Analysis of top reactions by demographic groups
                    st.subheader("Analysis and Interpretation")
                    st.markdown("""
                    This correlation analysis reveals patterns between patient demographics and adverse events:
                    """)

                    # Generate insights based on available data
                    insights = []

                    if not demo_vs_events["sex_data"].empty and "Sex" in demo_vs_events["sex_data"].columns and "Percentage" in demo_vs_events["sex_data"].columns:
                        for index, row in demo_vs_events["sex_data"].iterrows():
                            insights.append(f"- **{row['Sex']}** patients account for {row['Percentage']} of adverse events")

                    if not demo_vs_events["weight_data"].empty and "Weight Group" in demo_vs_events["weight_data"].columns and "Count" in demo_vs_events["weight_data"].columns:
                        max_weight_group = demo_vs_events["weight_data"].loc[demo_vs_events["weight_data"]["Count"].idxmax()]
                        insights.append(f"- Weight group **{max_weight_group['Weight Group']}** has the highest incidence of adverse events ({max_weight_group['Count']})")

                    if insights:
                        for insight in insights:
                            st.markdown(insight)
                    else:
                        st.info("Insufficient data to generate demographic insights.")
        except Exception as e:
            st.error(f"An error occurred while analyzing demographic data: {str(e)}")
            st.info("This could be due to network connectivity issues or API limitations. Please try adjusting the date range or sample size.")

    # Tab 3: Health Effects Comparison
    with tab3:
        st.header("Health Effects Comparison Across Categories")

        try:
            with st.spinner("Loading health effects data..."):
                health_effects = analyze_health_effects_across_categories()

                if all(df.empty for df in health_effects.values()):
                    st.warning("No health effects data available across categories.")
                else:
                    # Prepare data for unified analysis
                    combined_effects = []

                    # Process drug reactions
                    if not health_effects["drug_reactions"].empty:
                        top_drug_reactions = health_effects["drug_reactions"].head(10)
                        for _, row in top_drug_reactions.iterrows():
                            try:
                                effect_data = {
                                    "Category": "Drug",
                                    "Effect": row["Reaction"] if "Reaction" in row else row.iloc[0],
                                    "Count": row["Count"] if "Count" in row else row.iloc[1],
                                    "Type": row["Category"] if "Category" in row else "Uncategorized"
                                }
                                combined_effects.append(effect_data)
                            except Exception:
                                continue

                    # Process tobacco effects
                    if not health_effects["tobacco_effects"].empty:
                        top_tobacco_effects = health_effects["tobacco_effects"].head(10)
                        for _, row in top_tobacco_effects.iterrows():
                            try:
                                effect_data = {
                                    "Category": "Tobacco",
                                    "Effect": row["Health Effect"] if "Health Effect" in row else row.iloc[0],
                                    "Count": row["Count"] if "Count" in row else row.iloc[1],
                                    "Type": "Health Effect"
                                }
                                combined_effects.append(effect_data)
                            except Exception:
                                continue

                    # Process food symptoms
                    if not health_effects["food_symptoms"].empty:
                        top_food_symptoms = health_effects["food_symptoms"].head(10)
                        for _, row in top_food_symptoms.iterrows():
                            try:
                                effect_data = {
                                    "Category": "Food",
                                    "Effect": row["Symptom"] if "Symptom" in row else row.iloc[0],
                                    "Count": row["Count"] if "Count" in row else row.iloc[1],
                                    "Type": "Symptom"
                                }
                                combined_effects.append(effect_data)
                            except Exception:
                                continue

                    if combined_effects:
                        combined_df = pd.DataFrame(combined_effects)

                        # Create visualization
                        st.subheader("Top Health Effects Across Categories")

                        # Group bar chart comparing health effects across categories
                        fig = px.bar(
                            combined_df,
                            x="Effect",
                            y="Count",
                            color="Category",
                            title="Health Effects Comparison Across Product Categories",
                            barmode="group",
                            height=600
                        )
                        fig.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig, use_container_width=True)

                        # Heatmap for health effect types by category
                        pivot_df = combined_df.pivot_table(
                            index="Type",
                            columns="Category",
                            values="Count",
                            aggfunc="sum",
                            fill_value=0
                        )

                        # Create heatmap
                        fig = px.imshow(
                            pivot_df,
                            text_auto=True,
                            aspect="auto",
                            title="Health Effect Types by Product Category",
                            color_continuous_scale="Viridis"
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Display data table in a collapsible expander as requested
                        with st.expander("View Health Effects Data", expanded=False):
                            st.dataframe(combined_df, use_container_width=True)

                        # Analysis and interpretation
                        st.subheader("Analysis and Interpretation")

                        # Common effects across categories
                        common_effects = []
                        for effect in combined_df["Effect"].unique():
                            categories = combined_df[combined_df["Effect"] == effect]["Category"].unique()
                            if len(categories) > 1:
                                common_effects.append((effect, len(categories), categories))

                        if common_effects:
                            st.markdown("**Common Health Effects Across Multiple Categories:**")
                            for effect, count, categories in sorted(common_effects, key=lambda x: x[1], reverse=True):
                                st.markdown(f"- **{effect}** appears in {count} categories: {', '.join(categories)}")
                        else:
                            st.markdown("No common health effects found across different product categories.")

                        # Category-specific findings
                        st.markdown("**Category-Specific Findings:**")

                        for category in combined_df["Category"].unique():
                            category_df = combined_df[combined_df["Category"] == category]
                            if not category_df.empty:
                                top_effect = category_df.loc[category_df["Count"].idxmax()]
                                st.markdown(f"- **{category}**: Most common effect is *{top_effect['Effect']}* with {top_effect['Count']} reports")
                    else:
                        st.info("Insufficient data to perform health effects comparison.")
        except Exception as e:
            st.error(f"An error occurred while analyzing health effects data: {str(e)}")
            st.info("This could be due to network connectivity issues or API limitations. Please try adjusting the date range or sample size.")

    # Tab 4: Product-Problem Correlation
    with tab4:
        st.header("Product Type and Problem Correlation")

        try:
            with st.spinner("Loading product and problem data..."):
                product_problem_data = get_product_vs_problem_correlation()

                if all(df.empty for df in product_problem_data.values()):
                    st.warning("No product or problem data available.")
                else:
                    # Analyze tobacco data
                    st.subheader("Tobacco Products vs. Problem Types")

                    if not product_problem_data["tobacco_products"].empty and not product_problem_data["tobacco_problems"].empty:
                        # Display top products
                        col1, col2 = st.columns(2)

                        with col1:
                            top_products = product_problem_data["tobacco_products"].head(5)
                            y_col = "Product Type" if "Product Type" in top_products.columns else top_products.columns[0]
                            fig = px.bar(
                                top_products,
                                x="Count",
                                y=y_col,
                                title="Top Tobacco Products",
                                orientation="h"
                            )
                            st.plotly_chart(fig, use_container_width=True)

                        with col2:
                            top_problems = product_problem_data["tobacco_problems"].head(5)
                            y_col = "Problem Type" if "Problem Type" in top_problems.columns else top_problems.columns[0]
                            fig = px.bar(
                                top_problems,
                                x="Count",
                                y=y_col,
                                title="Top Problem Types",
                                orientation="h"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Insufficient tobacco product or problem data.")

                    # Device Types vs. Medical Specialties section removed as requested

                    # Correlation analysis and interpretation
                    st.subheader("Correlation Analysis and Insights")

                    st.markdown("""
                    This analysis explores the relationship between product types and the problems/issues reported:
                    """)

                    # Generate insights if data is available
                    insights = []

                    # Tobacco insights
                    if not product_problem_data["tobacco_products"].empty and not product_problem_data["tobacco_problems"].empty:
                        top_tobacco_product = product_problem_data["tobacco_products"].iloc[0]
                        top_tobacco_problem = product_problem_data["tobacco_problems"].iloc[0]

                        product_name = top_tobacco_product["Product Type"] if "Product Type" in top_tobacco_product.index else top_tobacco_product.iloc[0]
                        problem_name = top_tobacco_problem["Problem Type"] if "Problem Type" in top_tobacco_problem.index else top_tobacco_problem.iloc[0]
                        product_count = top_tobacco_product["Count"] if "Count" in top_tobacco_product else top_tobacco_product.iloc[1]
                        problem_count = top_tobacco_problem["Count"] if "Count" in top_tobacco_problem else top_tobacco_problem.iloc[1]

                        insights.append(f"- For tobacco products, **{product_name}** is most frequently reported with {product_count} reports")
                        insights.append(f"- The most common problem type is **{problem_name}** with {problem_count} reports")

                    # Remove device insights since we removed the device section

                    # Display insights
                    if insights:
                        for insight in insights:
                            st.markdown(insight)

                        st.markdown("""
                        **Regulatory Implications:**
                        - Products with high report frequencies may benefit from additional regulatory oversight
                        - Common problem patterns across products suggest potential areas for industry-wide improvements
                        - The correlation between product types and problem categories can inform targeted safety initiatives
                        """)
                    else:
                        st.info("Insufficient data to generate product-problem correlation insights.")
        except Exception as e:
            st.error(f"An error occurred while analyzing product-problem correlation data: {str(e)}")
            st.info("This could be due to network connectivity issues or API limitations. Please try adjusting the date range or sample size.")
