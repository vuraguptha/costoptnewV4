"""
MINI SKELETON VERSION - Streamlit App Structure
===============================================
This is a simplified version to understand the app structure.
It includes only 2-3 core components:
1. Session State Management
2. Sidebar with Input Mode Selection
3. Simple Form Inputs & Results Display

Run this file to see how the basic structure works.
"""

import streamlit as st
import pandas as pd
import plotly.express as px

# ============================================
# STEP 1: PAGE CONFIGURATION
# ============================================
# This sets up the page title, layout, and icon
st.set_page_config(
    page_title="Test App - Skeleton",
    layout="wide",
    page_icon="🧞‍♂️"
)

# ============================================
# STEP 2: SESSION STATE INITIALIZATION
# ============================================
# Session state stores data that persists across reruns
# This is crucial for maintaining user input and app state

# Initialize key session state variables
if "submitted" not in st.session_state:
    st.session_state.submitted = False  # Track if analysis has been submitted

if "input_mode" not in st.session_state:
    st.session_state.input_mode = "Manual"  # Current input mode

if "show_welcome" not in st.session_state:
    st.session_state.show_welcome = True  # Show welcome screen initially

# Initialize form input values
if "int_count" not in st.session_state:
    st.session_state.int_count = 0  # International transfer count

if "int_cost" not in st.session_state:
    st.session_state.int_cost = 0.0  # Cost per international transfer

if "dom_count" not in st.session_state:
    st.session_state.dom_count = 0  # Domestic transfer count

if "dom_cost" not in st.session_state:
    st.session_state.dom_cost = 0.0  # Cost per domestic transfer

# Store analysis results
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None

# ============================================
# STEP 3: SIMPLE CALCULATION FUNCTION
# ============================================
# This mimics the package analysis logic (simplified)
def calculate_simple_analysis(int_count, int_cost, dom_count, dom_cost):
    """
    Simple calculation: Total cost = (int_count * int_cost) + (dom_count * dom_cost)
    In the real app, this would call suggest_best_package() with complex logic
    """
    total_cost = (int_count * int_cost) + (dom_count * dom_cost)
    
    # Simulate package options (simplified)
    package_a_cost = total_cost * 0.8  # 20% savings
    package_b_cost = total_cost * 0.9  # 10% savings
    
    # Find best package
    if package_a_cost < package_b_cost:
        best_package = "Package A"
        savings = total_cost - package_a_cost
    else:
        best_package = "Package B"
        savings = total_cost - package_b_cost
    
    results = {
        "without_package": total_cost,
        "package_a": package_a_cost,
        "package_b": package_b_cost,
        "best": best_package,
        "savings": savings
    }
    
    return results

# ============================================
# STEP 4: SIDEBAR - INPUT MODE SELECTION
# ============================================
# The sidebar contains navigation and input controls

with st.sidebar:
    st.markdown("### 🧭 Choose Input Mode")
    
    # Mode selection buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📝 Manual", key="btn_manual", use_container_width=True):
            st.session_state.input_mode = "Manual"
            st.session_state.submitted = False
            st.session_state.show_welcome = True
            st.rerun()  # Rerun the app to reflect changes
    
    with col2:
        if st.button("🤖 AI", key="btn_ai", use_container_width=True):
            st.session_state.input_mode = "AI Assistant"
            st.session_state.submitted = False
            st.session_state.show_welcome = True
            st.rerun()
    
    st.markdown("---")
    
    # Show current mode
    st.info(f"Current Mode: **{st.session_state.input_mode}**")
    
    # ============================================
    # STEP 5: MANUAL MODE FORM (in sidebar)
    # ============================================
    if st.session_state.input_mode == "Manual":
        st.markdown("### 📝 Transaction Details")
        
        # International Transfers
        with st.expander("🌍 International Transfers", expanded=True):
            st.session_state.int_count = st.number_input(
                "Count", 
                min_value=0, 
                value=st.session_state.int_count,
                key="input_int_count"
            )
            st.session_state.int_cost = st.number_input(
                "Cost (AED)", 
                min_value=0.0, 
                value=st.session_state.int_cost,
                step=0.1,
                key="input_int_cost"
            )
        
        # Domestic Transfers
        with st.expander("🏠 Domestic Transfers"):
            st.session_state.dom_count = st.number_input(
                "Count", 
                min_value=0, 
                value=st.session_state.dom_count,
                key="input_dom_count"
            )
            st.session_state.dom_cost = st.number_input(
                "Cost (AED)", 
                min_value=0.0, 
                value=st.session_state.dom_cost,
                step=0.1,
                key="input_dom_cost"
            )
        
        # Analyze Button
        if st.button("🔍 Analyze", use_container_width=True, key="btn_analyze"):
            with st.spinner("Analyzing..."):
                # Perform calculation
                results = calculate_simple_analysis(
                    st.session_state.int_count,
                    st.session_state.int_cost,
                    st.session_state.dom_count,
                    st.session_state.dom_cost
                )
                
                # Store results in session state
                st.session_state.analysis_results = results
                st.session_state.submitted = True
                st.session_state.show_welcome = False
                st.rerun()
        
        # Reset Button (only show after analysis)
        if st.session_state.submitted and st.session_state.analysis_results:
            if st.button("🔄 Reset", use_container_width=True, key="btn_reset"):
                # Reset all values
                st.session_state.int_count = 0
                st.session_state.int_cost = 0.0
                st.session_state.dom_count = 0
                st.session_state.dom_cost = 0.0
                st.session_state.submitted = False
                st.session_state.show_welcome = True
                st.session_state.analysis_results = None
                st.rerun()
    
    # ============================================
    # STEP 6: AI ASSISTANT MODE (simplified)
    # ============================================
    elif st.session_state.input_mode == "AI Assistant":
        st.markdown("### 💬 AI Assistant")
        st.info("AI Assistant mode - Simplified version")
        st.write("In the full app, this would have a chat interface")
        st.write("that guides you through questions step by step.")

# ============================================
# STEP 7: MAIN CONTENT AREA
# ============================================

# Welcome Screen
if st.session_state.show_welcome:
    st.title("🧞‍♂️ Welcome to Test App - Skeleton Version")
    st.markdown("""
    ### How to use this skeleton:
    1. **Choose Input Mode** from the sidebar (Manual or AI)
    2. **Enter transaction details** in the sidebar form
    3. **Click Analyze** to see results
    4. **View results** in the main area below
    
    This skeleton demonstrates:
    - Session state management
    - Sidebar navigation
    - Form inputs
    - Results display
    """)

# ============================================
# STEP 8: RESULTS DISPLAY
# ============================================
# Show results if analysis has been submitted
if st.session_state.submitted and st.session_state.analysis_results:
    results = st.session_state.analysis_results
    
    st.markdown("---")
    st.markdown("## 📊 Analysis Results")
    
    # Display key metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Without Package", f"{results['without_package']:,.0f} AED")
    with col2:
        st.metric("Best Package", results['best'])
    with col3:
        st.metric("Savings", f"{results['savings']:,.0f} AED", delta=f"{results['savings']:,.0f} AED")
    
    st.markdown("---")
    
    # ============================================
    # STEP 9: SIMPLE CHART (using Plotly)
    # ============================================
    st.markdown("### 💰 Cost Comparison Chart")
    
    # Prepare data for chart
    chart_data = pd.DataFrame({
        "Option": ["Without Package", "Package A", "Package B"],
        "Cost (AED)": [
            results['without_package'],
            results['package_a'],
            results['package_b']
        ]
    })
    
    # Create bar chart
    fig = px.bar(
        chart_data,
        x="Option",
        y="Cost (AED)",
        title="Cost Comparison",
        color="Option",
        color_discrete_map={
            "Without Package": "#808080",
            "Package A": "#CD2026",
            "Package B": "#F4B6B6"
        }
    )
    
    # Update layout
    fig.update_layout(
        showlegend=False,
        plot_bgcolor='white',
        height=400
    )
    
    # Display chart
    st.plotly_chart(fig, use_container_width=True)
    
    # ============================================
    # STEP 10: DETAILED BREAKDOWN
    # ============================================
    st.markdown("### 📋 Detailed Breakdown")
    
    breakdown_df = pd.DataFrame({
        "Component": ["International", "Domestic"],
        "Count": [st.session_state.int_count, st.session_state.dom_count],
        "Cost per Unit": [st.session_state.int_cost, st.session_state.dom_cost],
        "Total Cost": [
            st.session_state.int_count * st.session_state.int_cost,
            st.session_state.dom_count * st.session_state.dom_cost
        ]
    })
    
    st.dataframe(breakdown_df, use_container_width=True)
    
    st.markdown("---")
    st.success(f"✅ Analysis complete! **{results['best']}** is recommended with savings of **{results['savings']:,.0f} AED**")

# ============================================
# NOTES FOR LEARNING:
# ============================================
"""
KEY CONCEPTS DEMONSTRATED:

1. SESSION STATE:
   - Used to store data across reruns
   - Essential for maintaining user inputs
   - Example: st.session_state.int_count

2. SIDEBAR:
   - Created with: with st.sidebar:
   - Contains navigation and input controls
   - Persists across page interactions

3. BUTTONS & RERUNS:
   - Buttons trigger actions
   - st.rerun() refreshes the app
   - Use keys to avoid conflicts

4. CONDITIONAL RENDERING:
   - if st.session_state.submitted: shows results
   - Different UI based on input_mode

5. CHARTS:
   - Using Plotly Express (px) for visualizations
   - Data prepared as pandas DataFrame

6. FORM INPUTS:
   - st.number_input() for numeric values
   - Values stored in session_state
   - Keys must be unique

TO EXPAND THIS SKELETON:
- Add more transaction types
- Implement real package calculation logic
- Add AI Assistant chat flow
- Add export functionality
- Add authentication
"""

