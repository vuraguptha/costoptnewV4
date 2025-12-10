import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import openai
import os
import json
import time
# import pyttsx3
# import base64 # Added for image encoding
import numpy as np
# import plotly.graph_objects as go
from matplotlib.colors import to_rgb
from gtts import gTTS
import tempfile
from export import export_to_csv, export_to_pdf
from streamlit_javascript import st_javascript
from config import ( APP_TITLE, APP_SUBTITLE, MAIN_APP_IMAGE_PATH, ADCB_LOGO_PATH,
                        WATERMARK_IMAGE_PATH, packages , manual_fields,CHAT_STAGES                    
                    )
from utils import play_text_as_speech, apply_custom_css, img_to_base64, check_password
from active_saver import get_activesaver_slabs,SavingsInterestCalculator
from UIRender import init_chat_state, process_user_response
from package_analysis import suggest_best_package, generate_narrative_summary, generate_analysis

# Set OpenAI API Key
openai.api_key =  st.secrets["OPENAI_API_KEY"]
 

# -------------------- UI CONFIG & APP NAMING --------------------

st.set_page_config(page_title="Fikra Genie", layout="wide", page_icon="🧞‍♂️")


# -------------------- SESSION STATE INITIALIZATION --------------------
# This section was previously part of the UI section but is better placed before UI rendering logic
# It ensures all session state variables are checked/initialized before any UI elements try to access them.

if "submitted" not in st.session_state:
    st.session_state.submitted = False
if "input_mode" not in st.session_state:
    st.session_state.input_mode = "Manual"
if "show_welcome" not in st.session_state:
    st.session_state.show_welcome = True
if "manual_wps_enabled" not in st.session_state: 
    st.session_state.manual_wps_enabled = False
if "tts_language" not in st.session_state: # Re-add session state for TTS language
    st.session_state.tts_language = "en" # Default to English
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# --- ActiveSaver State Management ---
if "show_activesaver" not in st.session_state:
    st.session_state.show_activesaver = False

# --- Client Profile and Form State Management ---
if "client_profiles" not in st.session_state:
    st.session_state.client_profiles = {}
if "current_client_name" not in st.session_state:
    st.session_state.current_client_name = ""
# Initialize all manual form fields in session_state if they don't exist

for key, default_value in manual_fields.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

if "chat_stage" not in st.session_state: 
    init_chat_state()

# --- Mobile Login Skip ---
# Get screen width from the browser. Default to a large number for desktop on first run.
screen_width = st_javascript("window.innerWidth", key="screen_width_js") or 1024

# If the screen is narrow (i.e., mobile), automatically set authentication to true.
if screen_width < 768:
    st.session_state.password_correct = True
# --- End Mobile Login Skip ---

# If not authenticated, show login and stop the app from running further.
# This check now respects the mobile skip logic from above.
if not st.session_state.get("password_correct", False):
    if not check_password():
        st.stop()

# --- Robust manual form reset logic: place this at the very top of your script, before any widgets ---
if 'manual_reset_pending' in st.session_state and st.session_state.manual_reset_pending:
    for key, default_value in manual_fields.items():
        st.session_state[key] = default_value
    st.session_state.show_welcome = True
    st.session_state.submitted = False
    if 'analysis_results' in st.session_state:
        del st.session_state.analysis_results
    st.session_state.manual_reset_pending = False
    st.rerun()

# -------------------- UI RENDER STARTS HERE --------------------

apply_custom_css()

# Add CSS for 80% main content width
st.markdown("""
    <style>
    .main-content-80 {
        max-width: 80vw !important;
        width: 80vw !important;
        margin-left: auto !important;
        margin-right: auto !important;
    }
    </style>
    <div class='main-content-80'>
""", unsafe_allow_html=True)

col1, col2 = st.columns([4, 1], vertical_alignment="center")
with col1:
    st.markdown(f'<p class="main-title">{APP_TITLE}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="sub-title">{APP_SUBTITLE}</p>', unsafe_allow_html=True)
with col2:
    try:
        st.image(ADCB_LOGO_PATH, width=400)
    except Exception:
        # This will show a small error message in the app if the logo is not found
        st.error(f"Logo not found", icon="🖼️")

# Main content area title
# st.title(APP_TITLE) 

# Sidebar UI
with st.sidebar:
    # New descriptive text at the top
    st.markdown("""
    <div class="sidebar-callout">
        <p>
            An intelligent, AI-driven pricing solution that empowers your team to confidently propose the optimal Business First Package.
        </p>
    </div>
    """, unsafe_allow_html=True)


    st.markdown("---") 

    st.markdown("### 🧭 Choose Input Mode")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📝 Manual Mode", key="mode_manual_revert", width="stretch", help="Enter inputs manually"):
            st.session_state.input_mode = 'Manual'
            st.session_state.submitted = False
            st.session_state.show_welcome = True
            if 'chat_stage' in st.session_state: # Clear AI state if switching from AI
                del st.session_state.chat_stage
            if 'messages' in st.session_state:
                del st.session_state.messages
            if 'transaction_data' in st.session_state: # Clear AI transaction data
                del st.session_state.transaction_data 
            # Reset manual form specific states if necessary, e.g., manual_wps_enabled for consistency
            st.session_state.manual_wps_enabled = False 
            st.rerun()
    with col2:
        if st.button("🤖 AI Assistant", key="mode_ai_revert", width="stretch", help="Chat with AI to analyze your needs"):
            st.session_state.input_mode = 'AI Assistant'
            st.session_state.submitted = False
            st.session_state.show_welcome = True
            init_chat_state() # Initialize/Reset AI chat state
            st.rerun()
    st.markdown("---")

    # Input mode specific UI (Manual Form or AI Assistant Chat)
    if st.session_state.input_mode == "Manual":
        # --- Client Profile Management ---
        with st.expander("👤 Client Profile Management"):
            
            # --- LOAD PROFILE ---
            saved_profiles = list(st.session_state.client_profiles.keys())
            if not saved_profiles:
                st.info("No saved profiles yet. Save one below.")
            else:
                profile_to_load = st.selectbox("Select a profile to load", options=[""] + saved_profiles, index=0)
                if st.button("Load Profile", width="stretch", disabled=(not profile_to_load)):
                    profile_data = st.session_state.client_profiles[profile_to_load]
                    for key, value in profile_data.items():
                        st.session_state[key] = value
                    st.session_state.current_client_name = profile_to_load
                    st.success(f"Profile '{profile_to_load}' loaded.")
                    st.rerun()

            # --- SAVE PROFILE ---
            st.markdown("---")
            new_client_name = st.text_input("Enter Client Name to Save", value=st.session_state.current_client_name)
            if st.button("Save Profile", width="stretch", disabled=(not new_client_name)):
                # Gather current form state into a dictionary
                profile_data_to_save = {key: st.session_state[key] for key in manual_fields.keys()}
                st.session_state.client_profiles[new_client_name] = profile_data_to_save
                st.session_state.current_client_name = new_client_name
                st.success(f"Profile for '{new_client_name}' saved.")

        # --- Manual Form Code (Stateful) ---
        def wps_checkbox_callback():
            # This just ensures the session state is updated immediately on checkbox change
            st.session_state.manual_wps_enabled = st.session_state.manual_form_wps_enabled_chkbx_onchange
        
        st.markdown("### 📝 Monthly Transaction Details")
        
        with st.expander("🌍 International Transfers", expanded=True):
            st.number_input("Count", 0, key="int_count_manual_form")
            st.number_input("Cost (AED)", 0.0, step=0.1, key="int_cost_manual_form")
        
        with st.expander("🏠 Domestic Transfers"):
            st.number_input("Count", 0, key="dom_count_manual_form")
            st.number_input("Cost (AED)", 0.0, step=0.1, key="dom_cost_manual_form")

        with st.expander("📑 Cheques"):
            st.number_input("Count", 0, key="chq_count_manual_form")
            st.number_input("Cost (AED)", 0.0, step=0.1, key="chq_cost_manual_form")

        with st.expander("💱 Foreign Exchange"):
            st.radio("Direction", ["Buy USD", "Sell USD"], key="fx_direction_manual_form")
            st.number_input("Amount (USD)", 0.0, key="fx_amount_manual_form")
            if st.session_state.fx_direction_manual_form == "Buy USD":
                st.number_input("Buy Rate (AED/USD)", min_value=0.0, step=0.0001, format="%.4f", key="fx_buy_rate_manual_form")
            else:
                st.number_input("Sell Rate (AED/USD)", min_value=0.0, step=0.0001, format="%.4f", key="fx_sell_rate_manual_form")

        with st.expander("💸 WPS/CST"):
            st.checkbox("Enable WPS/CST?", key="manual_form_wps_enabled_chkbx_onchange", on_change=wps_checkbox_callback)
            if st.session_state.manual_form_wps_enabled_chkbx_onchange:
                st.number_input("WPS/CST Cost (AED)", min_value=0.0, key="manual_form_wps_cost_input_field_onchange")

        with st.expander("📝 PDC Processing"):
            st.number_input("PDC Count", 0, key="pdc_count_manual_form")
            st.number_input("Cost per PDC (AED)", 0.0, step=0.1, key="pdc_cost_manual_form")

        with st.expander("📥 Inward FCY Remittance"):
            st.number_input("Inward FCY Remittance Count", 0, key="inward_fcy_count_manual_form")
            st.number_input("Cost per Inward FCY Remittance (AED)", 0.0, step=0.1, key="inward_fcy_cost_manual_form")

        with st.expander("🧾 Other Costs"):
            st.number_input("Total Other Monthly Costs (AED)", 0.0, help="Cheque submission, courier, labour, miscellaneous costs", key="other_costs_manual_form")

        if st.button("🔍 Analyze", width="stretch", key="manual_analyze_button"):
            with st.spinner("Analyzing your data... This may take a moment."):
                st.session_state.submitted = True
                st.session_state.show_welcome = False
                
                # Use the stateful values for analysis
                fx_rate = st.session_state.fx_buy_rate_manual_form if st.session_state.fx_direction_manual_form == "Buy USD" else st.session_state.fx_sell_rate_manual_form
                manual_wps_cost = st.session_state.manual_form_wps_cost_input_field_onchange if st.session_state.manual_form_wps_enabled_chkbx_onchange else 0.0

                user_data = {
                    "int_count": st.session_state.int_count_manual_form, "int_cost": st.session_state.int_cost_manual_form,
                    "dom_count": st.session_state.dom_count_manual_form, "dom_cost": st.session_state.dom_cost_manual_form,
                    "chq_count": st.session_state.chq_count_manual_form, "chq_cost": st.session_state.chq_cost_manual_form,
                    "fx_amount": st.session_state.fx_amount_manual_form, "fx_direction": st.session_state.fx_direction_manual_form,
                    "client_fx_rate": fx_rate,
                    "wps_enabled": st.session_state.manual_form_wps_enabled_chkbx_onchange, "wps_cost": manual_wps_cost,
                    "pdc_count": st.session_state.pdc_count_manual_form, "pdc_cost": st.session_state.pdc_cost_manual_form,
                    "inward_fcy_count": st.session_state.inward_fcy_count_manual_form, "inward_fcy_cost": st.session_state.inward_fcy_cost_manual_form,
                    "other_costs_input": st.session_state.other_costs_manual_form
                }
                tx = { "international": user_data["int_count"], "domestic": user_data["dom_count"], "cheque": user_data["chq_count"], "pdc": user_data["pdc_count"], "inward_fcy_remittance": user_data["inward_fcy_count"] }
                tx_cost = { "international": user_data["int_cost"], "domestic": user_data["dom_cost"], "cheque": user_data["chq_cost"], "pdc": user_data["pdc_cost"], "inward_fcy_remittance": user_data["inward_fcy_cost"] }
                
                best, savings, results = suggest_best_package( tx, tx_cost, user_data["fx_amount"], user_data["fx_direction"], user_data["client_fx_rate"], user_data["wps_cost"], user_data["other_costs_input"])
                
                if best:
                    # Generate narrative summary
                    with st.spinner("Generating AI-powered executive summary..."):
                        no_pkg_true_cost = results["Without Package"]["true_total_cost"]
                        narrative = generate_narrative_summary(best, savings, user_data, no_pkg_true_cost, results)
                    
                    st.session_state.analysis_results = { "best": best, "savings": savings, "results": results, "user_data": user_data, "tx": tx, "tx_cost": tx_cost, "narrative_summary": narrative }
                    st.rerun()
                else:
                    st.warning("No suitable package found or an error occurred during analysis.")
                    if "analysis_results" in st.session_state: del st.session_state.analysis_results
                    st.session_state.submitted = False

        # Show Reset button only after analysis/results are shown
        if st.session_state.submitted and "analysis_results" in st.session_state:
            if st.button("🔄 Reset", width="stretch", key="manual_reset_button_after_analysis"):
                st.session_state.manual_reset_pending = True
                st.rerun()

    elif st.session_state.input_mode == "AI Assistant":
        st.markdown("### 💬 AI Assistant") # Title for AI assistant mode
        
        # --- Client Profile Management for AI Mode ---
        with st.expander("👤 Client Profile Management", expanded=False):
            
            # --- LOAD PROFILE ---
            saved_profiles = list(st.session_state.client_profiles.keys())
            if not saved_profiles:
                st.info("No saved profiles yet. Save one below.")
            else:
                profile_to_load = st.selectbox("Select a profile to load", options=[""] + saved_profiles, index=0, key="ai_profile_load")
                if st.button("Load Profile", width="stretch", disabled=(not profile_to_load), key="ai_load_profile"):
                    profile_data = st.session_state.client_profiles[profile_to_load]
                    # Load data into AI transaction data structure
                    if 'transaction_data' in st.session_state:
                        # Map manual form data to AI transaction data structure
                        st.session_state.transaction_data.update({
                            "domestic": {"count": profile_data.get('dom_count_manual_form', 0), "cost": profile_data.get('dom_cost_manual_form', 0.0)},
                            "international": {"count": profile_data.get('int_count_manual_form', 0), "cost": profile_data.get('int_cost_manual_form', 0.0)},
                            "cheque": {"count": profile_data.get('chq_count_manual_form', 0), "cost": profile_data.get('chq_cost_manual_form', 0.0)},
                            "pdc": {"count": profile_data.get('pdc_count_manual_form', 0), "cost": profile_data.get('pdc_cost_manual_form', 0.0)},
                            "inward_fcy_remittance": {"count": profile_data.get('inward_fcy_count_manual_form', 0), "cost": profile_data.get('inward_fcy_cost_manual_form', 0.0)},
                            "fx": {"amount": profile_data.get('fx_amount_manual_form', 0.0), "direction": profile_data.get('fx_direction_manual_form', "Buy USD"), "rate": profile_data.get('fx_buy_rate_manual_form', 3.67)},
                            "wps": {"enabled": profile_data.get('manual_form_wps_enabled_chkbx_onchange', False), "cost": profile_data.get('manual_form_wps_cost_input_field_onchange', 0.0)},
                            "other_costs_input": profile_data.get('other_costs_manual_form', 0.0)
                        })
                    st.session_state.current_client_name = profile_to_load
                    st.success(f"Profile '{profile_to_load}' loaded into AI Assistant.")
                    st.rerun()

            # --- SAVE PROFILE ---
            st.markdown("---")
            new_client_name = st.text_input("Enter Client Name to Save", value=st.session_state.current_client_name, key="ai_client_name")
            if st.button("Save Profile", width="stretch", disabled=(not new_client_name), key="ai_save_profile"):
                # Gather current AI transaction data and convert to manual form format for saving
                if 'transaction_data' in st.session_state:
                    data = st.session_state.transaction_data
                    profile_data_to_save = {
                        'dom_count_manual_form': data["domestic"]["count"],
                        'dom_cost_manual_form': data["domestic"]["cost"],
                        'int_count_manual_form': data["international"]["count"],
                        'int_cost_manual_form': data["international"]["cost"],
                        'chq_count_manual_form': data["cheque"]["count"],
                        'chq_cost_manual_form': data["cheque"]["cost"],
                        'pdc_count_manual_form': data["pdc"]["count"],
                        'pdc_cost_manual_form': data["pdc"]["cost"],
                        'inward_fcy_count_manual_form': data["inward_fcy_remittance"]["count"],
                        'inward_fcy_cost_manual_form': data["inward_fcy_remittance"]["cost"],
                        'fx_amount_manual_form': data["fx"]["amount"],
                        'fx_direction_manual_form': data["fx"]["direction"],
                        'fx_buy_rate_manual_form': data["fx"]["rate"],
                        'fx_sell_rate_manual_form': data["fx"]["rate"],  # Use same rate for both
                        'manual_form_wps_enabled_chkbx_onchange': data["wps"]["enabled"],
                        'manual_form_wps_cost_input_field_onchange': data["wps"]["cost"],
                        'other_costs_manual_form': data["other_costs_input"]
                    }
                    st.session_state.client_profiles[new_client_name] = profile_data_to_save
                    st.session_state.current_client_name = new_client_name
                    st.success(f"Profile for '{new_client_name}' saved from AI Assistant data.")
                else:
                    st.warning("No AI transaction data available to save.")
        
        init_chat_state() # Ensure AI state is ready
        process_user_response(None) # This will handle chat UI and TTS calls
    
    # Placeholder for the Reset button, assuming it's outside the direct if/elif for input modes
    # Or if it's specific to when inputs are made.
    # For now, this structure assumes it's handled globally or was part of the reverted code.
    # Example: 
    # if st.session_state.submitted or (st.session_state.input_mode == "AI Assistant" and ...):
    #    if st.button("🔄 Reset", ...): ...

    # This HTML pushes the footer to the bottom. 
    # A more robust solution might involve CSS if the sidebar content varies a lot in height.
    st.markdown("""
    <style>
    .stApp [data-testid="stSidebarUserContent"] {
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        height: 100%; /* Might need adjustment based on other elements */
    }
    .sidebar-footer {
        text-align: center;
        color: #e4002b;
        font-size: 0.9em;
        padding-bottom: 10px; /* Add some padding */
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Footer text - will be pushed down by flexbox if the above CSS is effective
    # If not, it will appear after other elements. Forcing it to absolute bottom without enough content
    # to fill the sidebar is tricky with pure Streamlit markdown.
    st.markdown("<div class='sidebar-footer'>Powered by: CIBG Portfolio Strategy Management - Advanced Data & Analytics</div>", unsafe_allow_html=True)

    st.markdown("---")
    if st.button("Logout", width="stretch"):
        st.session_state.authenticated = False
        st.rerun()


# Main content area
if st.session_state.show_welcome:
    # Display the image on the welcome screen
    try:
        st.image(MAIN_APP_IMAGE_PATH, width="stretch") # Uses MAIN_APP_IMAGE_PATH
    except Exception as e:
        st.error(f"Main image not found at {MAIN_APP_IMAGE_PATH}. Please ensure the image is in the correct path. Error: {e}")
    
    st.markdown(f"""
    ### 🌟 Welcome to {APP_TITLE}! 
    How can I assist you today? Please choose your preferred mode from the sidebar:
    - **Manual Mode**: Enter your transaction details using forms
    - **AI Assistant**: Chat with our AI to analyze your needs
    """)

# Show analysis results in main area if available
if st.session_state.submitted and "analysis_results" in st.session_state:
    results_data = st.session_state.analysis_results
    best = results_data["best"]
    savings = results_data["savings"]
    results = results_data["results"] # This is the dictionary of all options
    user_data = results_data["user_data"]
    tx = results_data["tx"]
    tx_cost = results_data["tx_cost"]
    
    # --- Display AI Narrative Summary ---
    if "narrative_summary" in results_data and results_data["narrative_summary"]:
        st.markdown("###  Executive Summary")
        st.markdown(f"""
<div style="font-size: 1.1rem; font-style: italic; border-left: 5px solid #eee; padding-left: 1rem; margin: 1rem 0;">
{results_data['narrative_summary']}
</div>
""", unsafe_allow_html=True)
        st.markdown("---")
    
    # Graphs and export options
    if best:
        # NEW LAYOUT: Side-by-side with cost/savings graphs on left (60%) and calculations on right (40%)
        col1, col2 = st.columns([7, 3], gap="large")
        
        with col1:
            st.markdown("### 💰 Total Monthly Banking Services Cost")
            # Prepare data for the new bar logic
            # 1. Gather true total costs for all options
            all_costs = [(name, res["true_total_cost"]) for name, res in results.items()]
            no_pkg_item = next((item for item in all_costs if item[0] == "Without Package"), None)
            package_items = [item for item in all_costs if item[0] != "Without Package"]
            package_items.sort(key=lambda x: x[1])
            sorted_categories = [no_pkg_item[0]] + [name for name, _ in package_items]
            sorted_true_costs = [no_pkg_item[1]] + [cost for _, cost in package_items]

            # 2. Find the best package and its cost/fee
            best_pkg_name = best
            best_pkg_true_cost = results[best_pkg_name]["true_total_cost"]
            # This is the absolute FX cost of the BEST package, used as a baseline
            best_pkg_fx_cost_baseline = results[best_pkg_name]["breakdown"].get("Absolute FX Cost", 0)

            # 3. Calculate the 'display cost' bar value for each option
            # This logic aligns the chart with the breakdown cards on the right
            bar_values = []
            for true_cost in sorted_true_costs:
                # The display cost is the true total cost minus the absolute FX cost of the BEST package.
                # This shows all costs relative to the best FX rate.
                bar_values.append(true_cost - best_pkg_fx_cost_baseline)

            # 4. Assign bar colors
            best_idx = sorted_categories.index(best_pkg_name) if best_pkg_name in sorted_categories else -1
            n_pkgs = len(sorted_categories) - 1
            from matplotlib.colors import to_rgb
            def interpolate_color(c1, c2, t):
                return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))
            rgb_best = tuple(int(x*255) for x in to_rgb("#CD2026"))
            rgb_worst = tuple(int(x*255) for x in to_rgb("#F4B6B6"))
            bar_colors = ["#808080"]  # Without Package always gray
            for i in range(n_pkgs):
                t = i / max(n_pkgs-1, 1)  # 0 for best, 1 for worst
                rgb = interpolate_color(rgb_best, rgb_worst, t)
                bar_colors.append('#%02x%02x%02x' % rgb)
            if best_idx > 0:
                bar_colors[best_idx] = "#CD2026"
            import pandas as pd

            # Create a gap between the first bar and the rest
            x_positions = [0]
            for i in range(1, len(sorted_categories)):
                x_positions.append(i + 0.5)

            df_cost = pd.DataFrame({
                "x_pos": x_positions,
                "Category": sorted_categories,
                "Cost (AED)": bar_values,
                "Color": bar_colors
            })
            # Create the bar chart
            import plotly.express as px
            fig_cost = px.bar(df_cost, x="x_pos", y="Cost (AED)",
                              color="Category", color_discrete_sequence=bar_colors)
            
            # We will add text via annotations, so remove it from here
            fig_cost.update_traces(textposition='outside')

            fig_cost.update_layout(
                showlegend=False,
                xaxis_title=None,
                plot_bgcolor='white',
                margin=dict(l=40, r=40, t=20, b=20), # Reduced bottom margin
                height=520,
                bargap=0.2
            )
            # Remove Y-axis
            fig_cost.update_yaxes(showgrid=False, showticklabels=False, title_text=None, zeroline=False)
            
            # Format x-axis labels to be bold and on two lines
            def format_label(label):
                parts = label.split(" ")
                if len(parts) > 2: # For labels like "Package Essential plus"
                    return f"<b>{parts[0]} {parts[1]}</b><br><b>{parts[2]}</b>"
                elif len(parts) > 1: # For labels like "Package Digital"
                    return f"<b>{parts[0]}</b><br><b>{parts[1]}</b>"
                return f"<b>{label}</b>" # For single-word labels

            formatted_labels = [format_label(label) for label in sorted_categories]
            
            # Update X-axis to use new positions and larger font
            fig_cost.update_xaxes(
                showgrid=False, 
                tickangle=0, 
                tickvals=x_positions, 
                ticktext=formatted_labels,
                tickfont=dict(size=16)
            )

            # Add styled arrow for best package savings
            if best_pkg_name in sorted_categories:
                idx_no_pkg = 0
                idx_best = sorted_categories.index(best_pkg_name)
                
                x0_pos = x_positions[idx_no_pkg]
                x1_pos = x_positions[idx_best]

                y0 = bar_values[idx_no_pkg]
                y1 = bar_values[idx_best]
                savings_amt = results["Without Package"]["true_total_cost"] - best_pkg_true_cost

                fig_cost.add_annotation(
                    x=x1_pos,
                    y=y1,
                    ax=x0_pos,
                    ay=y0,
                    xref="x",
                    yref="y",
                    axref="x",
                    ayref="y",
                    text="",
                    showarrow=True,
                    arrowhead=3,
                    arrowsize=1.5,
                    arrowwidth=8,
                    arrowcolor="#240F8C", # Light green
                    opacity=1
                )

                # Add savings label centered between bars
                savings_label = f"<span style='font-size:15px;font-weight:bold;color:#228B22;line-height:1.1;'>*savings<br>{int(savings_amt):,} AED</span>"
                x_sav_pos = (x0_pos + x1_pos) / 2
                y_sav = max(y0, y1) + 0.12 * max(bar_values)
                fig_cost.add_annotation(
                    x=x_sav_pos,
                    y=y_sav,
                    text=savings_label,
                    showarrow=False,
                    font=dict(size=15, color="#228B22", family="Arial Black"),
                    align="center",
                    bordercolor=None,
                    borderwidth=0,
                    borderpad=0,
                    bgcolor=None,
                    xanchor="center",
                    yanchor="bottom"
                )

            # Add bar labels as annotations to ensure they are drawn on top of the arrow
            for i, row in df_cost.iterrows():
                fig_cost.add_annotation(
                    x=row['x_pos'],
                    y=row['Cost (AED)'],
                    text=f"{row['Cost (AED)']:,.0f}",
                    showarrow=False,
                    yshift=10,
                    font=dict(size=20, color="black"),
                    xanchor="center",
                )
            st.plotly_chart(fig_cost, width="stretch")
            
            # Add Savings Breakdown Chart
            if best_pkg_name in sorted_categories:
                st.markdown(f"### 🏅 Savings Breakdown for {best_pkg_name}")
                no_pkg_breakdown_main = results["Without Package"]["breakdown"]
                best_pkg_breakdown_main = results[best_pkg_name]["breakdown"]
                
                # Recreate the breakdown to match the screenshot's style
                savings_data = [
                    {"Component": "Other", "Savings (AED)": (no_pkg_breakdown_main.get("Other Costs (User Input)", 0) - best_pkg_breakdown_main.get("Other Costs (User Input)", 0)) + (no_pkg_breakdown_main.get("WPS/CST Cost", 0) - best_pkg_breakdown_main.get("WPS/CST Cost", 0))},
                    {"Component": "FCY", "Savings (AED)": no_pkg_breakdown_main.get("Inward Fcy Remittance Cost", 0) - best_pkg_breakdown_main.get("Inward Fcy Remittance Cost", 0)},
                    {"Component": "PDC", "Savings (AED)": no_pkg_breakdown_main.get("Pdc Cost", 0) - best_pkg_breakdown_main.get("Pdc Cost", 0)},
                    {"Component": "Chq", "Savings (AED)": no_pkg_breakdown_main.get("Cheque Transactions Cost", 0) - best_pkg_breakdown_main.get("Cheque Transactions Cost", 0)},
                    {"Component": "Dom", "Savings (AED)": no_pkg_breakdown_main.get("Domestic Transactions Cost", 0) - best_pkg_breakdown_main.get("Domestic Transactions Cost", 0)},
                    {"Component": "Intl", "Savings (AED)": no_pkg_breakdown_main.get("International Transactions Cost", 0) - best_pkg_breakdown_main.get("International Transactions Cost", 0)},
                    {"Component": "FX", "Savings (AED)": no_pkg_breakdown_main.get("Absolute FX Cost", 0) - best_pkg_breakdown_main.get("Absolute FX Cost", 0)}
                ]

                df_savings_main = pd.DataFrame(savings_data)

                # Define colors using a ranked grayscale for savings
                positive_savings = df_savings_main[df_savings_main['Savings (AED)'] > 0].sort_values('Savings (AED)', ascending=False)
                
                # Grayscale palette from dark to light based on user request
                gray_palette = ['#666666', '#808080', '#827F7F', '#A9A9A9', '#C0C0C0', '#D3D3D3']
                
                color_map = {
                    component: gray_palette[min(i, len(gray_palette) - 1)]
                    for i, component in enumerate(positive_savings['Component'])
                }

                bar_colors = []
                for _, row in df_savings_main.iterrows():
                    if row['Savings (AED)'] > 0:
                        bar_colors.append(color_map.get(row['Component'], '#CCCCCC')) # Mapped gray
                    elif row['Savings (AED)'] < 0:
                        bar_colors.append('#e4002b') # Red for costs
                    else:
                        bar_colors.append('#F0F0F0') # Light gray for zero

                fig_savings_main = go.Figure()

                fig_savings_main.add_trace(go.Bar(
                    y=df_savings_main["Component"],
                    x=df_savings_main["Savings (AED)"],
                    orientation='h',
                    marker=dict(color=bar_colors),
                    text=df_savings_main["Savings (AED)"].apply(lambda x: f"{x:,.0f} AED"),
                    textposition='auto'
                ))
                
                fig_savings_main.update_traces(
                    textangle=0, 
                    insidetextanchor='end',
                    textfont=dict(size=14, color='black')
                )

                fig_savings_main.update_layout(
                    plot_bgcolor='white',
                    xaxis_title="Savings (AED)",
                    yaxis_title=None,
                    margin=dict(l=50, r=50, t=50, b=50),
                    height=500,
                    showlegend=False,
                    # Set a fixed order for the y-axis to match the screenshot
                    yaxis=dict(
                        categoryorder='array', 
                        categoryarray=[r['Component'] for r in reversed(savings_data)]
                    )
                )

                fig_savings_main.update_xaxes(showgrid=False, zeroline=False)
                fig_savings_main.update_yaxes(showgrid=False, zeroline=False)
                
                st.plotly_chart(fig_savings_main, width="stretch")
            
            # --- PROMINENT SAVINGS HIGHLIGHT SECTION (FINAL COMPACT) ---
            st.markdown("---")

            # Construct the savings banner HTML (Benefits moved out)
            savings_banner_html = f"""
            <div style="background: #28a745; border-radius: 12px; padding: 20px; margin: 10px 0; text-align: center; box-shadow: 0 2px 8px rgba(40, 167, 69, 0.2);">
                <div style="color: white; font-size: 1.3rem; font-weight: 600; margin-bottom: 5px;">
                    🏆 RECOMMENDED PACKAGE
                </div>
                <div style="color: white; font-size: 1.8rem; font-weight: 800; margin-bottom: 12px; text-shadow: 1px 1px 2px rgba(0,0,0,0.2);">
                    {best_pkg_name}
                </div>
                <div style="color: white; font-size: 1rem; margin-bottom: 8px; opacity: 0.9;">
                    Monthly Savings
                </div>
                <div style="color: white; font-size: 2.2rem; font-weight: 800; text-shadow: 1px 1px 3px rgba(0,0,0,0.25); margin-bottom: 8px;">
                    {savings:,.0f} <span style="font-size: 1.8rem; vertical-align: middle;">.د.إ</span>
                </div>
                <div style="color: white; font-size: 0.9rem; opacity: 0.8;">
                    💰 That's {savings*12:,.0f} <span style="font-size: 0.8rem; vertical-align: middle;">.د.إ</span> annually!
                </div>
            </div>
            """
            st.markdown(savings_banner_html, unsafe_allow_html=True)

            # --- Complimentary Benefits Section (Outside the banner) ---
            complimentary_items_list = []
            if best_pkg_name in packages:
                pkg_details = packages[best_pkg_name]
                complimentary_items_list = pkg_details.get("complimentary_items", [])

            if complimentary_items_list:
                benefits_text = " &bull; ".join(complimentary_items_list)
                benefits_html = f"""
                <div style="background: #f8f9fa; border-left: 5px solid #28a745; border-radius: 8px; padding: 15px; margin: 15px 0;">
                    <h5 style="color: #28a745; margin-bottom: 10px; font-weight: 600;">🎁 Complimentary Benefits</h5>
                    <p style="color: #333; font-size: 0.95rem; margin-bottom: 0;">
                        {benefits_text}
                    </p>
                </div>
                """
                st.markdown(benefits_html, unsafe_allow_html=True)

            st.markdown("---")
            
            
            # What-If Analysis section (Moved into col1)
            with st.expander("🤔 Interactive What-If Analysis", expanded=False):
                st.markdown("Use the sliders to see how your savings change with different transaction volumes.")
                col_wa1, col_wa2 = st.columns(2)
                with col_wa1:
                    max_int = int(user_data['int_count'] * 2.5) + 10
                    what_if_int_count = st.slider("International Transfers", 0, max_int, user_data['int_count'], key="wa_int")

                    max_pdc = int(user_data['pdc_count'] * 2.5) + 10
                    what_if_pdc_count = st.slider("PDCs Processed", 0, max_pdc, user_data['pdc_count'], key="wa_pdc")

                with col_wa2:
                    max_fx = int(user_data['fx_amount'] * 2.5) + 5000
                    what_if_fx_amount = st.slider("FX Amount (USD)", 0.0, float(max_fx), float(user_data['fx_amount']), key="wa_fx")

                    max_dom = int(user_data['dom_count'] * 2.5) + 10
                    what_if_dom_count = st.slider("Domestic Transfers", 0, max_dom, user_data['dom_count'], key="wa_dom")

                # Prepare new inputs for re-calculation
                what_if_tx = tx.copy()
                what_if_tx['international'] = what_if_int_count
                what_if_tx['domestic'] = what_if_dom_count
                what_if_tx['pdc'] = what_if_pdc_count

                # Re-run the analysis with the slider values
                best_wi, savings_wi, results_wi = suggest_best_package(
                    what_if_tx, tx_cost, what_if_fx_amount, user_data["fx_direction"],
                    user_data["client_fx_rate"], user_data["wps_cost"], user_data["other_costs_input"]
                )

                st.markdown("---")
                if best_wi:
                    st.success(f"With these new values, **{best_wi}** would be the best package, saving you **{round(savings_wi):,} AED**.")
                else:
                    st.warning("With these values, no package offers savings over the client's current costs.")
                
                # What-If Cost Breakdown Chart (Styled to match main chart)
                if best_wi:
                    # 1. Gather true costs
                    all_costs_wi = [(name, res["true_total_cost"]) for name, res in results_wi.items()]
                    no_pkg_item_wi = next(item for item in all_costs_wi if item[0] == "Without Package")
                    package_items_wi = [item for item in all_costs_wi if item[0] != "Without Package"]
                    package_items_wi.sort(key=lambda x: x[1])
                    sorted_categories_wi = [no_pkg_item_wi[0]] + [name for name, _ in package_items_wi]
                    sorted_true_costs_wi = [no_pkg_item_wi[1]] + [cost for _, cost in package_items_wi]

                    # 2. Find best package details
                    best_pkg_true_cost_wi = results_wi[best_wi]["true_total_cost"]
                    # This is the absolute FX cost of the BEST what-if package, used as a baseline
                    best_pkg_fx_cost_baseline_wi = results_wi[best_wi]["breakdown"].get("Absolute FX Cost", 0)

                    # 3. Calculate bar values
                    bar_values_wi = []
                    for true_cost in sorted_true_costs_wi:
                        # The display cost is the true total cost minus the absolute FX cost of the BEST package.
                        bar_values_wi.append(true_cost - best_pkg_fx_cost_baseline_wi)

                    # 4. Assign colors
                    best_idx_wi = sorted_categories_wi.index(best_wi)
                    n_pkgs_wi = len(sorted_categories_wi) - 1
                    bar_colors_wi = ["#808080"]
                    for i in range(n_pkgs_wi):
                        t = i / max(n_pkgs_wi - 1, 1)
                        rgb_wi = interpolate_color(rgb_best, rgb_worst, t)
                        bar_colors_wi.append('#%02x%02x%02x' % rgb_wi)
                    if best_idx_wi > 0: bar_colors_wi[best_idx_wi] = "#CD2026"

                    # 5. Create chart with gap and formatting
                    x_positions_wi = [0] + [i + 0.5 for i in range(1, len(sorted_categories_wi))]
                    formatted_labels_wi = [format_label(label) for label in sorted_categories_wi]

                    df_cost_wi = pd.DataFrame({
                        "x_pos": x_positions_wi,
                        "Category": sorted_categories_wi,
                        "Cost (AED)": bar_values_wi
                    })

                    fig_cost_wi = px.bar(df_cost_wi, x="x_pos", y="Cost (AED)",
                                         color="Category", color_discrete_sequence=bar_colors_wi)
                    # We will add text via annotations, so remove it from here
                    fig_cost_wi.update_traces(textposition='outside')
                    fig_cost_wi.update_layout(
                        title="💰 What-If: Total Cost by Option",
                        showlegend=False,
                        xaxis_title=None,
                        plot_bgcolor='white',
                        margin=dict(l=40, r=40, t=40, b=20),
                        height=520,
                        bargap=0.2
                    )
                    fig_cost_wi.update_yaxes(showgrid=False, showticklabels=False, title_text=None, zeroline=False)
                    fig_cost_wi.update_xaxes(
                        showgrid=False, tickangle=0, tickvals=x_positions_wi,
                        ticktext=formatted_labels_wi, tickfont=dict(size=16)
                    )
                    
                    # Add styled arrow for What-If chart
                    idx_no_pkg_wi = 0
                    idx_best_wi = sorted_categories_wi.index(best_wi)

                    x0_pos_wi = x_positions_wi[idx_no_pkg_wi]
                    x1_pos_wi = x_positions_wi[idx_best_wi]

                    y0_wi = bar_values_wi[idx_no_pkg_wi]
                    y1_wi = bar_values_wi[idx_best_wi]
                    savings_amt_wi = results_wi["Without Package"]["true_total_cost"] - best_pkg_true_cost_wi

                    fig_cost_wi.add_annotation(
                        x=x1_pos_wi, y=y1_wi, ax=x0_pos_wi, ay=y0_wi,
                        xref="x", yref="y", axref="x", ayref="y", text="",
                        showarrow=True, arrowhead=3, arrowsize=1.5,
                        arrowwidth=8, arrowcolor="#240F8C", opacity=1
                    )

                    savings_label_wi = f"<span style='font-size:15px;font-weight:bold;color:#228B22;line-height:1.1;'>*savings<br>{int(savings_amt_wi):,} AED</span>"
                    x_sav_pos_wi = (x0_pos_wi + x1_pos_wi) / 2
                    y_sav_wi = max(y0_wi, y1_wi) + 0.12 * max(bar_values_wi)
                    fig_cost_wi.add_annotation(
                        x=x_sav_pos_wi, y=y_sav_wi, text=savings_label_wi,
                        showarrow=False, font=dict(size=15, color="#228B22", family="Arial Black"),
                        align="center", bordercolor=None, borderwidth=0,
                        borderpad=0, bgcolor=None, xanchor="center", yanchor="bottom"
                    )

                    # Add bar labels as annotations to ensure they are drawn on top of the arrow
                    for i, row in df_cost_wi.iterrows():
                        fig_cost_wi.add_annotation(
                            x=row['x_pos'],
                            y=row['Cost (AED)'],
                            text=f"{row['Cost (AED)']:,.0f}",
                            showarrow=False,
                            yshift=10,
                            font=dict(size=20, color="black"),
                            xanchor="center",
                        )
                    st.plotly_chart(fig_cost_wi, width="stretch")

                    # What-If Savings Breakdown (Styled to match main chart)
                    st.markdown(f"### 🏅 What-If Savings Breakdown for {best_wi}")
                    no_pkg_breakdown_wi = results_wi["Without Package"]["breakdown"]
                    best_pkg_breakdown_wi = results_wi[best_wi]["breakdown"]

                    savings_data_wi = [
                        {"Component": "Other", "Savings (AED)": (no_pkg_breakdown_wi.get("Other Costs (User Input)", 0) - best_pkg_breakdown_wi.get("Other Costs (User Input)", 0)) + (no_pkg_breakdown_wi.get("WPS/CST Cost", 0) - best_pkg_breakdown_wi.get("WPS/CST Cost", 0))},
                        {"Component": "FCY", "Savings (AED)": no_pkg_breakdown_wi.get("Inward Fcy Remittance Cost", 0) - best_pkg_breakdown_wi.get("Inward Fcy Remittance Cost", 0)},
                        {"Component": "PDC", "Savings (AED)": no_pkg_breakdown_wi.get("Pdc Cost", 0) - best_pkg_breakdown_wi.get("Pdc Cost", 0)},
                        {"Component": "Chq", "Savings (AED)": no_pkg_breakdown_wi.get("Cheque Transactions Cost", 0) - best_pkg_breakdown_wi.get("Cheque Transactions Cost", 0)},
                        {"Component": "Dom", "Savings (AED)": no_pkg_breakdown_wi.get("Domestic Transactions Cost", 0) - best_pkg_breakdown_wi.get("Domestic Transactions Cost", 0)},
                        {"Component": "Intl", "Savings (AED)": no_pkg_breakdown_wi.get("International Transactions Cost", 0) - best_pkg_breakdown_wi.get("International Transactions Cost", 0)},
                        {"Component": "FX", "Savings (AED)": no_pkg_breakdown_wi.get("Absolute FX Cost", 0) - best_pkg_breakdown_wi.get("Absolute FX Cost", 0)}
                    ]

                    df_savings_wi = pd.DataFrame(savings_data_wi)

                    # Grayscale logic
                    positive_savings_wi = df_savings_wi[df_savings_wi['Savings (AED)'] > 0].sort_values('Savings (AED)', ascending=False)
                    gray_palette_wi = ['#666666', '#808080', '#827F7F', '#A9A9A9', '#C0C0C0', '#D3D3D3']
                    color_map_wi = {component: gray_palette_wi[min(i, len(gray_palette_wi) - 1)] for i, component in enumerate(positive_savings_wi['Component'])}

                    bar_colors_savings_wi = []
                    for _, row in df_savings_wi.iterrows():
                        if row['Savings (AED)'] > 0:
                            bar_colors_savings_wi.append(color_map_wi.get(row['Component'], '#CCCCCC'))
                        elif row['Savings (AED)'] < 0:
                            bar_colors_savings_wi.append('#e4002b')
                        else:
                            bar_colors_savings_wi.append('#F0F0F0')

                    fig_savings_wi = go.Figure()
                    fig_savings_wi.add_trace(go.Bar(
                        y=df_savings_wi["Component"],
                        x=df_savings_wi["Savings (AED)"],
                        orientation='h',
                        marker=dict(color=bar_colors_savings_wi),
                        text=df_savings_wi["Savings (AED)"].apply(lambda x: f"{x:,.0f} AED"),
                        textposition='auto'
                    ))
                    fig_savings_wi.update_traces(textangle=0, insidetextanchor='end', textfont=dict(size=14, color='black'))
                    fig_savings_wi.update_layout(
                        title="📊 What-If: Savings Breakdown",
                        plot_bgcolor='white',
                        xaxis_title="Savings (AED)",
                        yaxis_title=None,
                        margin=dict(l=50, r=50, t=50, b=50),
                        height=500,
                        showlegend=False,
                        yaxis=dict(categoryorder='array', categoryarray=[r['Component'] for r in reversed(savings_data_wi)])
                    )
                    fig_savings_wi.update_xaxes(showgrid=False, zeroline=False)
                    fig_savings_wi.update_yaxes(showgrid=False, zeroline=False)
                    st.plotly_chart(fig_savings_wi, width="stretch")
        
        with col2:
            # --- Pre-calculate all cost components ---
            no_pkg_breakdown = results["Without Package"]["breakdown"]
            no_pkg_true_cost = results["Without Package"]["true_total_cost"]
            no_pkg_fx_cost_val = no_pkg_breakdown.get('Absolute FX Cost', 0)

            best_pkg_breakdown = results[best]["breakdown"]
            best_pkg_true_cost = results[best]["true_total_cost"]
            best_pkg_fx_cost_val = best_pkg_breakdown.get('Absolute FX Cost', 0)

            # Marginal FX cost is the EXTRA cost incurred by the client for NOT having the package rate
            fx_marginal_cost = no_pkg_fx_cost_val - best_pkg_fx_cost_val

            # --- Client's Current Setup Card (Shows the marginal FX cost) ---
            # Total for display = (All non-FX costs) + (Marginal FX cost)
            display_total_no_pkg = (no_pkg_true_cost - no_pkg_fx_cost_val) + fx_marginal_cost

            no_pkg_paid_lines_list = []
            no_pkg_txn_items = {
                "International": "International Transactions Cost", "Domestic": "Domestic Transactions Cost",
                "Cheques": "Cheque Transactions Cost", "PDCs": "Pdc Cost", "Inward FCY": "Inward Fcy Remittance Cost"
            }
            for name, key in no_pkg_txn_items.items():
                cost = no_pkg_breakdown.get(key, 0)
                no_pkg_paid_lines_list.append(f"&emsp;├─ {name.ljust(15)} = {round(cost):,} AED")
            
            if no_pkg_paid_lines_list:
                no_pkg_paid_lines_list = [line.replace(' ', '&nbsp;') for line in no_pkg_paid_lines_list]
                last_line = no_pkg_paid_lines_list[-1].replace('├─', '└─')
                no_pkg_paid_lines = "<br>".join(no_pkg_paid_lines_list[:-1] + [last_line])
            else:
                no_pkg_paid_lines = "&emsp;└─ No transaction costs."

            st.markdown(f'''
<div style="background: #f8fafd; border: 2px solid #808080; border-radius: 18px; padding: 32px 36px 28px 36px; margin-bottom: 32px; font-family: 'Consolas', 'Menlo', 'Monaco', 'monospace'; font-size: 1.15rem; color: #222; box-shadow: 0 4px 24px 0 rgba(128,128,128,0.07); max-width: 700px;">
    <div style="font-size:1.3rem; font-weight:700; color:#e4002b; margin-bottom:18px; letter-spacing:1px;">
        🧾 Client's current charges with competitor bank
    </div>
    <div style="margin-bottom:18px;">
        <span style="color:#1f77b4; font-size:1.1em;">📁 Transaction Costs</span><br>
        {no_pkg_paid_lines}
    </div>
    <div style="margin-bottom:18px;">
        <span style="color:#228B22; font-size:1.1em;">🪙 FX Impact (vs Package Rate)</span><br>
        &emsp;└─ Additional Cost = <b style='color:#333;'>{round(fx_marginal_cost):,} AED</b>
    </div>
    <div style="margin-bottom:18px;">
        <span style="color:#ff9900; font-size:1.1em;">🛠️ Other Costs</span><br>
        &emsp;├─ WPS / CST&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;=&nbsp;{round(no_pkg_breakdown.get("WPS/CST Cost", 0)):,} AED<br>
        &emsp;└─ Miscellaneous&nbsp;&nbsp;=&nbsp;{round(no_pkg_breakdown.get("Other Costs (User Input)", 0)):,} AED 
    </div>
    <div style="margin-top:18px; font-size:1.18em; color:#fff; background:#808080; display:inline-block; padding:8px 18px; border-radius:8px; font-weight:700;">
        ❌ Total Fees & Marginal Costs: {round(display_total_no_pkg):,} AED
    </div>
</div>
''', unsafe_allow_html=True)
            
            # --- Best Package Card (Shows the baseline) ---
            # Total for display = True cost of package - its own absolute FX cost (since FX is now the baseline)
            display_total_best_pkg = best_pkg_true_cost - best_pkg_fx_cost_val

            paid_lines_list = []
            package_fee = best_pkg_breakdown.get("Package Cost", 0)
            paid_lines_list.append(f"&emsp;├─ {'Package Fee'.ljust(15)} = {round(package_fee):,} AED")
            
            paid_txn_items = {
                "International": "International Transactions Cost", "Domestic": "Domestic Transactions Cost",
                "Cheques": "Cheque Transactions Cost", "PDCs": "Pdc Cost", "Inward FCY": "Inward Fcy Remittance Cost"
            }
            for name, key in paid_txn_items.items():
                cost = best_pkg_breakdown.get(key, 0)
                paid_lines_list.append(f"&emsp;├─ {name.ljust(15)} = {round(cost):,} AED")
            
            if len(paid_lines_list) > 0:
                paid_lines_list = [line.replace(' ', '&nbsp;') for line in paid_lines_list]
                last_line = paid_lines_list[-1].replace('├─', '└─')
                paid_lines = "<br>".join(paid_lines_list[:-1] + [last_line])
            else:
                paid_lines = "&emsp;└─ No additional costs."

            st.markdown(f'''
<div style="background: #f8fafd; border: 2px solid #228B22; border-radius: 18px; padding: 32px 36px 28px 36px; margin-bottom: 32px; font-family: 'Consolas', 'Menlo', 'Monaco', 'monospace'; font-size: 1.15rem; color: #222; box-shadow: 0 4px 24px 0 rgba(31,119,180,0.07); max-width: 700px;">
    <div style="font-size:1.3rem; font-weight:700; color:#1f77b4; margin-bottom:18px; letter-spacing:1px;">
        📦 Cost Breakdown with ADCB's {best}
    </div>
    <div style="margin-bottom:18px;">
        <span style="color:#1f77b4; font-size:1.1em;">📁 Transaction & Fee Costs</span><br>
        {paid_lines}
    </div>
    <div style="margin-bottom:18px;">
        <span style="color:#228B22; font-size:1.1em;">🪙 FX Cost</span><br>
        &emsp;└─ Using Package Rate = <b style='color:#333;'>0 AED (Baseline)</b>
    </div>
    <div style="margin-bottom:18px;">
        <span style="color:#ff9900; font-size:1.1em;">🛠️ Other Costs</span><br>
        &emsp;├─ WPS / CST&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;=&nbsp;0 AED (included in package)<br>
        &emsp;└─ Miscellaneous&nbsp;&nbsp;=&nbsp;{round(best_pkg_breakdown.get("Other Costs (User Input)", 0)):,} AED 
    </div>
    <div style="margin-top:18px; font-size:1.18em; color:#fff; background:#228B22; display:inline-block; padding:8px 18px; border-radius:8px; font-weight:700;">
        ✅ Total Fees & Marginal Costs: {round(display_total_best_pkg):,} AED
    </div>
</div>
''', unsafe_allow_html=True)

    # Export Options
    st.markdown("### 📤 Export Results")
    col1, col2 = st.columns(2)
    with col1:
        csv = export_to_csv(results, user_data, best, savings, results["Without Package"]["true_total_cost"])
        st.download_button("📥 Export to CSV", data=csv, file_name="package_comparison.csv", mime="text/csv")
    with col2:
        pdf_bytes = export_to_pdf(results, user_data, best, savings, results["Without Package"]["true_total_cost"], results_data.get("narrative_summary", ""))
        st.download_button("📄 Export to PDF", data=pdf_bytes, file_name="package_comparison.pdf", mime="application/pdf")
    
    # --- ACTIVESAVER INTEGRATION (Only appears after export, and only if user clicks) ---
    st.markdown("---")
    
    # Create 2-column layout for ActiveSaver section
    col_as_main1, col_as_main2 = st.columns([7, 3])
    
    with col_as_main1:
        # --- ACTIVESAVER INTEGRATION ---
        st.markdown("### 💰 Want to Save Even More on Your Package Fee?")
        st.markdown("**ADCB ActiveSaver** is a high-interest operational CASA account that can offset your package costs through interest earnings!")
        
        # Ask if user wants to explore ActiveSaver
        col_as1, col_as2 = st.columns(2)
        with col_as1:
            if st.button("🚀 Explore ActiveSaver Benefits", key="explore_activesaver"):
                st.session_state.show_activesaver = True
                st.rerun()
        with col_as2:
            if st.button("❌ Skip ActiveSaver", key="skip_activesaver"):
                st.session_state.show_activesaver = False
                st.rerun()
        
        # Show ActiveSaver calculator if user wants to explore
        if st.session_state.get("show_activesaver", False):
            st.markdown("---")
            st.markdown("### 🧮 ActiveSaver Interest Calculator")
            st.markdown("Enter your expected balance patterns to see how much interest you can earn:")

            # --- Currency selector ---
            if "activesaver_currency" not in st.session_state:
                st.session_state.activesaver_currency = "AED"
            
            # Replace selectbox with buttons
            col_cur1, col_cur2 = st.columns(2)
            with col_cur1:
                if st.button("AED", key="as_btn_aed"):
                    st.session_state.activesaver_currency = "AED"
                    st.rerun()
            with col_cur2:
                if st.button("USD", key="as_btn_usd"):
                    st.session_state.activesaver_currency = "USD"
                    st.rerun()
            
            st.markdown(f"**Selected Currency:** <span style='font-weight:bold;'>{st.session_state.activesaver_currency}</span>", unsafe_allow_html=True)
            currency = st.session_state.activesaver_currency

            
            if currency == "AED":
                activesaver_slabs = AED_SLABS
                currency_symbol = "AED"
            else:
                activesaver_slabs = USD_SLABS
                currency_symbol = "USD"

            # Get package cost for analysis
            package_cost = results[best]["breakdown"].get("Package Cost", 0)

            # --- Dynamic Balance/Days Rows ---
            if "activesaver_rows" not in st.session_state or st.session_state.get("activesaver_currency_last", None) != currency:
                st.session_state.activesaver_rows = [{"balance": 100000.0, "days": 30}]
                st.session_state.activesaver_currency_last = currency
            rows = st.session_state.activesaver_rows

            st.markdown(f"#### 📊 Balance Pattern Input ({currency_symbol})")
            st.markdown("Add as many rows as you want. Each row is a balance and the number of days it is maintained.")

            # Render rows
            remove_indices = []
            for i, row in enumerate(rows):
                c1, c2, c3 = st.columns([4, 4, 1])
                with c1:
                    rows[i]["balance"] = st.number_input(f"Balance ({currency_symbol}) #{i+1}", min_value=0.0, value=row["balance"], step=1000.0, key=f"as_balance_{i}_{currency_symbol}")
                with c2:
                    rows[i]["days"] = st.number_input(f"Days #{i+1}", min_value=1, value=row["days"], step=1, key=f"as_days_{i}_{currency_symbol}")
                with c3:
                    if len(rows) > 1:
                        if st.button("🗑️", key=f"remove_row_{i}_{currency_symbol}"):
                            remove_indices.append(i)
            # Remove rows marked for deletion
            for idx in sorted(remove_indices, reverse=True):
                del rows[idx]
            # Add row button
            if st.button("➕ Add Row", key=f"add_activesaver_row_{currency_symbol}"):
                rows.append({"balance": 0.0, "days": 1})

            # Calculate ActiveSaver benefits
            if st.button(f"💡 Calculate ActiveSaver Benefits ({currency_symbol})", key=f"calc_activesaver_{currency_symbol}"):
                # Build balance-days dictionary
                balance_days_dict = {}
                for row in rows:
                    bal = float(row["balance"])
                    days = int(row["days"])
                    if bal > 0 and days > 0:
                        balance_days_dict[bal] = days
                if not balance_days_dict:
                    st.error(f"Please enter at least one valid balance and days row for {currency_symbol}.")
                else:
                    # Use the correct slabs for calculation
                    calculator = SavingsInterestCalculator(activesaver_slabs)
                    interest_result = calculator.calculate_interest_simple(balance_days_dict)
                    period_interest = interest_result['total_interest']
                    net_cost = max(0, package_cost - period_interest)
                    offset_pct = (period_interest / package_cost * 100) if package_cost > 0 else 0
                    st.markdown("---")
                    st.markdown(f"### 🎯 ActiveSaver Analysis Results ({currency_symbol})")
                    st.markdown(f"**Interest for Entered Period:** <span style='font-size:1.3em;font-weight:bold;'>{period_interest:,.0f} {currency_symbol}</span>", unsafe_allow_html=True)
                    st.markdown(f"**Net Package Cost after Interest:** <span style='font-size:1.3em;font-weight:bold;color:#28a745;'>{net_cost:,.0f} {currency_symbol}</span>", unsafe_allow_html=True)
                    st.markdown(f"**Offset Percentage:** <span style='font-size:1.1em;font-weight:bold;'>{offset_pct:.1f}%</span>", unsafe_allow_html=True)
                    st.markdown("---")
                    # --- Enhanced Simulation & Suggestions ---
                    st.markdown(f"### 🔍 Simulation & Suggestions ({currency_symbol})")
                    # 1. What balance for what days is needed to fully offset the package fee?
                    # Try to find the minimum balance (in the highest tier) for the max days entered that would cover the package fee
                    max_days = max([r['days'] for r in rows])
                    found = False
                    # Start 70% width container
                    st.markdown('<div style="width:70%;margin:auto;">', unsafe_allow_html=True)
                    for slab in reversed(activesaver_slabs):
                        # Try to solve for balance: balance * rate * (days/365) = package_cost
                        rate = slab['interest_rate'] / 100
                        needed_balance = package_cost / (rate * (max_days/365)) if rate > 0 else None
                        if needed_balance is not None and (slab['min_balance'] <= needed_balance < (slab['max_balance'] if slab['max_balance'] else float('inf'))):
                            st.markdown(f"To fully offset your package fee, you need to keep at least <b>{needed_balance:,.0f} {currency_symbol}</b> in the <b>{slab['description']}</b> tier for <b>{max_days}</b> days.", unsafe_allow_html=True)
                            found = True
                            break
                    if not found:
                        st.warning(f"With the current tiers, it's not possible to fully offset the package fee for {max_days} days.")

    # col_as_main2 is intentionally left empty

# Close the main-content-80 container
st.markdown("</div>", unsafe_allow_html=True)
