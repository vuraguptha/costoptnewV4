import streamlit as st
from utils import play_text_as_speech
from package_analysis import generate_analysis
# Add these new functions for interactive chat flow
def init_chat_state():
    if "chat_stage" not in st.session_state:
        st.session_state.chat_stage = "welcome"
    if "transaction_data" not in st.session_state:
        st.session_state.transaction_data = {
            "domestic": {"count": 0, "cost": 0.0},
            "international": {"count": 0, "cost": 0.0},
            "cheque": {"count": 0, "cost": 0.0},
            "pdc": {"count": 0, "cost": 0.0},
            "inward_fcy_remittance": {"count": 0, "cost": 0.0},
            "fx": {"amount": 0.0, "direction": "Buy USD", "rate": 3.67},
            "wps": {"enabled": False, "cost": 0.0},
            "other_costs_input": 0.0
        }
    if "messages" not in st.session_state:
        st.session_state.messages = []

def process_user_response(response):
    stage = st.session_state.chat_stage
    data = st.session_state.transaction_data
    
    # Text to be spoken for the current stage
    text_for_speech = ""

    if stage == "welcome":
        text_for_speech = "Hi! I'm your AI Banking Assistant. Is your client currently making domestic transfers as part of their recurring business expenditures??"
        st.write(f"👋 {text_for_speech}")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Yes", key="yes_domestic"):
                st.session_state.chat_stage = "domestic_count"
                st.rerun()
        with col2:
            if st.button("No", key="no_domestic"):
                data["domestic"]["count"] = 0
                data["domestic"]["cost"] = 0
                st.session_state.chat_stage = "international_ask"
                st.rerun()
                
    elif stage == "domestic_count":
        text_for_speech = "Enter number of domestic transfers:"
        st.write(text_for_speech)
        count = st.number_input("Count", min_value=0, step=1, key="domestic_count")
        if st.button("Continue", key="submit_domestic_count"):
            data["domestic"]["count"] = count
            st.session_state.chat_stage = "domestic_cost"
            st.rerun()
            
    elif stage == "domestic_cost":
        text_for_speech = "Enter cost per domestic transfer, in AED:"
        st.write(text_for_speech)
        cost = st.number_input("Cost", min_value=0.0, step=0.1, key="domestic_cost")
        if st.button("Continue", key="submit_domestic_cost"):
            data["domestic"]["cost"] = cost
            st.session_state.chat_stage = "international_ask"
            st.rerun()
            
    elif stage == "international_ask":
        text_for_speech = "Do you make international transfers?"
        st.write(text_for_speech)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Yes", key="yes_international"):
                st.session_state.chat_stage = "international_count"
                st.rerun()
        with col2:
            if st.button("No", key="no_international"):
                data["international"]["count"] = 0
                data["international"]["cost"] = 0
                st.session_state.chat_stage = "cheque_ask"
                st.rerun()
                
    elif stage == "international_count":
        text_for_speech = "Enter number of international transfers:"
        st.write(text_for_speech)
        count = st.number_input("Count", min_value=0, step=1, key="international_count")
        if st.button("Continue", key="submit_international_count"):
            data["international"]["count"] = count
            st.session_state.chat_stage = "international_cost"
            st.rerun()
            
    elif stage == "international_cost":
        text_for_speech = "Enter cost per international transfer, in AED:"
        st.write(text_for_speech)
        cost = st.number_input("Cost", min_value=0.0, step=0.1, key="international_cost")
        if st.button("Continue", key="submit_international_cost"):
            data["international"]["cost"] = cost
            st.session_state.chat_stage = "cheque_ask"
            st.rerun()
            
    elif stage == "cheque_ask":
        text_for_speech = "Do you process cheques?"
        st.write(text_for_speech)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Yes", key="yes_cheque"):
                st.session_state.chat_stage = "cheque_count"
                st.rerun()
        with col2:
            if st.button("No", key="no_cheque"):
                data["cheque"]["count"] = 0
                data["cheque"]["cost"] = 0
                st.session_state.chat_stage = "fx_ask"
                st.rerun()
                
    elif stage == "cheque_count":
        text_for_speech = "Enter number of cheques:"
        st.write(text_for_speech)
        count = st.number_input("Count", min_value=0, step=1, key="cheque_count")
        if st.button("Continue", key="submit_cheque_count"):
            data["cheque"]["count"] = count
            st.session_state.chat_stage = "cheque_cost"
            st.rerun()
            
    elif stage == "cheque_cost":
        text_for_speech = "Enter cost per cheque, in AED:"
        st.write(text_for_speech)
        cost = st.number_input("Cost", min_value=0.0, step=0.1, key="cheque_cost")
        if st.button("Continue", key="submit_cheque_cost"):
            data["cheque"]["cost"] = cost
            st.session_state.chat_stage = "fx_ask"
            st.rerun()
            
    elif stage == "fx_ask":
        text_for_speech = "Do you need foreign exchange?"
        st.write(text_for_speech)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Yes", key="yes_fx"):
                st.session_state.chat_stage = "fx_amount"
                st.rerun()
        with col2:
            if st.button("No", key="no_fx"):
                data["fx"]["amount"] = 0
                data["fx"]["rate"] = 3.63
                st.session_state.chat_stage = "wps_ask"
                st.rerun()
                
    elif stage == "fx_amount":
        text_for_speech = "Enter FX amount in USD:"
        st.write(text_for_speech)
        amount = st.number_input("Amount", min_value=0.0, step=100.0, key="fx_amount")
        if st.button("Continue", key="submit_fx_amount"):
            data["fx"]["amount"] = amount
            st.session_state.chat_stage = "fx_direction"
            st.rerun()
            
    elif stage == "fx_direction":
        text_for_speech = "Are you buying or selling USD?"
        st.write(text_for_speech)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Buy USD", key="buy_usd"):
                data["fx"]["direction"] = "Buy USD"
                st.session_state.chat_stage = "fx_rate"
                st.rerun()
        with col2:
            if st.button("Sell USD", key="sell_usd"):
                data["fx"]["direction"] = "Sell USD"
                st.session_state.chat_stage = "fx_rate"
                st.rerun()
                
    elif stage == "fx_rate":
        text_for_speech = f"Enter your {data['fx']['direction']} rate (AED/USD):"
        st.write(text_for_speech)
        rate = st.number_input("Rate", min_value=0.0, step=0.01, key="fx_rate")
        if st.button("Continue", key="submit_fx_rate"):
            data["fx"]["rate"] = rate
            st.session_state.chat_stage = "wps_ask"
            st.rerun()
                
    elif stage == "wps_ask":
        text_for_speech = "Do you use WPS or CST (Wages Protection System or Corporate Self Transfer)?"
        st.write(text_for_speech)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Yes", key="yes_wps"):
                st.session_state.chat_stage = "wps_cost"
                st.rerun()
        with col2:
            if st.button("No", key="no_wps"):
                data["wps"]["enabled"] = False
                data["wps"]["cost"] = 0
                st.session_state.chat_stage = "pdc_ask"
                st.rerun()
                
    elif stage == "wps_cost":
        text_for_speech = "Enter monthly WPS or CST cost, in AED:"
        st.write(text_for_speech)
        cost = st.number_input("WPS/CST Cost", min_value=0.0, step=10.0, key="wps_cost_ai")
        if st.button("Continue", key="submit_wps_cost_ai"):
            data["wps"]["enabled"] = True
            data["wps"]["cost"] = cost
            st.session_state.chat_stage = "pdc_ask"
            st.rerun()

    # New stages for PDC
    elif stage == "pdc_ask":
        text_for_speech = "Do you process Post-Dated Cheques (PDCs)?"
        st.write(text_for_speech)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Yes", key="yes_pdc"):
                st.session_state.chat_stage = "pdc_count"
                st.rerun()
        with col2:
            if st.button("No", key="no_pdc"):
                data["pdc"]["count"] = 0
                data["pdc"]["cost"] = 0
                st.session_state.chat_stage = "inward_fcy_ask"
                st.rerun()

    elif stage == "pdc_count":
        text_for_speech = "Enter number of PDCs processed monthly:"
        st.write(text_for_speech)
        count = st.number_input("PDC Count", min_value=0, step=1, key="pdc_count_ai")
        if st.button("Continue", key="submit_pdc_count_ai"):
            data["pdc"]["count"] = count
            st.session_state.chat_stage = "pdc_cost"
            st.rerun()

    elif stage == "pdc_cost":
        text_for_speech = "Enter cost per PDC, in AED:"
        st.write(text_for_speech)
        cost = st.number_input("Cost per PDC", min_value=0.0, step=0.1, key="pdc_cost_ai_item")
        if st.button("Continue", key="submit_pdc_cost_ai_item"):
            data["pdc"]["cost"] = cost
            st.session_state.chat_stage = "inward_fcy_ask"
            st.rerun()

    # New stages for Inward FCY Remittance
    elif stage == "inward_fcy_ask":
        text_for_speech = "Do you receive Inward FCY Remittances?"
        st.write(text_for_speech)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Yes", key="yes_inward_fcy"):
                st.session_state.chat_stage = "inward_fcy_count"
                st.rerun()
        with col2:
            if st.button("No", key="no_inward_fcy"):
                data["inward_fcy_remittance"]["count"] = 0
                data["inward_fcy_remittance"]["cost"] = 0
                st.session_state.chat_stage = "other_costs_ask"
                st.rerun()
    
    elif stage == "inward_fcy_count":
        text_for_speech = "Enter number of Inward FCY Remittances monthly:"
        st.write(text_for_speech)
        count = st.number_input("Inward FCY Remittance Count", min_value=0, step=1, key="inward_fcy_count_ai")
        if st.button("Continue", key="submit_inward_fcy_count_ai"):
            data["inward_fcy_remittance"]["count"] = count
            st.session_state.chat_stage = "inward_fcy_cost"
            st.rerun()

    elif stage == "inward_fcy_cost":
        text_for_speech = "Enter cost per Inward FCY Remittance, in AED:"
        st.write(text_for_speech)
        cost = st.number_input("Cost per Inward FCY Remittance", min_value=0.0, step=0.1, key="inward_fcy_cost_ai_item")
        if st.button("Continue", key="submit_inward_fcy_cost_ai_item"):
            data["inward_fcy_remittance"]["cost"] = cost
            st.session_state.chat_stage = "other_costs_ask"
            st.rerun()

    # New stage for Other Costs
    elif stage == "other_costs_ask":
        text_for_speech = "Do you have any other monthly costs such as cheque submission, courier, or miscellaneous fees?"
        st.write(text_for_speech)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Yes", key="yes_other_costs"):
                st.session_state.chat_stage = "other_costs_input"
                st.rerun()
        with col2:
            if st.button("No", key="no_other_costs"):
                data["other_costs_input"] = 0.0
                analysis_successful = generate_analysis()
                st.session_state.chat_stage = "analysis" if analysis_successful else "no_savings_found"
                st.rerun()

    elif stage == "other_costs_input":
        text_for_speech = "Enter total of these other monthly costs, in AED:"
        st.write(text_for_speech)
        other_total_cost = st.number_input("Total Other Costs", min_value=0.0, step=1.0, key="other_costs_input_ai")
        if st.button("View Analysis", key="submit_other_costs_ai"):
            with st.spinner("Analyzing your data... This may take a moment."):
                data["other_costs_input"] = other_total_cost
                analysis_successful = generate_analysis()
                st.session_state.chat_stage = "analysis" if analysis_successful else "no_savings_found"
            st.rerun()

    elif stage == "analysis":
        text_for_speech = "Analysis Complete! View the results in the main panel."
        st.success(text_for_speech)
        # Optionally, add a button here to start a new AI chat analysis
        if st.button("Start New AI Analysis", key="new_ai_analysis_from_success"):
            init_chat_state() # Reset AI chat state
            st.session_state.submitted = False # Ensure manual results are not shown
            st.session_state.show_welcome = True # Show welcome screen
            if 'analysis_results' in st.session_state: # Clean up previous results
                del st.session_state.analysis_results
            st.rerun()

    elif stage == "no_savings_found":
        text_for_speech = "Based on your inputs, no package offers savings over not using one. You can adjust your inputs or try the Manual mode."
        st.warning("Based on your inputs, no package offers savings over not using one.")
        st.info("You can adjust your inputs or try the Manual mode.")
        if st.button("Try Again with AI", key="try_again_ai"):
            init_chat_state() # Reset AI chat state
            st.session_state.submitted = False
            st.session_state.show_welcome = True
            if 'analysis_results' in st.session_state:
                del st.session_state.analysis_results
            st.rerun()
    
    # Play the speech if text_for_speech is set
    if text_for_speech:
        play_text_as_speech(text_for_speech)

    # Show current progress
    st.markdown("---")
    st.markdown("### Current Information:")
    if data["domestic"]["count"] > 0:
        st.write(f"✓ Domestic: {data['domestic']['count']} transfers at {data['domestic']['cost']} AED")
    if data["international"]["count"] > 0:
        st.write(f"✓ International: {data['international']['count']} transfers at {data['international']['cost']} AED")
    if data["cheque"]["count"] > 0:
        st.write(f"✓ Cheques: {data['cheque']['count']} at {data['cheque']['cost']} AED")
    if data["fx"]["amount"] > 0:
        st.write(f"✓ FX: {data['fx']['amount']} USD ({data['fx']['direction']}) at rate {data['fx']['rate']}")
    if data["wps"]["enabled"]:
        st.write(f"✓ WPS/CST Cost: {data['wps']['cost']} AED")
    if data["pdc"]["count"] > 0:
        st.write(f"✓ PDC: {data['pdc']['count']} at {data['pdc']['cost']} AED each")
    if data["inward_fcy_remittance"]["count"] > 0:
        st.write(f"✓ Inward FCY: {data['inward_fcy_remittance']['count']} at {data['inward_fcy_remittance']['cost']} AED each")
    if data["other_costs_input"] > 0:
        st.write(f"✓ Other Costs: {data['other_costs_input']} AED")
