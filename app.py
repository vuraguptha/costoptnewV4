import streamlit as st
import pandas as pd
import plotly.express as px
import openai
import os
import json
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import time
from gtts import gTTS # Import gTTS
import base64 # Added for image encoding
import numpy as np
import plotly.graph_objects as go


# Set OpenAI API Key
openai.api_key =  st.secrets["OPENAI_API_KEY"]

def img_to_base64(image_path):
    """Convert image to base64"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        # This error is handled where the function is called, to avoid stopping the app
        return None

# -------------------- TTS HELPER FUNCTION --------------------
def play_text_as_speech(text_to_speak):
    """Generates speech from text and plays it using st.audio, respecting session language."""
    # Use language from session state, default to 'en' if not found
    selected_language = st.session_state.get("tts_language", "en") 
    try:
        tts = gTTS(text=text_to_speak, lang=selected_language, slow=False)
        audio_fp = BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0) # Important: move to the beginning of the BytesIO stream
        st.audio(audio_fp, format='audio/mp3', autoplay=True)
    except Exception as e:
        st.warning(f"Could not play speech (lang: {selected_language}): {e}")

# -------------------- UI CONFIG & APP NAMING --------------------
st.set_page_config(page_title="Takhfīd Genie", layout="wide", page_icon="🧞‍♂️")
APP_TITLE = "Takhfīd Genie"
APP_SUBTITLE = "Your AI-Powered Business First Package Pricing Expert."
MAIN_APP_IMAGE_FILENAME = "takhfid_genie_image.png"
ADCB_LOGO_FILENAME = "adcb_logo.png" # Provide this in the script directory
WATERMARK_IMAGE_FILENAME = "adcb_watermark.png" # Provide this in the script directory

# Construct absolute paths to the images
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_APP_IMAGE_PATH = os.path.join(SCRIPT_DIR, MAIN_APP_IMAGE_FILENAME)
ADCB_LOGO_PATH = os.path.join(SCRIPT_DIR, ADCB_LOGO_FILENAME)
WATERMARK_IMAGE_PATH = os.path.join(SCRIPT_DIR, WATERMARK_IMAGE_FILENAME)

# -------------------- PACKAGE CONFIG --------------------
packages = {
    "Package Digital": {
        "cost": 150,
        "transactions": {
            "international": {"free_count": 0, "rate_after_free": 60},
            "domestic": {"free_count": 9999, "rate_after_free": None}, # Unlimited free, or use client's rate if somehow exceeded
            "cheque": {"free_count": 0, "rate_after_free": 1}
        },
        "pdc": {"free_count": 0, "rate_after_free": 25},
        "inward_fcy_remittance": {"free_count": 0, "rate_after_free": 25},
        "other_costs_apply_input": True, # User's 'Other Costs' are added to this package's total
        "fx_buy_rate": 3.6770,
        "fx_sell_rate": 3.6690,
        "complimentary_items": ["Free Credit Cards", "Free Debit Cards", "ProCash Soft Token"]
    },
    "Package Essential": {
        "cost": 275,
        "transactions": {
            "international": {"free_count": 0, "rate_after_free": 30},
            "domestic": {"free_count": 9999, "rate_after_free": None},
            "cheque": {"free_count": 17, "rate_after_free": 1}
        },
        "pdc": {"free_count": 0, "rate_after_free": 25},
        "inward_fcy_remittance": {"free_count": 0, "rate_after_free": 25},
        "other_costs_apply_input": True,
        "fx_buy_rate": 3.6770,
        "fx_sell_rate": 3.6690,
        "complimentary_items": ["Free Credit Cards", "Free Debit Cards", "ProCash Soft Token"]
    },
    "Package Essential plus": {
        "cost": 375,
        "transactions": {
            "international": {"free_count": 0, "rate_after_free": 25},
            "domestic": {"free_count": 9999, "rate_after_free": None},
            "cheque": {"free_count": 17, "rate_after_free": 1}
        },
        "pdc": {"free_count": 99999, "rate_after_free": 0}, # Effectively free
        "inward_fcy_remittance": {"free_count": 99999, "rate_after_free": 0}, # Effectively free
        "other_costs_apply_input": False, # User's 'Other Costs' are NOT added (free with package)
        "fx_buy_rate": 3.6760,
        "fx_sell_rate": 3.6700,
        "complimentary_items": ["Free Credit Cards", "Free Debit Cards", "ProCash Soft Token"]
    },
    "Package Executive": {
        "cost": 800,
        "transactions": {
            "international": {"free_count": 75, "rate_after_free": 20},
            "domestic": {"free_count": 9999, "rate_after_free": None},
            "cheque": {"free_count": 17, "rate_after_free": 1}
        },
        "pdc": {"free_count": 99999, "rate_after_free": 0},
        "inward_fcy_remittance": {"free_count": 99999, "rate_after_free": 0},
        "other_costs_apply_input": False,
        "fx_buy_rate": 3.6740,
        "fx_sell_rate": 3.6710,
        "complimentary_items": ["Free Credit Cards", "Free Debit Cards", "ProCash Soft Token"]
    },
    "Package Executive plus": {
        "cost": 1650,
        "transactions": {
            "international": {"free_count": 75, "rate_after_free": 20},
            "domestic": {"free_count": 9999, "rate_after_free": None},
            "cheque": {"free_count": 17, "rate_after_free": 1}
        },
        "pdc": {"free_count": 99999, "rate_after_free": 0},
        "inward_fcy_remittance": {"free_count": 99999, "rate_after_free": 0},
        "other_costs_apply_input": False,
        "fx_buy_rate": 3.6735,
        "fx_sell_rate": 3.6720,
        "complimentary_items": ["Free Credit Cards", "Free Debit Cards", "ProCash Soft Token"]
    }
}

# -------------------- FUNCTIONS --------------------
def calculate_total_cost(transactions, transaction_costs, fx_amount, fx_direction, package_fx_rate, client_fx_rate, wps_cost, other_costs_input, package):
    package_cost = package["cost"]
    package_rules = package # The full package dictionary
    total_with_package = package_cost
    breakdown = {"Package Cost": package_cost}

    # Standard Transactions (International, Domestic, Cheque)
    for t_type in ["international", "domestic", "cheque"]:
        t_count = transactions.get(t_type, 0)
        if t_count == 0:
            breakdown[f"{t_type.capitalize()} Transactions Cost"] = 0.0
            continue
        rule = package_rules["transactions"].get(t_type)
        cost_for_this_type = 0.0
        if rule:
            free_count = rule.get("free_count", 0)
            rate_after_free = rule.get("rate_after_free")
            extra_transactions = max(0, t_count - free_count)
            if extra_transactions > 0:
                cost_for_this_type = extra_transactions * (rate_after_free if rate_after_free is not None else transaction_costs[t_type])
        else:
            cost_for_this_type = t_count * transaction_costs[t_type]
        breakdown[f"{t_type.capitalize()} Transactions Cost"] = cost_for_this_type
        total_with_package += cost_for_this_type

    # PDC and Inward FCY Remittance
    for service_type in ["pdc", "inward_fcy_remittance"]:
        s_count = transactions.get(service_type, 0)
        if s_count == 0:
            breakdown[f"{service_type.replace('_', ' ').title()} Cost"] = 0.0
            continue
        rule = package_rules.get(service_type)
        cost_for_this_service = 0.0
        if rule:
            free_count = rule.get("free_count", 0)
            rate_after_free = rule.get("rate_after_free")
            extra_services = max(0, s_count - free_count)
            if extra_services > 0:
                cost_for_this_service = extra_services * rate_after_free 
        else:
            cost_for_this_service = s_count * transaction_costs[service_type]
        breakdown[f"{service_type.replace('_', ' ').title()} Cost"] = cost_for_this_service
        total_with_package += cost_for_this_service

    # FX Impact (marginal cost/saving)
    if fx_amount > 0:
        if fx_direction == "Buy USD":
            fx_impact = (package_fx_rate - client_fx_rate) * fx_amount
            breakdown["FX Additional Cost"] = fx_impact
        else: # Sell USD
            fx_impact = (client_fx_rate - package_fx_rate) * fx_amount
            breakdown["FX Additional Cost"] = fx_impact
        total_with_package += fx_impact
    else:
        breakdown["FX Additional Cost"] = 0.0

    # WPS/CST Cost is free with all packages
    breakdown["WPS/CST Cost"] = 0.0

    # Other Costs
    if package_rules.get("other_costs_apply_input", False):
        breakdown["Other Costs (User Input)"] = other_costs_input
        total_with_package += other_costs_input
    else:
        breakdown["Other Costs (User Input)"] = 0.0 # Free with this package

    return total_with_package, breakdown


def suggest_best_package(transactions, transaction_costs, fx_amount, fx_direction, client_fx_rate, wps_cost, other_costs_input):
    # Calculate non-package cost
    base_txn_cost = sum(transactions.get(t, 0) * transaction_costs.get(t, 0) for t in ["international", "domestic", "cheque"])
    base_services_cost = transactions.get("pdc", 0) * transaction_costs.get("pdc", 0) + \
                         transactions.get("inward_fcy_remittance", 0) * transaction_costs.get("inward_fcy_remittance", 0)

    # For 'no package', FX impact is 0 (baseline)
    no_pkg_fx_impact = 0
    no_pkg_cost = base_txn_cost + base_services_cost + no_pkg_fx_impact + wps_cost + other_costs_input

    results = {}
    best_pkg = None
    max_savings = 0 # Initialize to 0, as savings can be negative (package costs more)

    for name, pkg in packages.items():
        package_fx_rate = pkg["fx_buy_rate"] if fx_direction == "Buy USD" else pkg["fx_sell_rate"]
        cost, breakdown = calculate_total_cost(
            transactions, transaction_costs, fx_amount, fx_direction, package_fx_rate, client_fx_rate, wps_cost, other_costs_input, pkg
        )
        savings = no_pkg_cost - cost
        results[name] = {"total_cost": cost, "savings": savings, "breakdown": breakdown}
        # Allow for negative savings, find package with highest savings (even if it means least costly)
        if best_pkg is None or savings > max_savings: 
            max_savings = savings
            best_pkg = name

    return best_pkg, max_savings, results, no_pkg_cost


def export_to_csv(results, user_data, best, savings, no_pkg_cost):
    # Base data for summary
    data = []
    # Detailed breakdown data
    detailed_data = []
    detailed_data.append(["Component", "Details", "Cost (AED)"])

    # --- No Package Cost Breakdown ---
    detailed_data.append(["No Package", "International Txns", f"{round(user_data['int_count'] * user_data['int_cost']):,d}"])
    detailed_data.append(["No Package", "Domestic Txns", f"{round(user_data['dom_count'] * user_data['dom_cost']):,d}"])
    detailed_data.append(["No Package", "Cheque Txns", f"{round(user_data['chq_count'] * user_data['chq_cost']):,d}"])
    detailed_data.append(["No Package", "PDC Txns", f"{round(user_data['pdc_count'] * user_data['pdc_cost']):,d}"])
    detailed_data.append(["No Package", "Inward FCY Remittances", f"{round(user_data['inward_fcy_count'] * user_data['inward_fcy_cost']):,d}"])
    if user_data['fx_amount'] > 0:
        detailed_data.append(["No Package", "FX Conversion (Market Rate)", f"{round(user_data['fx_amount'] * user_data['client_fx_rate']):,d}"])
    else:
        detailed_data.append(["No Package", "FX Conversion (Market Rate)", "0"])
    detailed_data.append(["No Package", "WPS/CST Cost", f"{round(user_data['wps_cost']):,d}"])
    detailed_data.append(["No Package", "Other Costs (User Input)", f"{round(user_data['other_costs_input']):,d}"])
    detailed_data.append(["No Package", "TOTAL COST (NO PACKAGE)", f"{round(no_pkg_cost):,d}"])
    detailed_data.append([]) # Blank line for separation

    # --- Package Cost Breakdowns ---
    for name, result_details in results.items():
        pkg_config = packages[name] # Accessing global 'packages'
        detailed_data.append([name, "Package Fee", f"{pkg_config['cost']:,d}"])
        
        # Loop for International, Domestic, Cheque (Transactions)
        pkg_tx_rules_csv = pkg_config["transactions"]
        for t_type_csv in ["international", "domestic", "cheque"]:
            user_txn_count_csv = user_data[t_type_csv.replace('international', 'int').replace('domestic', 'dom').replace('cheque', 'chq') + '_count']
            client_rate_csv = user_data[t_type_csv.replace('international', 'int').replace('domestic', 'dom').replace('cheque', 'chq') + '_cost']
            rule_csv = pkg_tx_rules_csv.get(t_type_csv)

            paid_count_csv = 0
            rate_applied_csv = client_rate_csv
            cost_of_paid_csv = 0

            if rule_csv:
                free_count_csv = rule_csv.get("free_count", 0)
                paid_count_csv = max(0, user_txn_count_csv - free_count_csv)
                if paid_count_csv > 0:
                    if rule_csv.get("rate_after_free") is not None:
                        rate_applied_csv = rule_csv["rate_after_free"]
                    cost_of_paid_csv = paid_count_csv * rate_applied_csv
            else:
                paid_count_csv = user_txn_count_csv
                cost_of_paid_csv = paid_count_csv * rate_applied_csv
            
            detailed_data.append([name, f"Paid {t_type_csv.capitalize()} Txns ({paid_count_csv:,} @ {rate_applied_csv:,.2f})", f"{round(cost_of_paid_csv):,d}"])

        # Loop for PDC and Inward FCY Remittance (Services)
        for s_type_csv in ["pdc", "inward_fcy_remittance"]:
            if s_type_csv == "inward_fcy_remittance":
                user_service_count_csv = user_data['inward_fcy_count'] 
                client_service_rate_csv = user_data['inward_fcy_cost']
            else: # For 'pdc'
                user_service_count_csv = user_data[s_type_csv + '_count'] # This will be 'pdc_count'
                client_service_rate_csv = user_data[s_type_csv + '_cost']  # This will be 'pdc_cost'

            rule_service_csv = pkg_config.get(s_type_csv) # These are direct keys in package

            paid_service_count_csv = 0
            rate_applied_service_csv = client_service_rate_csv
            cost_of_paid_service_csv = 0

            if rule_service_csv:
                free_service_count_csv = rule_service_csv.get("free_count", 0)
                paid_service_count_csv = max(0, user_service_count_csv - free_service_count_csv)
                if paid_service_count_csv > 0:
                    if rule_service_csv.get("rate_after_free") is not None:
                        rate_applied_service_csv = rule_service_csv["rate_after_free"]
                    cost_of_paid_service_csv = paid_service_count_csv * rate_applied_service_csv
            else:
                paid_service_count_csv = user_service_count_csv
                cost_of_paid_service_csv = paid_service_count_csv * rate_applied_service_csv
            
            detailed_data.append([name, f"Paid {s_type_csv.replace('_', ' ').title()} ({paid_service_count_csv:,} @ {rate_applied_service_csv:,.2f})", f"{round(cost_of_paid_service_csv):,d}"])

        if user_data['fx_amount'] > 0:
            pkg_fx_rate = pkg_config["fx_buy_rate"] if user_data["fx_direction"] == "Buy USD" else pkg_config["fx_sell_rate"]
            detailed_data.append([name, "FX Conversion (Package Rate)", f"{round(user_data['fx_amount'] * pkg_fx_rate):,d}"])
        else:
            detailed_data.append([name, "FX Conversion (Package Rate)", "0"])
        detailed_data.append([name, "WPS/CST Cost", "0"])
        
        # Other Costs with Package (CSV)
        other_costs_pkg_val = 0.0
        if pkg_config.get("other_costs_apply_input", False):
            other_costs_pkg_val = user_data['other_costs_input']
        detailed_data.append([name, f"Other Costs (User Input)", f"{round(other_costs_pkg_val):,d}"])

        detailed_data.append([name, f"TOTAL COST ({name})", f"{round(result_details['total_cost']):,d}"])
        detailed_data.append([name, f"SAVINGS ({name})", f"{round(result_details['savings']):,d}"])
        
        # Complimentary Items (CSV)
        complimentary_items_csv = pkg_config.get("complimentary_items", [])
        if complimentary_items_csv:
            detailed_data.append([name, "Complimentary Items", "; ".join(complimentary_items_csv)])
        else:
            detailed_data.append([name, "Complimentary Items", "None listed"])

        detailed_data.append([]) # Blank line for separation

    # Convert detailed data to DataFrame for CSV export
    df_detailed_export = pd.DataFrame(detailed_data[1:], columns=detailed_data[0])
    
    # Summary table data (remains for a quick overview)
    data.append(["Category", "Total Cost (AED)", "Savings (AED)"])
    data.append(["No Package", f"{round(no_pkg_cost):,d}", "-"])
    for name, result in results.items():
        data.append([name, f"{round(result['total_cost']):,d}", f"{round(result['savings']):,d}"])
    df_summary_export = pd.DataFrame(data[1:], columns=data[0])

    # Combine summary and detailed breakdown with a separator
    csv_buffer = BytesIO()
    df_summary_export.to_csv(csv_buffer, index=False, encoding='utf-8')
    csv_buffer.write(b"\n\nDetailed Calculation Breakdown:\n") # Add a title for the second part
    df_detailed_export.to_csv(csv_buffer, index=False, encoding='utf-8', header=True)
    
    return csv_buffer.getvalue()


def export_to_pdf(results, user_data, best, savings, no_pkg_cost, narrative_summary=""):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    # Title
    title_style = styles['Title']
    elements.append(Paragraph("Bank Package Savings Report", title_style))
    elements.append(Spacer(1, 12))
    
    # AI Narrative Summary
    if narrative_summary:
        narrative_style = styles['Italic']
        narrative_style.fontSize = 11
        elements.append(Paragraph(narrative_summary, narrative_style))
        elements.append(Spacer(1, 12))

    # Best package summary
    normal_style = styles['Normal']
    elements.append(Paragraph(f"<b>Best Package:</b> {best} | <b>Total Savings:</b> {round(savings):,} AED", normal_style))
    elements.append(Spacer(1, 12))

    # Create table data
    table_data = [
        ["Package Name", "Total Cost (AED)", "Savings (AED)"]
    ]
    for name, result in results.items():
        table_data.append([name, f"{round(result['total_cost']):,}", f"{round(result['savings']):,}"])

    # Add No Package row
    table_data.append(["No Package", f"{round(no_pkg_cost):,}", "-"])

    # Create table and style
    pdf_table = Table(table_data)
    pdf_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))

    elements.append(pdf_table)
    elements.append(Spacer(1, 24)) # Add more space before the breakdown

    # Detailed Calculation Breakdown Section
    breakdown_title_style = styles['h2']
    elements.append(Paragraph("Detailed Calculation Breakdown", breakdown_title_style))
    elements.append(Spacer(1, 12))

    # User Data for calculations (from the main app logic)
    # Ensure user_data, tx, tx_cost, best package name are available
    # We'll assume 'best' is the name of the best package, and 'packages' dict is accessible
    # or relevant parts of it are within 'results[best]'

    # Reconstruct user_data if not fully passed or structure differently
    # For this example, I'll assume 'user_data' contains all necessary client inputs like in the main app
    # and 'results[best]' contains the breakdown for the best package.
    # 'no_pkg_cost' is the total cost without a package.
    
    # --- No Package Cost Breakdown ---
    elements.append(Paragraph("<b>1. Costs Without Any Package:</b>", normal_style))
    elements.append(Spacer(1, 6))
    
    # Transaction Costs (No Package)
    elements.append(Paragraph(f"   - International Transactions: {user_data['int_count']:,} × {user_data['int_cost']:,.2f} = {round(user_data['int_count'] * user_data['int_cost']):,} AED", normal_style))
    elements.append(Paragraph(f"   - Domestic Transactions: {user_data['dom_count']:,} × {user_data['dom_cost']:,.2f} = {round(user_data['dom_count'] * user_data['dom_cost']):,} AED", normal_style))
    elements.append(Paragraph(f"   - Cheque Transactions: {user_data['chq_count']:,} × {user_data['chq_cost']:,.2f} = {round(user_data['chq_count'] * user_data['chq_cost']):,} AED", normal_style))
    elements.append(Paragraph(f"   - PDC Transactions: {user_data['pdc_count']:,} × {user_data['pdc_cost']:,.2f} = {round(user_data['pdc_count'] * user_data['pdc_cost']):,} AED", normal_style))
    elements.append(Paragraph(f"   - Inward FCY Remittance: {user_data['inward_fcy_count']:,} × {user_data['inward_fcy_cost']:,.2f} = {round(user_data['inward_fcy_count'] * user_data['inward_fcy_cost']):,} AED", normal_style))
    total_no_pkg_txn_costs = user_data['int_count'] * user_data['int_cost'] + \
                             user_data['dom_count'] * user_data['dom_cost'] + \
                             user_data['chq_count'] * user_data['chq_cost'] + \
                             user_data['pdc_count'] * user_data['pdc_cost'] + \
                             user_data['inward_fcy_count'] * user_data['inward_fcy_cost']
    elements.append(Paragraph(f"   - <b>Total Transaction & Service Costs (No Package):</b> {round(total_no_pkg_txn_costs):,} AED", normal_style))
    elements.append(Spacer(1, 6))

    # FX Impact (No Package)
    elements.append(Paragraph(f"   - <b>FX Impact (No Package):</b>", normal_style))
    elements.append(Paragraph(f"     - Direction: {user_data['fx_direction']}", normal_style))
    elements.append(Paragraph(f"     - Client's Rate: {user_data['client_fx_rate']:,.4f} AED/USD", normal_style))
    elements.append(Paragraph(f"     - FX Amount: {user_data['fx_amount']:,.2f} USD", normal_style))
    if user_data["fx_direction"] == "Buy USD":
        no_pkg_fx_display_value = user_data['fx_amount'] * user_data['client_fx_rate']
        elements.append(Paragraph(f"     - Resulting FX Cost (Buying USD): {round(no_pkg_fx_display_value):,} AED", normal_style))
    else: # Sell USD
        no_pkg_fx_display_value = user_data['fx_amount'] * user_data['client_fx_rate']
        elements.append(Paragraph(f"     - Resulting FX Proceeds (Selling USD at Client's Rate): {round(no_pkg_fx_display_value):,} AED", normal_style))
    elements.append(Spacer(1, 6))

    # WPS/CST Cost (No Package)
    elements.append(Paragraph(f"   - WPS/CST Cost: {round(user_data['wps_cost']):,} AED", normal_style))
    elements.append(Spacer(1, 6))
    
    # Other Costs (No Package)
    elements.append(Paragraph(f"   - Other Costs (User Input): {round(user_data['other_costs_input']):,} AED", normal_style))
    elements.append(Spacer(1, 6))

    elements.append(Paragraph(f"   - <b>Total Cost Without Any Package:</b> {round(no_pkg_cost):,} AED", normal_style))
    elements.append(Spacer(1, 12))

    # --- Best Package Cost Breakdown ---
    if best and best in results: # Check if a best package exists
        elements.append(Paragraph(f"<b>2. Costs With Best Package ({best}):</b>", normal_style))
        elements.append(Spacer(1, 6))
        
        pkg_details = packages[best] # Accessing global 'packages'
        best_pkg_result = results[best]

        elements.append(Paragraph(f"   - Package Fee: {pkg_details['cost']:,} AED", normal_style))
        
        # Transaction costs with package
        elements.append(Paragraph("   - <b>Paid Transactions & Services (after free units):</b>", normal_style))
        
        for item_type_pdf in ["international", "domestic", "cheque", "pdc", "inward_fcy_remittance"]:
            user_item_count_pdf = 0
            client_item_rate_pdf = 0.0
            # Get count and cost from appropriate user_data keys
            if item_type_pdf == "international":
                user_item_count_pdf = user_data['int_count']
                client_item_rate_pdf = user_data['int_cost']
            elif item_type_pdf == "domestic":
                user_item_count_pdf = user_data['dom_count']
                client_item_rate_pdf = user_data['dom_cost']
            elif item_type_pdf == "cheque":
                user_item_count_pdf = user_data['chq_count']
                client_item_rate_pdf = user_data['chq_cost']
            elif item_type_pdf == "pdc":
                user_item_count_pdf = user_data['pdc_count']
                client_item_rate_pdf = user_data['pdc_cost']
            elif item_type_pdf == "inward_fcy_remittance":
                user_item_count_pdf = user_data['inward_fcy_count']
                client_item_rate_pdf = user_data['inward_fcy_cost']

            rule_pdf = None
            if item_type_pdf in ["international", "domestic", "cheque"]:
                rule_pdf = pkg_details["transactions"].get(item_type_pdf)
            else: # pdc, inward_fcy_remittance are direct keys
                rule_pdf = pkg_details.get(item_type_pdf)
            
            paid_count_pdf = 0
            rate_applied_pdf = client_item_rate_pdf
            cost_of_paid_pdf = 0

            if rule_pdf:
                free_count_pdf = rule_pdf.get("free_count", 0)
                paid_count_pdf = max(0, user_item_count_pdf - free_count_pdf)
                if paid_count_pdf > 0:
                    if rule_pdf.get("rate_after_free") is not None:
                        rate_applied_pdf = rule_pdf["rate_after_free"]
                    cost_of_paid_pdf = paid_count_pdf * rate_applied_pdf
            else:
                paid_count_pdf = user_item_count_pdf
                cost_of_paid_pdf = paid_count_pdf * rate_applied_pdf
            
            elements.append(Paragraph(f"     - {item_type_pdf.replace('_', ' ').title()}: {paid_count_pdf:,} × {rate_applied_pdf:,.2f} = {round(cost_of_paid_pdf):,} AED", normal_style))
        elements.append(Spacer(1, 6))

        # FX Impact with package
        elements.append(Paragraph(f"   - <b>FX Impact with Package ({best}):</b>", normal_style))
        elements.append(Paragraph(f"     - Direction: {user_data['fx_direction']}", normal_style))
        pkg_fx_rate = pkg_details["fx_buy_rate"] if user_data["fx_direction"] == "Buy USD" else pkg_details["fx_sell_rate"]
        elements.append(Paragraph(f"     - Package Rate: {pkg_fx_rate:,.4f} AED/USD", normal_style))
        elements.append(Paragraph(f"     - FX Amount: {user_data['fx_amount']:,.2f} USD", normal_style))
        if user_data["fx_direction"] == "Buy USD":
            pkg_fx_display_value = user_data['fx_amount'] * pkg_fx_rate
            elements.append(Paragraph(f"     - Resulting FX Cost (Buying USD): {round(pkg_fx_display_value):,} AED", normal_style))
        else: # Sell USD
            pkg_fx_display_value = user_data['fx_amount'] * pkg_fx_rate
            elements.append(Paragraph(f"     - Resulting FX Proceeds (Selling USD at Package Rate): {round(pkg_fx_display_value):,} AED", normal_style))
            fx_gain_from_package_rate = pkg_fx_display_value - (user_data['fx_amount'] * user_data['client_fx_rate'])
            elements.append(Paragraph(f"     - Additional gain from package rate vs client's rate: {round(fx_gain_from_package_rate):,} AED", normal_style))
        elements.append(Spacer(1, 6))

        # WPS/CST Cost (same as no package)
        elements.append(Paragraph(f"   - WPS/CST Cost: 0 AED", normal_style))
        elements.append(Spacer(1, 6))

        # Other Costs with Package (PDF)
        elements.append(Paragraph(f"   - <b>Other Costs (User Input) with Package ({best}):</b>", normal_style))
        if pkg_details.get("other_costs_apply_input", False):
            elements.append(Paragraph(f"     - Other Costs Added: {round(user_data['other_costs_input']):,} AED", normal_style))
        else:
            elements.append(Paragraph(f"     - Other Costs Included/Free with Package: 0 AED", normal_style))
        elements.append(Spacer(1, 6))

        elements.append(Paragraph(f"   - <b>Total Cost With Best Package ({best}):</b> {round(best_pkg_result['total_cost']):,} AED", normal_style))
        elements.append(Spacer(1, 12))

        # --- Savings ---
        elements.append(Paragraph(f"<b>3. Total Savings ({best}):</b> {round(savings):,} AED", normal_style))
        elements.append(Spacer(1, 12))

        # Complimentary Items (PDF)
        elements.append(Paragraph(f"<b>4. Complimentary Items with Package ({best}):</b>", normal_style))
        complimentary_items_pdf = pkg_details.get("complimentary_items", [])
        if complimentary_items_pdf:
            for item in complimentary_items_pdf:
                elements.append(Paragraph(f"   - {item}", normal_style))
        else:
            elements.append(Paragraph("   - None listed.", normal_style))

    # Build PDF
    doc.build(elements)

    # Move to beginning of buffer and return bytes
    buffer.seek(0)
    return buffer.getvalue()

def create_breakdown_chart(results, user_data, tx, tx_cost):
    """Creates a stacked bar chart to show the breakdown of costs for each package."""
    breakdown_data = []

    # 1. No package breakdown
    no_pkg_breakdown = {
        "Transactions (Int'l, Dom, Chq)": (tx['international'] * tx_cost['international'] +
                                           tx['domestic'] * tx_cost['domestic'] +
                                           tx['cheque'] * tx_cost['cheque']),
        "Services (PDC, Inward FCY)": (user_data['pdc_count'] * user_data['pdc_cost'] +
                                       user_data['inward_fcy_count'] * user_data['inward_fcy_cost']),
        "WPS/CST Cost": user_data['wps_cost'],
        "Other Costs": user_data['other_costs_input'],
    }
    if user_data['fx_amount'] > 0:
        fx_impact_no_pkg = user_data['fx_amount'] * user_data['client_fx_rate']
        if user_data['fx_direction'] == "Sell USD":
             fx_impact_no_pkg = -fx_impact_no_pkg
        no_pkg_breakdown["FX Impact"] = fx_impact_no_pkg
    
    for component, cost in no_pkg_breakdown.items():
        if cost != 0:
             breakdown_data.append({"Category": "Without Package", "Cost Component": component, "Cost (AED)": cost})

    # 2. Package breakdowns from results
    for name, result_details in results.items():
        breakdown = result_details['breakdown']
        
        # Group components for a cleaner chart legend
        grouped_breakdown = {}
        if breakdown.get("Package Cost", 0) != 0:
            grouped_breakdown["Package Fee"] = breakdown.get("Package Cost", 0)
        
        txn_cost = (breakdown.get("International Transactions Cost", 0) + 
                    breakdown.get("Domestic Transactions Cost", 0) + 
                    breakdown.get("Cheque Transactions Cost", 0))
        if txn_cost != 0: grouped_breakdown["Transactions (Paid)"] = txn_cost
        
        services_cost = (breakdown.get("Pdc Cost", 0) + 
                         breakdown.get("Inward Fcy Remittance Cost", 0))
        if services_cost != 0: grouped_breakdown["Services (Paid)"] = services_cost
        
        if breakdown.get("FX Impact", 0) != 0:
            grouped_breakdown["FX Impact"] = breakdown["FX Impact"]
        
        if breakdown.get("Other Costs (User Input)", 0) != 0:
            grouped_breakdown["Other Costs"] = breakdown["Other Costs (User Input)"]
        
        for component, cost in grouped_breakdown.items():
            breakdown_data.append({"Category": name, "Cost Component": component, "Cost (AED)": cost})

    if not breakdown_data:
        return None

    df_breakdown = pd.DataFrame(breakdown_data)
    fig_breakdown = px.bar(df_breakdown, x="Category", y="Cost (AED)", color="Cost Component", 
                           title="📊 Detailed Cost Breakdown by Component",
                           labels={"Cost (AED)": "Cost (AED)", "Category": "Option", "Cost Component": "Component"},
                           text_auto=',.0f')
    fig_breakdown.update_layout(bargap=0.3, legend_title_text='Cost Component')
    return fig_breakdown


def generate_narrative_summary(best_pkg, savings, user_data, no_pkg_cost, results):
    """Generates a narrative summary of the analysis using an AI model."""
    if not best_pkg:
        return "No savings were identified with any package based on the provided data."

    # Create a simplified data summary for the prompt
    data_summary = {
        "Client's Monthly Transactions": {
            "International": f"{user_data['int_count']} at {user_data['int_cost']:.2f} AED each",
            "Domestic": f"{user_data['dom_count']} at {user_data['dom_cost']:.2f} AED each",
            "Cheques": f"{user_data['chq_count']} at {user_data['chq_cost']:.2f} AED each",
            "PDCs": f"{user_data['pdc_count']} at {user_data['pdc_cost']:.2f} AED each",
            "Inward FCY": f"{user_data['inward_fcy_count']} at {user_data['inward_fcy_cost']:.2f} AED each"
        },
        "FX Volume": f"{user_data['fx_amount']:.2f} USD ({user_data['fx_direction']})",
        "Other Monthly Costs": {
            "WPS/CST": f"{user_data['wps_cost']:.2f} AED",
            "Miscellaneous": f"{user_data['other_costs_input']:.2f} AED"
        },
        "Analysis Outcome": {
            "Cost without any package": f"{no_pkg_cost:,.0f} AED",
            "Recommended Package": best_pkg,
            "Cost with this package": f"{results[best_pkg]['total_cost']:,.0f} AED",
            "Total Monthly Savings": f"{savings:,.0f} AED"
        }
    }
    
    # Identify key saving drivers
    savings_drivers = []
    best_pkg_rules = packages[best_pkg]
    
    # Transaction savings
    if user_data['int_count'] > best_pkg_rules['transactions']['international']['free_count']:
        savings_drivers.append("a significant number of free international transfers")
    # FX savings
    if user_data['fx_amount'] > 0:
        savings_drivers.append("preferential FX rates")
    # Service savings (PDC, etc.)
    if user_data['pdc_count'] > best_pkg_rules['pdc']['free_count'] or user_data['inward_fcy_count'] > best_pkg_rules['inward_fcy_remittance']['free_count']:
        savings_drivers.append("inclusive processing of PDCs and inward remittances")
    # Other costs
    if not best_pkg_rules['other_costs_apply_input'] and user_data['other_costs_input'] > 0:
         savings_drivers.append("the waiver of miscellaneous monthly fees")

    if savings_drivers:
        data_summary["Key Savings Drivers"] = ", ".join(savings_drivers)

    prompt_data = json.dumps(data_summary, indent=2)

    system_prompt = (
        "You are a sophisticated financial advisor's assistant for ADCB. Your task is to write a brief, professional, one-paragraph executive summary for a client report. "
        "The summary should be confident and persuasive, written in a formal business tone. "
        "It must highlight the recommended package, the total estimated monthly savings (in AED), and the primary financial advantages (key savings drivers) that lead to this recommendation. "
        "Use the provided JSON data to craft your response. Do not invent new facts. Start the summary with 'Based on a comprehensive analysis of your transaction profile...'"
    )

    try:
        client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Please generate the executive summary based on this data:\n{prompt_data}"}
            ],
            temperature=0.4,
            max_tokens=250,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Could not generate AI summary: {e}")
        return "An AI-generated summary could not be created at this time."

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

def generate_analysis():
    data = st.session_state.transaction_data
    
    tx = {
        "international": data["international"]["count"],
        "domestic": data["domestic"]["count"],
        "cheque": data["cheque"]["count"],
        "pdc": data["pdc"]["count"],
        "inward_fcy_remittance": data["inward_fcy_remittance"]["count"]
    }
    tx_cost = {
        "international": data["international"]["cost"],
        "domestic": data["domestic"]["cost"],
        "cheque": data["cheque"]["cost"],
        "pdc": data["pdc"]["cost"],
        "inward_fcy_remittance": data["inward_fcy_remittance"]["cost"]
    }
    
    best, savings, results, no_pkg_cost = suggest_best_package(
        tx, tx_cost,
        data["fx"]["amount"],
        data["fx"]["direction"],
        data["fx"]["rate"],  # This is the client_fx_rate for "no package" calculation
        data["wps"]["cost"],
        data["other_costs_input"]
    )
    
    user_data_for_main_display = {
        "int_count": data["international"]["count"],
        "int_cost": data["international"]["cost"],
        "dom_count": data["domestic"]["count"],
        "dom_cost": data["domestic"]["cost"],
        "chq_count": data["cheque"]["count"],
        "chq_cost": data["cheque"]["cost"],
        "pdc_count": data["pdc"]["count"],
        "pdc_cost": data["pdc"]["cost"],
        "inward_fcy_count": data["inward_fcy_remittance"]["count"],
        "inward_fcy_cost": data["inward_fcy_remittance"]["cost"],
        "fx_amount": data["fx"]["amount"],
        "fx_direction": data["fx"]["direction"],
        "client_fx_rate": data["fx"]["rate"],
        "wps_enabled": data["wps"]["enabled"],
        "wps_cost": data["wps"]["cost"],
        "other_costs_input": data["other_costs_input"]
    }

    if best:
        # Generate narrative summary for AI mode
        with st.spinner("Generating AI-powered executive summary..."):
            narrative = generate_narrative_summary(best, savings, user_data_for_main_display, no_pkg_cost, results)
        
        st.session_state.analysis_results = {
            "best": best,
            "savings": savings,
            "results": results,
            "no_pkg_cost": no_pkg_cost,
            "user_data": user_data_for_main_display,
            "tx": tx,
            "tx_cost": tx_cost,
            "narrative_summary": narrative
        }
        st.session_state.submitted = True
        st.session_state.show_welcome = False
        return True # Indicate success
    else:
        if "analysis_results" in st.session_state:
            del st.session_state.analysis_results
        st.session_state.submitted = False
        # show_welcome can remain as is, or be set to True if we want to show welcome screen again.
        # For now, let it be, so the user sees the 'no savings' message in context.
        return False # Indicate no savings/best package found

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

# --- Client Profile and Form State Management ---
if "client_profiles" not in st.session_state:
    st.session_state.client_profiles = {}
if "current_client_name" not in st.session_state:
    st.session_state.current_client_name = ""
# Initialize all manual form fields in session_state if they don't exist
manual_fields = {
    'int_count_manual_form': 0, 'int_cost_manual_form': 0.0,
    'dom_count_manual_form': 0, 'dom_cost_manual_form': 0.0,
    'chq_count_manual_form': 0, 'chq_cost_manual_form': 0.0,
    'fx_direction_manual_form': "Buy USD", 'fx_amount_manual_form': 0.0,
    'fx_buy_rate_manual_form': 3.67, 'fx_sell_rate_manual_form': 3.63,
    'manual_form_wps_enabled_chkbx_onchange': False, 
    'manual_form_wps_cost_input_field_onchange': 0.0,
    'pdc_count_manual_form': 0, 'pdc_cost_manual_form': 0.0,
    'inward_fcy_count_manual_form': 0, 'inward_fcy_cost_manual_form': 0.0,
    'other_costs_manual_form': 0.0
}
for key, default_value in manual_fields.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

if "chat_stage" not in st.session_state: 
    init_chat_state()

# --- Authentication Logic ---
def check_password():
    """Returns `True` if the user is authenticated, `False` otherwise."""
    if st.session_state.get("authenticated", False):
        return True

    st.title("Takhfīd Genie 🧞‍♂️ - Login")
    st.markdown("Please enter the password to access the application.")
    
    password = st.text_input("Password", type="password", key="password_input")

    if st.button("Login", key="login_button"):
        # The password is now managed in .streamlit/secrets.toml
        correct_password = st.secrets["APP_PASSWORD"]
        if password == correct_password:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("The password you entered is incorrect.")
            
    return False

# If not authenticated, show login and stop the app from running further.
if not check_password():
    st.stop()


# -------------------- UI RENDER STARTS HERE --------------------

def apply_custom_css():
    """Applies custom CSS for styling, including sidebar watermark."""
    watermark_base64 = img_to_base64(WATERMARK_IMAGE_PATH)
    
    # Base CSS for titles and sidebar width
    st.markdown(f"""
    <style>
        /* Main Title Styles */
        .main-title {{
            font-size: 4rem !important;
            font-weight: 700;
            color: #e4002b; /* ADCB Red */
            text-align: left;
        }}
        .sub-title {{
            font-size: 1.25rem;
            color: #333;
            text-align: left;
            margin-bottom: 20px;
        }}
        /* Fixed sidebar width */
        [data-testid="stSidebar"] {{
            min-width: 320px !important;
            max-width: 450px !important;
            width: 350px !important;
        }}
    </style>
    """, unsafe_allow_html=True)

    # Add watermark if image is available
    if watermark_base64:
        st.markdown(f"""
        <style>
            [data-testid="stSidebar"] > div:first-child {{
                position: relative;
            }}
            [data-testid="stSidebar"] > div:first-child::before {{
                content: "";
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background-image: url("data:image/png;base64,{watermark_base64}");
                background-size: 70%;
                background-repeat: no-repeat;
                background-position: center 20px;
                opacity: 0.1;
                z-index: -1;
                pointer-events: none;
            }}
        </style>
        """, unsafe_allow_html=True)
    else:
        # Show a warning in the sidebar if the watermark image is not found
        st.sidebar.warning("Sidebar watermark image not found.", icon="⚠️")

apply_custom_css()

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
    <p style="color: #e60013; font-size: 20px; line-height: 1.6; text-align: justify; margin-left: 14px;">
        <strong><em>
            &nbsp;&nbsp;&nbsp;&nbsp;An intelligent, AI-driven pricing solution that empowers your team to confidently 
            propose the optimal Business First Package. Powered by advanced analytics and a deep evaluation of each client's 
            financial profile and existing banking costs, it fosters data-backed conversations, enhanced client alignment, 
            and measurable value creation.
        </em></strong>
    </p>
    """, unsafe_allow_html=True)


    st.markdown("---") 

    st.markdown("### 🧭 Choose Input Mode")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📝 Manual Mode", key="mode_manual_revert", use_container_width=True, help="Enter inputs manually"):
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
        if st.button("🤖 AI Assistant", key="mode_ai_revert", use_container_width=True, help="Chat with AI to analyze your needs"):
            st.session_state.input_mode = 'AI Assistant'
            st.session_state.submitted = False
            st.session_state.show_welcome = True
            init_chat_state() # Initialize/Reset AI chat state
            st.rerun()
    st.markdown("---")

    # TTS Language Selection (Only relevant for AI Assistant mode)
    if st.session_state.input_mode == "AI Assistant":
        st.markdown("### 🗣️ AI Voice Language")
        language_options = {
            "English": "en",
            "العربية (Arabic)": "ar",
            "Français (French)": "fr",
            # "हिन्दी (Hindi)": "hi"
        }
        selected_lang_display = st.selectbox(
            "Select voice language:", 
            options=list(language_options.keys()),
            index=list(language_options.values()).index(st.session_state.get("tts_language", "en")), 
            key="tts_lang_select_sidebar"
        )
        st.session_state.tts_language = language_options[selected_lang_display]
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
                if st.button("Load Profile", use_container_width=True, disabled=(not profile_to_load)):
                    profile_data = st.session_state.client_profiles[profile_to_load]
                    for key, value in profile_data.items():
                        st.session_state[key] = value
                    st.session_state.current_client_name = profile_to_load
                    st.success(f"Profile '{profile_to_load}' loaded.")
                    st.rerun()

            # --- SAVE PROFILE ---
            st.markdown("---")
            new_client_name = st.text_input("Enter Client Name to Save", value=st.session_state.current_client_name)
            if st.button("Save Profile", use_container_width=True, disabled=(not new_client_name)):
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

        if st.button("🔍 Analyze", use_container_width=True, key="manual_analyze_button"):
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
                
                best, savings, results, no_pkg_cost = suggest_best_package( tx, tx_cost, user_data["fx_amount"], user_data["fx_direction"], user_data["client_fx_rate"], user_data["wps_cost"], user_data["other_costs_input"])
                
                if best:
                    # Generate narrative summary
                    with st.spinner("Generating AI-powered executive summary..."):
                        narrative = generate_narrative_summary(best, savings, user_data, no_pkg_cost, results)
                    
                    st.session_state.analysis_results = { "best": best, "savings": savings, "results": results, "no_pkg_cost": no_pkg_cost, "user_data": user_data, "tx": tx, "tx_cost": tx_cost, "narrative_summary": narrative }
                    st.rerun()
                else:
                    st.warning("No suitable package found or an error occurred during analysis.")
                    if "analysis_results" in st.session_state: del st.session_state.analysis_results
                    st.session_state.submitted = False

    elif st.session_state.input_mode == "AI Assistant":
        st.markdown("### 💬 AI Assistant") # Title for AI assistant mode
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
    if st.button("Logout", use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()


# Main content area
if st.session_state.show_welcome:
    # Display the image on the welcome screen
    try:
        st.image(MAIN_APP_IMAGE_PATH, use_container_width=True) # Uses MAIN_APP_IMAGE_PATH
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
    results = st.session_state.analysis_results
    best = results["best"]
    savings = results["savings"]
    no_pkg_cost = results["no_pkg_cost"]
    user_data = results["user_data"]
    tx = results["tx"]
    tx_cost = results["tx_cost"]
    
    # --- Display AI Narrative Summary ---
    if "narrative_summary" in results and results["narrative_summary"]:
        st.markdown("###  Executive Summary")
        st.markdown(f"> *{results['narrative_summary']}*")
        st.markdown("---")

    # --- Calculation details with correct FX Additional Cost logic ---
    best_pkg_fx_rate = packages[best]["fx_buy_rate"] if user_data["fx_direction"] == "Buy USD" else packages[best]["fx_sell_rate"]
    if user_data["fx_direction"] == "Buy USD":
        fx_additional_cost_no_pkg = (user_data["client_fx_rate"] - best_pkg_fx_rate) * user_data["fx_amount"]
    else:
        fx_additional_cost_no_pkg = (best_pkg_fx_rate - user_data["client_fx_rate"]) * user_data["fx_amount"]
    no_pkg_total_cost = no_pkg_cost + fx_additional_cost_no_pkg
    best_pkg_fees = results['results'][best]['total_cost'] - results['results'][best]['breakdown'].get('FX Additional Cost', 0)
    savings = no_pkg_total_cost - best_pkg_fees

    md_lines = []
    md_lines.append("### 🔢 **Calculation Details**")
    md_lines.append(f"Here's how choosing **{best}** saves you **{round(savings):,} AED**:")
    md_lines.append("\n1. **Transaction Costs Without Any Package**:")
    md_lines.append(f"   - International: {tx['international']:,} × {tx_cost['international']:,.2f} = {round(tx['international'] * tx_cost['international']):,} AED")
    md_lines.append(f"   - Domestic: {tx['domestic']:,} × {tx_cost['domestic']:,.2f} = {round(tx['domestic'] * tx_cost['domestic']):,} AED")
    md_lines.append(f"   - Cheque: {tx['cheque']:,} × {tx_cost['cheque']:,.2f} = {round(tx['cheque'] * tx_cost['cheque']):,} AED")
    md_lines.append(f"   - PDC: {user_data['pdc_count']:,} × {user_data['pdc_cost']:,.2f} = {round(user_data['pdc_count'] * user_data['pdc_cost']):,} AED")
    md_lines.append(f"   - Inward FCY Remittance: {user_data['inward_fcy_count']:,} × {user_data['inward_fcy_cost']:,.2f} = {round(user_data['inward_fcy_count'] * user_data['inward_fcy_cost']):,} AED")
    total_no_pkg_base_txn_costs = tx['international'] * tx_cost['international'] + \
                                  tx['domestic'] * tx_cost['domestic'] + \
                                  tx['cheque'] * tx_cost['cheque'] + \
                                  user_data['pdc_count'] * user_data['pdc_cost'] + \
                                  user_data['inward_fcy_count'] * user_data['inward_fcy_cost']
    # elements.append(Paragraph(f"   - **Total Base Transaction & Service Costs**: {round(total_no_pkg_base_txn_costs):,} AED", normal_style))
    # elements.append(Spacer(1, 6))

    md_lines.append("\n2. **FX Additional Cost Without Any Package (vs. Best Package)**:")
    md_lines.append(f"   - FX Additional Cost: {round(fx_additional_cost_no_pkg):,} AED (This is the extra you would have paid in FX by not using the {best} package rate.)")
    
    md_lines.append("\n3. **WPS/CST Cost**:")
    md_lines.append(f"   - WPS/CST Enabled: {'Yes' if user_data['wps_enabled'] else 'No'}")
    md_lines.append(f"   - WPS/CST Cost: {round(user_data['wps_cost']):,} AED")
    
    md_lines.append("\n4. **Other Costs (User Input - No Package)**:")
    md_lines.append(f"   - Other Costs: {round(user_data['other_costs_input']):,} AED")
    
    md_lines.append("\n5. **Total Cost Without Any Package**:")
    md_lines.append(f"   - Total Cost: {round(no_pkg_total_cost):,} AED")
    
    md_lines.append(f"\n6. **Cost With Best Package ({best})**:")
    md_lines.append(f"   - Package Cost: {packages[best]['cost']:,} AED")
    md_lines.append("   - Remaining Transactions & Services (Paid): ")
    
    pkg_rules = packages[best]
    for item_type in ["international", "domestic", "cheque", "pdc", "inward_fcy_remittance"]:
        user_item_count = 0
        client_item_rate = 0.0
        if item_type == "international":
            user_item_count = user_data['int_count']
            client_item_rate = user_data['int_cost']
        elif item_type == "domestic":
            user_item_count = user_data['dom_count']
            client_item_rate = user_data['dom_cost']
        elif item_type == "cheque":
            user_item_count = user_data['chq_count']
            client_item_rate = user_data['chq_cost']
        elif item_type == "pdc":
            user_item_count = user_data['pdc_count']
            client_item_rate = user_data['pdc_cost']
        elif item_type == "inward_fcy_remittance":
            user_item_count = user_data['inward_fcy_count']
            client_item_rate = user_data['inward_fcy_cost']
        rule = None
        if item_type in ["international", "domestic", "cheque"]:
            rule = pkg_rules["transactions"].get(item_type)
        else:
            rule = pkg_rules.get(item_type)
        paid_count = 0
        rate_applied = client_item_rate
        cost_of_paid = 0
        if rule:
            free_count = rule.get("free_count", 0)
            paid_count = max(0, user_item_count - free_count)
            if paid_count > 0:
                if rule.get("rate_after_free") is not None:
                    rate_applied = rule["rate_after_free"]
                cost_of_paid = paid_count * rate_applied
        else:
            paid_count = user_item_count
            cost_of_paid = paid_count * rate_applied
        md_lines.append(f"     - {item_type.replace('_', ' ').title()}: {paid_count} × {rate_applied:.2f} = {round(cost_of_paid):,} AED")
    
    md_lines.append(f"\n   - FX Additional Cost with Package ({best}): 0 AED (You are using the best package rate)")
    md_lines.append(f"\n   - WPS/CST Cost: 0 AED")
    md_lines.append(f"\n   - Other Costs (User Input) with Package ({best}):")
    if packages[best].get("other_costs_apply_input", False):
        md_lines.append(f"     - Other Costs Added: {round(user_data['other_costs_input']):,} AED")
    else:
        md_lines.append(f"     - Other Costs Included/Free with Package: 0 AED")
    md_lines.append(f"   - **Total Cost With Best Package**: {round(best_pkg_fees):,} AED")
    md_lines.append("\n7. **Savings**:")
    md_lines.append(f"   - Total Cost Without Package - Total Cost With Best Package = **{round(savings):,} AED**")
    md_lines.append("\n8. **Complimentary Items with this Package**:")
    complimentary = packages[best].get("complimentary_items", [])
    if complimentary:
        for item in complimentary:
            md_lines.append(f"   - {item}")
    else:
        md_lines.append("   - None listed.")
    calculation_details_md = "\n".join(md_lines)
    with st.expander("🔍 How We Calculated Your Savings", expanded=False):
        st.markdown(calculation_details_md, unsafe_allow_html=True)

    # Graphs and export options
    if best:
        # --- Infographic-style smooth area/peak plot for cost comparison ---
        categories = ["Without Package"] + list(results["results"].keys())
        costs = [no_pkg_total_cost] + [results["results"][pkg]["total_cost"] for pkg in results["results"]]
        chart_colors = ['#444', '#1f77b4', '#17becf', '#2ca02c', '#ffbb78', '#ff7f0e', '#d62728']
        fig = go.Figure()
        for i, (cat, cost) in enumerate(zip(categories, costs)):
            x_peak = np.linspace(i-0.4, i+0.4, 50)
            y_peak = cost * np.exp(-((x_peak - i) ** 2) / 0.04)
            fig.add_trace(go.Scatter(
                x=x_peak, y=y_peak,
                fill='tozeroy',
                mode='lines',
                line=dict(color=chart_colors[i % len(chart_colors)], width=2),
                fillcolor=chart_colors[i % len(chart_colors)],
                name=cat,
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=[i], y=[cost],
                mode='markers+text',
                marker=dict(size=18, color=chart_colors[i % len(chart_colors)], line=dict(width=2, color='white')),
                text=[f"{int(cost):,}"],
                textposition="top center",
                showlegend=False
            ))
        fig.update_xaxes(
            tickvals=list(range(len(categories))),
            ticktext=categories,
            showgrid=False
        )
        fig.update_yaxes(title_text="Total Cost (AED)", showgrid=False)
        fig.update_layout(
            title="💰 Total Monthly Cost by Option",
            plot_bgcolor='white',
            margin=dict(l=40, r=40, t=60, b=40),
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

        # Graph 2: Savings Comparison and Pie Chart
        with st.container():
            col1, col2 = st.columns([2, 1], gap="large")
            with col1:
                st.markdown('<br>', unsafe_allow_html=True)
                st.markdown("### 💸 Net Financial Impact by Package")
                import pandas as pd
                df_savings = pd.DataFrame({
                    "Package": list(results["results"].keys()),
                    "Savings (AED)": [results["results"][pkg]["savings"] for pkg in results["results"]],
                })
                bar_colors = ['#e4002b', '#ffbb78', '#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']
                fig_savings = go.Figure()
                for i, row in df_savings.iterrows():
                    badge = " (Best)" if row["Package"] == best else ""
                    color = '#888' if row["Package"] == "No Package" else bar_colors[i % len(bar_colors)]
                    fig_savings.add_trace(go.Bar(
                        x=[row["Package"]],
                        y=[row["Savings (AED)"]],
                        marker=dict(
                            color=color,
                            line=dict(width=2, color='#fff'),
                            ),
                        text=[f"{row['Savings (AED)']:,.0f}{badge}"],
                        textposition='outside',
                        width=0.5,
                        name=row["Package"]
                    ))
                fig_savings.update_layout(
                    width=800,
                    height=500,
                    margin=dict(l=20, r=20, t=20, b=80),
                    plot_bgcolor='#fff',
                    paper_bgcolor='#fff',
                    bargap=0.3,
                    bargroupgap=0.1,
                    font=dict(size=16),
                    showlegend=False,
                    xaxis=dict(showgrid=False, zeroline=False, showline=False, linecolor='#fff', gridcolor='#fff'),
                    yaxis=dict(showgrid=False, zeroline=False, showline=False, linecolor='#fff', gridcolor='#fff'),
                )
                st.plotly_chart(fig_savings, use_container_width=True)
            with col2:
                st.markdown('<br>', unsafe_allow_html=True)
                # st.markdown(f"### 🥇 {best} Savings Breakdown")
                st.markdown(f"### 🥇 Savings Overview")
                # Calculate savings components
                no_pkg = results["results"].get("No Package", None)
                best_pkg = results["results"][best]
                if no_pkg:
                    no_pkg_breakdown = no_pkg["breakdown"]
                else:
                    no_pkg_breakdown = {
                        "International Transactions Cost": user_data["int_count"] * user_data["int_cost"],
                        "Domestic Transactions Cost": user_data["dom_count"] * user_data["dom_cost"],
                        "Cheque Transactions Cost": user_data["chq_count"] * user_data["chq_cost"],
                        "Pdc Cost": user_data["pdc_count"] * user_data["pdc_cost"],
                        "Inward Fcy Remittance Cost": user_data["inward_fcy_count"] * user_data["inward_fcy_cost"],
                        "FX Additional Cost": user_data["fx_amount"] * user_data["client_fx_rate"] if user_data["fx_amount"] > 0 else 0,
                        "Other Costs (User Input)": user_data.get("other_costs_input", 0)
                    }
                best_pkg_breakdown = best_pkg["breakdown"]
                savings_components = {}
                fx_saving = no_pkg_breakdown.get("FX Additional Cost", 0) - best_pkg_breakdown.get("FX Additional Cost", 0)
                if abs(fx_saving) > 1e-2:
                    savings_components["FX Savings"] = fx_saving
                int_saving = no_pkg_breakdown.get("International Transactions Cost", 0) - best_pkg_breakdown.get("International Transactions Cost", 0)
                if abs(int_saving) > 1e-2:
                    savings_components["International Txn Savings"] = int_saving
                dom_saving = no_pkg_breakdown.get("Domestic Transactions Cost", 0) - best_pkg_breakdown.get("Domestic Transactions Cost", 0)
                if abs(dom_saving) > 1e-2:
                    savings_components["Domestic Txn Savings"] = dom_saving
                chq_saving = no_pkg_breakdown.get("Cheque Transactions Cost", 0) - best_pkg_breakdown.get("Cheque Transactions Cost", 0)
                if abs(chq_saving) > 1e-2:
                    savings_components["Cheque Savings"] = chq_saving
                pdc_saving = no_pkg_breakdown.get("Pdc Cost", 0) - best_pkg_breakdown.get("Pdc Cost", 0)
                if abs(pdc_saving) > 1e-2:
                    savings_components["PDC Savings"] = pdc_saving
                inward_fcy_saving = no_pkg_breakdown.get("Inward Fcy Remittance Cost", 0) - best_pkg_breakdown.get("Inward Fcy Remittance Cost", 0)
                if abs(inward_fcy_saving) > 1e-2:
                    savings_components["Inward FCY Savings"] = inward_fcy_saving
                other_saving = no_pkg_breakdown.get("Other Costs (User Input)", 0) - best_pkg_breakdown.get("Other Costs (User Input)", 0)
                if abs(other_saving) > 1e-2:
                    savings_components["Other Savings"] = other_saving
                # Shorten/clarify labels for bar chart
                label_map = {
                    "FX Savings": "FX",
                    "International Txn Savings": "Intl",
                    "Domestic Txn Savings": "Dom",
                    "Cheque Savings": "Chq",
                    "PDC Savings": "PDC",
                    "Inward FCY Savings": "FCY",
                    "Other Savings": "Other"
                }
                bar_labels = [label_map.get(lbl, lbl) for lbl in savings_components.keys()]
                bar_values = [abs(v) for v in savings_components.values()]
                bar_colors = ['#e4002b', '#ffbb78', '#1f77b4', '#2ca02c', '#ff7f0e', '#d62728'][:len(bar_labels)]
                fig_bar = go.Figure(go.Bar(
                    x=bar_values,
                    y=bar_labels,
                    orientation='h',
                    marker=dict(
                        color=bar_colors,
                        line=dict(color='#fff', width=2)
                    ),
                    text=[f"{v:,.0f} AED" for v in bar_values],
                    textposition='outside'
                ))
                fig_bar.update_layout(
                    title=f"{best} Savings Breakdown",
                    xaxis_title="Savings (AED)",
                    yaxis_title="Component",
                    plot_bgcolor='#fff',
                    paper_bgcolor='#fff',
                    margin=dict(l=10, r=10, t=40, b=10),
                    width=600,
                    height=500,
                )
                st.plotly_chart(fig_bar)

        # What-If Analysis section
        with st.expander("🤔 Interactive What-If Analysis", expanded=False):
            st.markdown("Use the sliders to see how your savings change with different transaction volumes.")
            
            col1, col2 = st.columns(2)
            with col1:
                max_int = int(user_data['int_count'] * 2.5) + 10
                what_if_int_count = st.slider("International Transfers", 0, max_int, user_data['int_count'])
                
                max_pdc = int(user_data['pdc_count'] * 2.5) + 10
                what_if_pdc_count = st.slider("PDCs Processed", 0, max_pdc, user_data['pdc_count'])

            with col2:
                max_fx = int(user_data['fx_amount'] * 2.5) + 5000
                what_if_fx_amount = st.slider("FX Amount (USD)", 0.0, float(max_fx), float(user_data['fx_amount']))

                max_dom = int(user_data['dom_count'] * 2.5) + 10
                what_if_dom_count = st.slider("Domestic Transfers", 0, max_dom, user_data['dom_count'])

            # Prepare new inputs for re-calculation
            what_if_tx = tx.copy()
            what_if_tx['international'] = what_if_int_count
            what_if_tx['domestic'] = what_if_dom_count
            what_if_tx['pdc'] = what_if_pdc_count
            
            # Re-run the analysis with the slider values
            best_wi, savings_wi, results_wi, no_pkg_cost_wi = suggest_best_package(
                what_if_tx, tx_cost, what_if_fx_amount, user_data["fx_direction"],
                user_data["client_fx_rate"], user_data["wps_cost"], user_data["other_costs_input"]
            )

            st.markdown("---")
            if best_wi:
                st.success(f"With these new values, **{best_wi}** would be the best package, saving you **{round(savings_wi):,} AED**.")
            else:
                st.warning("With these values, no package offers savings over the client's current costs.")

            # Re-draw the charts for the what-if scenario
            # For what-if, recalculate the FX Additional Cost for the new best package
            best_pkg_breakdown_wi = results_wi[best_wi]['breakdown'] if best_wi else {}
            # For what-if, no_pkg_cost_wi is the baseline (no FX Additional Cost)
            displayed_no_pkg_cost_wi = no_pkg_cost_wi
            df_cost_wi = pd.DataFrame({
                "Category": ["Without Package"] + list(results_wi.keys()),
                "Cost (AED)": [displayed_no_pkg_cost_wi] + [r["total_cost"] for r in results_wi.values()],
            })
            fig_cost_wi = px.bar(df_cost_wi, x="Category", y="Cost (AED)", title="💰 What-If: Total Cost",
                                 color="Category", text_auto=',.0f')
            st.plotly_chart(fig_cost_wi, use_container_width=True)
            
            df_savings_wi = pd.DataFrame({
                "Package": list(results_wi.keys()),
                "Savings (AED)": [r["savings"] for r in results_wi.values()],
            })
            fig_savings_wi = px.bar(df_savings_wi, x="Package", y="Savings (AED)", title="💸 What-If: Savings",
                                 color="Package", text_auto=',.0f')
            
            st.plotly_chart(fig_savings_wi, use_container_width=True)

        # Export Options
        st.markdown("### 📤 Export Results")
        col1, col2 = st.columns(2)
        with col1:
            csv = export_to_csv(results["results"], user_data, best, savings, no_pkg_cost)
            st.download_button("📥 Export to CSV", data=csv, file_name="package_comparison.csv", mime="text/csv")
        with col2:
            pdf_bytes = export_to_pdf(results["results"], user_data, best, savings, no_pkg_cost, results.get("narrative_summary", ""))
            st.download_button("📄 Export to PDF", data=pdf_bytes, file_name="package_comparison.pdf", mime="application/pdf")
