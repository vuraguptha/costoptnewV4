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
import pyttsx3
import base64 # Added for image encoding
import numpy as np
import plotly.graph_objects as go
from matplotlib.colors import to_rgb
from gtts import gTTS


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
    """Generates speech from text and plays it using st.audio."""
    try:
        # Use a specific language and TLD that consistently provides a male voice
        tts = gTTS(text=text_to_speak, lang='en', tld='com.au', slow=False)
        audio_fp = BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        st.audio(audio_fp, format='audio/mp3', autoplay=True)
    except Exception as e:
        st.warning(f"Could not play speech: {e}")

# -------------------- UI CONFIG & APP NAMING --------------------
APP_TITLE = "Fikra Genie"
# APP_SUBTITLE = "Your AI-Powered Business First Package Pricing Expert"
APP_SUBTITLE = """Powered by advanced analytics and a deep evaluation of each client's financial profile,<br>it enables data-driven conversations, stronger client alignment, and measurable value creation."""
st.set_page_config(page_title="Fikra Genie", layout="wide", page_icon="🧞‍♂️")
MAIN_APP_IMAGE_FILENAME = "takhfid_genie_image.png"
ADCB_LOGO_FILENAME = "adcb_logo.png"  # ADCB logo file
WATERMARK_IMAGE_FILENAME = "adcb_watermark.png"

# Directory setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_APP_IMAGE_PATH = os.path.join(SCRIPT_DIR, MAIN_APP_IMAGE_FILENAME)
ADCB_LOGO_PATH = os.path.join(SCRIPT_DIR, ADCB_LOGO_FILENAME)
WATERMARK_IMAGE_PATH = os.path.join(SCRIPT_DIR, WATERMARK_IMAGE_FILENAME)

# At the top with other constants
# WATERMARK_IMAGE_PATH = "sidebar_logo.png"
# ADCB_LOGO_PATH = "sidebar_logo.png"  # Using the same logo for now, you can replace with actual ADCB logo later

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
    """
    This function is DEPRECATED. The logic has been integrated into the new suggest_best_package.
    """
    return 0, {}


def suggest_best_package(transactions, transaction_costs, fx_amount, fx_direction, client_fx_rate, wps_cost, other_costs_input):
    """
    Calculates the best package based on the lowest true total cost, as per the user's simulation.
    The "true total cost" is the actual, absolute amount of money spent for each option.
    It returns the results dictionary containing true costs, savings, and breakdowns for all options.
    """
    results = {}

    # --- Stage 1: Calculate True Total Cost for "Without Package" ---
    no_pkg_txn_cost = sum(
        transactions.get(t, 0) * transaction_costs.get(t, 0)
        for t in ["international", "domestic", "cheque", "pdc", "inward_fcy_remittance"]
    )
    # Absolute FX Cost for the client's current setup
    no_pkg_fx_cost = (client_fx_rate * fx_amount) if fx_amount > 0 else 0
    no_pkg_true_total_cost = no_pkg_txn_cost + no_pkg_fx_cost + wps_cost + other_costs_input
    
    results["Without Package"] = {
        "true_total_cost": no_pkg_true_total_cost,
        "breakdown": {
            "International Transactions Cost": transactions.get("international", 0) * transaction_costs.get("international", 0),
            "Domestic Transactions Cost": transactions.get("domestic", 0) * transaction_costs.get("domestic", 0),
            "Cheque Transactions Cost": transactions.get("cheque", 0) * transaction_costs.get("cheque", 0),
            "Pdc Cost": transactions.get("pdc", 0) * transaction_costs.get("pdc", 0),
            "Inward Fcy Remittance Cost": transactions.get("inward_fcy_remittance", 0) * transaction_costs.get("inward_fcy_remittance", 0),
            "Absolute FX Cost": no_pkg_fx_cost,
            "Other Costs (User Input)": other_costs_input,
            "WPS/CST Cost": wps_cost,
            "Package Cost": 0.0,
        }
    }

    # --- Stage 2: Calculate True Total Cost for each package ---
    for name, pkg in packages.items():
        pkg_fee = pkg["cost"]
        breakdown = {"Package Cost": pkg_fee}

        # Calculate cost of transactions NOT covered by the package
        paid_txn_cost = 0
        for t_type in ["international", "domestic", "cheque"]:
            t_count = transactions.get(t_type, 0)
            rule = pkg["transactions"].get(t_type)
            cost_for_this_type = 0
            if rule:
                extra_transactions = max(0, t_count - rule.get("free_count", 0))
                if extra_transactions > 0 and rule.get("rate_after_free") is not None:
                    cost_for_this_type = extra_transactions * rule.get("rate_after_free")
            else: # If no rule, client pays their standard rate
                 cost_for_this_type = t_count * transaction_costs[t_type]
            breakdown[f"{t_type.capitalize()} Transactions Cost"] = cost_for_this_type
            paid_txn_cost += cost_for_this_type
        
        # Calculate cost of services NOT covered
        paid_services_cost = 0
        for service_type in ["pdc", "inward_fcy_remittance"]:
            s_count = transactions.get(service_type, 0)
            rule = pkg.get(service_type)
            cost_for_this_service = 0
            if rule:
                extra_services = max(0, s_count - rule.get("free_count", 0))
                if extra_services > 0 and rule.get("rate_after_free") is not None:
                    cost_for_this_service = extra_services * rule.get("rate_after_free")
            breakdown[f"{service_type.replace('_', ' ').title()} Cost"] = cost_for_this_service
            paid_services_cost += cost_for_this_service

        # Calculate Absolute FX cost using the PACKAGE'S rate
        package_fx_rate = pkg["fx_buy_rate"] if fx_direction == "Buy USD" else pkg["fx_sell_rate"]
        absolute_fx_cost_of_package = (package_fx_rate * fx_amount) if fx_amount > 0 else 0
        breakdown["Absolute FX Cost"] = absolute_fx_cost_of_package

        # WPS cost is free with packages
        breakdown["WPS/CST Cost"] = 0.0

        # Add other costs if they apply to the package
        other_costs_with_pkg = 0.0
        if pkg.get("other_costs_apply_input", False):
            other_costs_with_pkg = other_costs_input
        breakdown["Other Costs (User Input)"] = other_costs_with_pkg

        # Sum everything for the final True Total Cost
        total_with_package = (
            pkg_fee
            + paid_txn_cost
            + paid_services_cost
            + absolute_fx_cost_of_package
            + other_costs_with_pkg
        )

        results[name] = {
            "true_total_cost": total_with_package,
            "breakdown": breakdown
        }

    # --- Stage 3: Find the best option and calculate savings for all options ---
    best_option_name = min(results, key=lambda k: results[k]['true_total_cost'])
    
    # Calculate savings relative to "Without Package" cost
    no_pkg_total_cost_for_savings = results["Without Package"]["true_total_cost"]
    for name in results:
        results[name]['savings'] = no_pkg_total_cost_for_savings - results[name]['true_total_cost']

    total_savings = results[best_option_name]['savings']

    return best_option_name, total_savings, results


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
        if name == "Without Package":
            continue  # Skip "Without Package" as it's already handled above
            
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

        detailed_data.append([name, f"TOTAL COST ({name})", f"{round(result_details['true_total_cost']):,d}"])
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
        if name == "Without Package": continue
        data.append([name, f"{round(result['true_total_cost']):,d}", f"{round(result['savings']):,d}"])
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
        if name == "Without Package": continue
        table_data.append([name, f"{round(result['true_total_cost']):,}", f"{round(result['savings']):,}"])

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
    # elements.append(Paragraph(f"   - <b>Total Transaction & Service Costs (No Package):</b> {round(total_no_pkg_txn_costs):,} AED", normal_style))
    # elements.append(Spacer(1, 6))

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

        elements.append(Paragraph(f"   - <b>Total Cost With Best Package ({best}):</b> {round(best_pkg_result['true_total_cost']):,} AED", normal_style))
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
            "Cost with this package": f"{results[best_pkg]['true_total_cost']:,.0f} AED",
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
    
    best, savings, results = suggest_best_package(
        tx, tx_cost,
        data["fx"]["amount"],
        data["fx"]["direction"],
        data["fx"]["rate"],
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
            no_pkg_true_cost = results["Without Package"]["true_total_cost"]
            narrative = generate_narrative_summary(best, savings, user_data_for_main_display, no_pkg_true_cost, results)
        
        st.session_state.analysis_results = {
            "best": best,
            "savings": savings,
            "results": results,
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
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["APP_PASSWORD"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False
            st.error("Incorrect password. Please try again.")

    # Initialize session state
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False
        
    # First run, show input for password
    if not st.session_state["password_correct"]:
        # Convert logo to base64
        adcb_logo_base64 = img_to_base64(ADCB_LOGO_PATH)
        
        # Use direct title with custom styling and ADCB logo
        st.markdown(f"""
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; padding: 1rem;">
                <h1 style="color: #e4002b; font-size: 4rem; font-weight: 700; margin: 0;">
                    Fikra Genie 🧞‍♂️💡 - Login
                </h1>
                <img src="data:image/png;base64,{adcb_logo_base64}" style="height: 80px; object-fit: contain;" alt="ADCB Logo">
            </div>
        """, unsafe_allow_html=True)
        st.markdown("""
            <h2 style="font-size: 2rem; margin-bottom: 1rem;">
                Please enter the password to access the application.
            </h2>
        """, unsafe_allow_html=True)
        
        # Password input and login button
        password = st.text_input("Password", type="password", key="password")
        if st.button("Login", type="primary"):
            if password == st.secrets["APP_PASSWORD"]:
                st.session_state["password_correct"] = True
                st.rerun()
            else:
                st.error("Incorrect password. Please try again.")
        return False
    
    return True

# If not authenticated, show login and stop the app from running further.
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

def apply_custom_css():
    """Applies custom CSS for styling, including sidebar watermark."""
    watermark_base64 = img_to_base64(WATERMARK_IMAGE_PATH)
    
    # Base CSS for titles and sidebar width
    st.markdown(f"""
    <style>
        /* Main Title Styles */
        .main-title {{
            font-size: 6rem !important;  /* Increased from 4rem to 6rem */
            font-weight: 700;
            color: #e4002b; /* ADCB Red */
            text-align: left;
            margin-bottom: 0.5rem;  /* Added margin for better spacing */
            line-height: 1.2;  /* Added for better line height */
        }}
        /* Login Title Styles */
        .login-title {{
            color: #e4002b !important;  /* ADCB Red */
            font-size: 3rem !important;
            font-weight: 700 !important;
            margin-bottom: 2rem !important;
        }}
        /* Additional Login Title Style to ensure color override */
        [data-testid="stMarkdownContainer"] h1 {{
            color: #e4002b !important;
        }}
        /* Welcome Image Styles */
        .welcome-image {{
            max-width: 200px;  /* Adjust this value to control image size */
            width: 90%;
            height: auto;
            margin: 2rem auto;
            display: block;
        }}
        .sub-title {{
            font-size: 1.8rem;
            color: #333;
            text-align: left;
            margin-bottom: 20px;
            line-height: 1.4;
        }}
        /* Fixed sidebar width */
        [data-testid="stSidebar"] {{
            width: 450px !important;
        }}
        /* Change sidebar font size */
        [data-testid="stSidebar"] * {{
            font-size: 1.3rem !important;
        }}
        /* Custom Expander Styles for 'What-If' */
        [data-testid="stExpander"] summary {{
            font-size: 1.5rem !important;
            font-weight: bold !important;
            padding: 1rem 0 !important;
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
                opacity: 0;
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
    <p style="color: #e60013; font-size: 28px; line-height: 1.6; text-align: justify; margin-left: 14px;">
        <strong><em>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;An intelligent, AI-driven pricing solution that empowers your team to confidently 
             propose the optimal Business First Package. 
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
            if st.button("🔄 Reset", use_container_width=True, key="manual_reset_button_after_analysis"):
                st.session_state.manual_reset_pending = True
                st.rerun()

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
            best_pkg_fee = results[best_pkg_name]["breakdown"].get("Package Cost", 0)

            # 3. Calculate the 'actual cost' bar value for each option
            bar_values = []
            for name, true_cost in zip(sorted_categories, sorted_true_costs):
                if name == best_pkg_name:
                    bar_values.append(best_pkg_fee)
                else:
                    # For all others: (True Cost - Best True Cost) + Best Fee
                    bar_values.append((true_cost - best_pkg_true_cost) + best_pkg_fee)

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
            
            # Increase number size on bars
            fig_cost.update_traces(texttemplate='%{y:,.0f}', textposition='outside', textfont_size=20)

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
            st.plotly_chart(fig_cost, use_container_width=True)
            
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
                
                st.plotly_chart(fig_savings_main, use_container_width=True)
            
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
                    best_pkg_fee_wi = results_wi[best_wi]["breakdown"].get("Package Cost", 0)

                    # 3. Calculate bar values
                    bar_values_wi = []
                    for name, true_cost in zip(sorted_categories_wi, sorted_true_costs_wi):
                        if name == best_wi: bar_values_wi.append(best_pkg_fee_wi)
                        else: bar_values_wi.append((true_cost - best_pkg_true_cost_wi) + best_pkg_fee_wi)

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
                    fig_cost_wi.update_traces(texttemplate='%{y:,.0f}', textposition='outside', textfont_size=20)
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
                        arrowwidth=8, arrowcolor="#A0E6A0", opacity=1
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
                    st.plotly_chart(fig_cost_wi, use_container_width=True)

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
                    st.plotly_chart(fig_savings_wi, use_container_width=True)
        
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
    
