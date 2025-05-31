import streamlit as st
import pandas as pd
import plotly as px
# import openai
import os
import json
# from fpdf import FPDF
from dotenv import load_dotenv
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import time


# load_dotenv()

# Set OpenAI API Key
# openai.api_key = os.getenv("OPENAI_API_KEY")

# -------------------- UI CONFIG & APP NAMING --------------------
st.set_page_config(page_title="Takhfīd Genie", layout="wide", page_icon="🧞‍♂️") 
APP_TITLE = "Takhfīd Genie"
MAIN_APP_IMAGE_FILENAME = "takhfid_genie_image.png" 
SIDEBAR_IMAGE_FILENAME = "sidebar_logo.png" # <<< YOU NEED TO PROVIDE THIS IMAGE FILE

# Construct absolute paths to the images
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_APP_IMAGE_PATH = os.path.join(SCRIPT_DIR, MAIN_APP_IMAGE_FILENAME)
SIDEBAR_IMAGE_PATH = os.path.join(SCRIPT_DIR, SIDEBAR_IMAGE_FILENAME)

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
def calculate_total_cost(transactions, transaction_costs, fx_amount, fx_direction, fx_rate_used, wps_cost, other_costs_input, package):
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
        
        rule = package_rules.get(service_type) # These rules are directly under package, not under 'transactions' sub-dict
        cost_for_this_service = 0.0
        if rule: # Package has specific rules for this service
            free_count = rule.get("free_count", 0)
            rate_after_free = rule.get("rate_after_free")
            extra_services = max(0, s_count - free_count)
            if extra_services > 0:
                cost_for_this_service = extra_services * rate_after_free 
        else:
             # Fallback: if for some reason a package is missing these new service rules (should not happen with new dict)
            cost_for_this_service = s_count * transaction_costs[service_type]
        breakdown[f"{service_type.replace('_', ' ').title()} Cost"] = cost_for_this_service
        total_with_package += cost_for_this_service

    # FX Impact
    if fx_amount > 0:
        if fx_direction == "Buy USD":
            fx_impact = fx_amount * fx_rate_used
            breakdown["FX Cost (Buying USD)"] = fx_impact
        else: # Sell USD
            fx_impact = -(fx_amount * fx_rate_used)
            breakdown["FX Benefit (Selling USD)"] = -fx_impact 
        total_with_package += fx_impact

    # WPS/CST Cost
    breakdown["WPS/CST Cost"] = wps_cost
    total_with_package += wps_cost

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

    if fx_amount > 0:
        if fx_direction == "Buy USD":
            no_pkg_fx_impact = fx_amount * client_fx_rate
        else: # Sell USD
            no_pkg_fx_impact = -(fx_amount * client_fx_rate)
    else:
        no_pkg_fx_impact = 0

    no_pkg_cost = base_txn_cost + base_services_cost + no_pkg_fx_impact + wps_cost + other_costs_input

    results = {}
    best_pkg = None
    max_savings = 0 # Initialize to 0, as savings can be negative (package costs more)

    for name, pkg in packages.items():
        fx_rate_for_pkg = pkg["fx_buy_rate"] if fx_direction == "Buy USD" else pkg["fx_sell_rate"]
        cost, breakdown = calculate_total_cost(
            transactions, transaction_costs, fx_amount, fx_direction, fx_rate_for_pkg, wps_cost, other_costs_input, pkg
        )
        savings = no_pkg_cost - cost
        results[name] = {"total_cost": cost, "savings": savings, "breakdown": breakdown}
        # Allow for negative savings, find package with highest savings (even if it means least costly)
        if best_pkg is None or savings > max_savings: 
            max_savings = savings
            best_pkg = name

    return best_pkg, max_savings, results, no_pkg_cost


# def parse_input_with_ai(user_input):
#     system_prompt = (
#         "You are a financial assistant. Extract transaction counts and unit costs from text. "
#         "Reply in this exact JSON format: "
#         '{"international_count": int, "international_cost": float, '
#         '"domestic_count": int, "domestic_cost": float, '
#         '"cheque_count": int, "cheque_cost": float, '
#         '"wps_enabled": true/false, "wps_cost": float, '
#         '"fx_amount": float, "fx_direction": "Buy USD"/"Sell USD", "fx_rate": float}'
#     )
#     try:
#         response = openai.ChatCompletion.create(
#             model="gpt-3.5-turbo",
#             messages=[
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": user_input}
#             ],
#             temperature=0.2,
#         )
#         json_str = response.choices[0].message.content.strip()
#         parsed = json.loads(json_str)

#         # Validate FX direction
#         if parsed["fx_direction"] not in ["Buy USD", "Sell USD"]:
#             parsed["fx_direction"] = "Buy USD"
#         return parsed
#     except Exception as e:
#         st.error(f"Error parsing AI response: {e}")
#         return None


def export_to_csv(results, user_data, best, savings, no_pkg_cost):
    # Base data for summary
    data = []
    # Detailed breakdown data
    detailed_data = []
    detailed_data.append(["Component", "Details", "Cost (AED)"])

    # --- No Package Cost Breakdown ---
    detailed_data.append(["No Package", "International Txns", f"{user_data['int_count'] * user_data['int_cost']:.2f}"])
    detailed_data.append(["No Package", "Domestic Txns", f"{user_data['dom_count'] * user_data['dom_cost']:.2f}"])
    detailed_data.append(["No Package", "Cheque Txns", f"{user_data['chq_count'] * user_data['chq_cost']:.2f}"])
    detailed_data.append(["No Package", "PDC Txns", f"{user_data['pdc_count'] * user_data['pdc_cost']:.2f}"])
    detailed_data.append(["No Package", "Inward FCY Remittances", f"{user_data['inward_fcy_count'] * user_data['inward_fcy_cost']:.2f}"])
    if user_data['fx_amount'] > 0:
        detailed_data.append(["No Package", "FX Conversion (Market Rate)", f"{user_data['fx_amount'] * user_data['client_fx_rate']:.2f}"])
    else:
        detailed_data.append(["No Package", "FX Conversion (Market Rate)", "0.00"])
    detailed_data.append(["No Package", "WPS/CST Cost", f"{user_data['wps_cost']:.2f}"])
    detailed_data.append(["No Package", "Other Costs (User Input)", f"{user_data['other_costs_input']:.2f}"])
    detailed_data.append(["No Package", "TOTAL COST (NO PACKAGE)", f"{no_pkg_cost:.2f}"])
    detailed_data.append([]) # Blank line for separation

    # --- Package Cost Breakdowns ---
    for name, result_details in results.items():
        pkg_config = packages[name] # Accessing global 'packages'
        detailed_data.append([name, "Package Fee", f"{pkg_config['cost']:.2f}"])
        
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
            
            detailed_data.append([name, f"Paid {t_type_csv.capitalize()} Txns ({paid_count_csv} @ {rate_applied_csv:.2f})", f"{cost_of_paid_csv:.2f}"])

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
            
            detailed_data.append([name, f"Paid {s_type_csv.replace('_', ' ').title()} ({paid_service_count_csv} @ {rate_applied_service_csv:.2f})", f"{cost_of_paid_service_csv:.2f}"])

        if user_data['fx_amount'] > 0:
            pkg_fx_rate = pkg_config["fx_buy_rate"] if user_data["fx_direction"] == "Buy USD" else pkg_config["fx_sell_rate"]
            detailed_data.append([name, "FX Conversion (Package Rate)", f"{user_data['fx_amount'] * pkg_fx_rate:.2f}"])
        else:
            detailed_data.append([name, "FX Conversion (Package Rate)", "0.00"])
        detailed_data.append([name, "WPS/CST Cost", f"{user_data['wps_cost']:.2f}"])
        
        # Other Costs with Package (CSV)
        other_costs_pkg_val = 0.0
        if pkg_config.get("other_costs_apply_input", False):
            other_costs_pkg_val = user_data['other_costs_input']
        detailed_data.append([name, f"Other Costs (User Input)", f"{other_costs_pkg_val:.2f}"])

        detailed_data.append([name, f"TOTAL COST ({name})", f"{result_details['total_cost']:.2f}"])
        detailed_data.append([name, f"SAVINGS ({name})", f"{result_details['savings']:.2f}"])
        
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
    data.append(["No Package", f"{no_pkg_cost:.2f}", "-"])
    for name, result in results.items():
        data.append([name, f"{result['total_cost']:.2f}", f"{result['savings']:.2f}"])
    df_summary_export = pd.DataFrame(data[1:], columns=data[0])

    # Combine summary and detailed breakdown with a separator
    csv_buffer = BytesIO()
    df_summary_export.to_csv(csv_buffer, index=False, encoding='utf-8')
    csv_buffer.write(b"\n\nDetailed Calculation Breakdown:\n") # Add a title for the second part
    df_detailed_export.to_csv(csv_buffer, index=False, encoding='utf-8', header=True)
    
    return csv_buffer.getvalue()


def export_to_pdf(results, user_data, best, savings, no_pkg_cost):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    # Title
    title_style = styles['Title']
    elements.append(Paragraph("Bank Package Savings Report", title_style))
    elements.append(Spacer(1, 12))

    # Best package summary
    normal_style = styles['Normal']
    elements.append(Paragraph(f"<b>Best Package:</b> {best} | <b>Total Savings:</b> {savings:.2f} AED", normal_style))
    elements.append(Spacer(1, 12))

    # Create table data
    table_data = [
        ["Package Name", "Total Cost (AED)", "Savings (AED)"]
    ]
    for name, result in results.items():
        table_data.append([name, f"{result['total_cost']:.2f}", f"{result['savings']:.2f}"])

    # Add No Package row
    table_data.append(["No Package", f"{no_pkg_cost:.2f}", "-"])

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
    elements.append(Paragraph(f"   - International Transactions: {user_data['int_count']} × {user_data['int_cost']:.2f} = {user_data['int_count'] * user_data['int_cost']:.2f} AED", normal_style))
    elements.append(Paragraph(f"   - Domestic Transactions: {user_data['dom_count']} × {user_data['dom_cost']:.2f} = {user_data['dom_count'] * user_data['dom_cost']:.2f} AED", normal_style))
    elements.append(Paragraph(f"   - Cheque Transactions: {user_data['chq_count']} × {user_data['chq_cost']:.2f} = {user_data['chq_count'] * user_data['chq_cost']:.2f} AED", normal_style))
    elements.append(Paragraph(f"   - PDC Transactions: {user_data['pdc_count']} × {user_data['pdc_cost']:.2f} = {user_data['pdc_count'] * user_data['pdc_cost']:.2f} AED", normal_style))
    elements.append(Paragraph(f"   - Inward FCY Remittances: {user_data['inward_fcy_count']} × {user_data['inward_fcy_cost']:.2f} = {user_data['inward_fcy_count'] * user_data['inward_fcy_cost']:.2f} AED", normal_style))
    total_no_pkg_txn_costs = user_data['int_count'] * user_data['int_cost'] + \
                             user_data['dom_count'] * user_data['dom_cost'] + \
                             user_data['chq_count'] * user_data['chq_cost'] + \
                             user_data['pdc_count'] * user_data['pdc_cost'] + \
                             user_data['inward_fcy_count'] * user_data['inward_fcy_cost']
    elements.append(Paragraph(f"   - <b>Total Transaction & Service Costs (No Package):</b> {total_no_pkg_txn_costs:.2f} AED", normal_style))
    elements.append(Spacer(1, 6))

    # FX Impact (No Package)
    elements.append(Paragraph(f"   - <b>FX Impact (No Package):</b>", normal_style))
    elements.append(Paragraph(f"     - Direction: {user_data['fx_direction']}", normal_style))
    elements.append(Paragraph(f"     - Client's Rate: {user_data['client_fx_rate']:.4f} AED/USD", normal_style))
    elements.append(Paragraph(f"     - FX Amount: {user_data['fx_amount']:.2f} USD", normal_style))
    if user_data["fx_direction"] == "Buy USD":
        no_pkg_fx_display_value = user_data['fx_amount'] * user_data['client_fx_rate']
        elements.append(Paragraph(f"     - Resulting FX Cost (Buying USD): {no_pkg_fx_display_value:.2f} AED", normal_style))
    else: # Sell USD
        no_pkg_fx_display_value = user_data['fx_amount'] * user_data['client_fx_rate']
        elements.append(Paragraph(f"     - Resulting FX Proceeds (Selling USD at Client's Rate): {no_pkg_fx_display_value:.2f} AED", normal_style))
    elements.append(Spacer(1, 6))

    # WPS/CST Cost (No Package)
    elements.append(Paragraph(f"   - WPS/CST Cost: {user_data['wps_cost']:.2f} AED", normal_style))
    elements.append(Spacer(1, 6))
    
    # Other Costs (No Package)
    elements.append(Paragraph(f"   - Other Costs (User Input): {user_data['other_costs_input']:.2f} AED", normal_style))
    elements.append(Spacer(1, 6))

    elements.append(Paragraph(f"   - <b>Total Cost Without Any Package:</b> {no_pkg_cost:.2f} AED", normal_style))
    elements.append(Spacer(1, 12))

    # --- Best Package Cost Breakdown ---
    if best and best in results: # Check if a best package exists
        elements.append(Paragraph(f"<b>2. Costs With Best Package ({best}):</b>", normal_style))
        elements.append(Spacer(1, 6))
        
        pkg_details = packages[best] # Accessing global 'packages'
        best_pkg_result = results[best]

        elements.append(Paragraph(f"   - Package Fee: {pkg_details['cost']:.2f} AED", normal_style))
        
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
            
            elements.append(Paragraph(f"     - {item_type_pdf.replace('_', ' ').title()}: {paid_count_pdf} × {rate_applied_pdf:.2f} = {cost_of_paid_pdf:.2f} AED", normal_style))
        elements.append(Spacer(1, 6))

        # FX Impact with package
        elements.append(Paragraph(f"   - <b>FX Impact with Package ({best}):</b>", normal_style))
        elements.append(Paragraph(f"     - Direction: {user_data['fx_direction']}", normal_style))
        pkg_fx_rate = pkg_details["fx_buy_rate"] if user_data["fx_direction"] == "Buy USD" else pkg_details["fx_sell_rate"]
        elements.append(Paragraph(f"     - Package Rate: {pkg_fx_rate:.4f} AED/USD", normal_style))
        elements.append(Paragraph(f"     - FX Amount: {user_data['fx_amount']:.2f} USD", normal_style))
        if user_data["fx_direction"] == "Buy USD":
            pkg_fx_display_value = user_data['fx_amount'] * pkg_fx_rate
            elements.append(Paragraph(f"     - Resulting FX Cost (Buying USD): {pkg_fx_display_value:.2f} AED", normal_style))
        else: # Sell USD
            pkg_fx_display_value = user_data['fx_amount'] * pkg_fx_rate
            elements.append(Paragraph(f"     - Resulting FX Proceeds (Selling USD at Package Rate): {pkg_fx_display_value:.2f} AED", normal_style))
            fx_gain_from_package_rate = pkg_fx_display_value - (user_data['fx_amount'] * user_data['client_fx_rate'])
            elements.append(Paragraph(f"     - Additional gain from package rate vs client's rate: {fx_gain_from_package_rate:.2f} AED", normal_style))
        elements.append(Spacer(1, 6))

        # WPS/CST Cost (same as no package)
        elements.append(Paragraph(f"   - WPS/CST Cost: {user_data['wps_cost']:.2f} AED", normal_style))
        elements.append(Spacer(1, 6))

        # Other Costs with Package (PDF)
        elements.append(Paragraph(f"   - <b>Other Costs (User Input) with Package ({best}):</b>", normal_style))
        if pkg_details.get("other_costs_apply_input", False):
            elements.append(Paragraph(f"     - Other Costs Added: {user_data['other_costs_input']:.2f} AED", normal_style))
        else:
            elements.append(Paragraph(f"     - Other Costs Included/Free with Package: 0.00 AED", normal_style))
        elements.append(Spacer(1, 6))

        elements.append(Paragraph(f"   - <b>Total Cost With Best Package ({best}):</b> {best_pkg_result['total_cost']:.2f} AED", normal_style))
        elements.append(Spacer(1, 12))

        # --- Savings ---
        elements.append(Paragraph(f"<b>3. Total Savings ({best}):</b> {savings:.2f} AED", normal_style))
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
    
    # Initialize the chat interface
    if stage == "welcome":
        st.write("👋 Hi! I'm your AI Banking Assistant. Do you make domestic transfers?")
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
        st.write("Enter number of domestic transfers:")
        count = st.number_input("Count", min_value=0, step=1, key="domestic_count")
        if st.button("Continue", key="submit_domestic_count"):
            data["domestic"]["count"] = count
            st.session_state.chat_stage = "domestic_cost"
            st.rerun()
            
    elif stage == "domestic_cost":
        st.write("Enter cost per domestic transfer (in AED):")
        cost = st.number_input("Cost", min_value=0.0, step=0.1, key="domestic_cost")
        if st.button("Continue", key="submit_domestic_cost"):
            data["domestic"]["cost"] = cost
            st.session_state.chat_stage = "international_ask"
            st.rerun()
            
    elif stage == "international_ask":
        st.write("Do you make international transfers?")
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
        st.write("Enter number of international transfers:")
        count = st.number_input("Count", min_value=0, step=1, key="international_count")
        if st.button("Continue", key="submit_international_count"):
            data["international"]["count"] = count
            st.session_state.chat_stage = "international_cost"
            st.rerun()
            
    elif stage == "international_cost":
        st.write("Enter cost per international transfer (in AED):")
        cost = st.number_input("Cost", min_value=0.0, step=0.1, key="international_cost")
        if st.button("Continue", key="submit_international_cost"):
            data["international"]["cost"] = cost
            st.session_state.chat_stage = "cheque_ask"
            st.rerun()
            
    elif stage == "cheque_ask":
        st.write("Do you process cheques?")
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
        st.write("Enter number of cheques:")
        count = st.number_input("Count", min_value=0, step=1, key="cheque_count")
        if st.button("Continue", key="submit_cheque_count"):
            data["cheque"]["count"] = count
            st.session_state.chat_stage = "cheque_cost"
            st.rerun()
            
    elif stage == "cheque_cost":
        st.write("Enter cost per cheque (in AED):")
        cost = st.number_input("Cost", min_value=0.0, step=0.1, key="cheque_cost")
        if st.button("Continue", key="submit_cheque_cost"):
            data["cheque"]["cost"] = cost
            st.session_state.chat_stage = "fx_ask"
            st.rerun()
            
    elif stage == "fx_ask":
        st.write("Do you need foreign exchange (FX)?")
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
        st.write("Enter FX amount in USD:")
        amount = st.number_input("Amount", min_value=0.0, step=100.0, key="fx_amount")
        if st.button("Continue", key="submit_fx_amount"):
            data["fx"]["amount"] = amount
            st.session_state.chat_stage = "fx_direction"
            st.rerun()
            
    elif stage == "fx_direction":
        st.write("Are you buying or selling USD?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Buy USD", key="buy_usd"):
                data["fx"]["direction"] = "Buy USD"
                data["fx"]["rate"] = 3.67
                st.session_state.chat_stage = "wps_ask"
                st.rerun()
        with col2:
            if st.button("Sell USD", key="sell_usd"):
                data["fx"]["direction"] = "Sell USD"
                data["fx"]["rate"] = 3.63
                st.session_state.chat_stage = "wps_ask"
                st.rerun()
                
    elif stage == "wps_ask":
        st.write("Do you use WPS/CST (Wages Protection System/Corporate Self Transfer)?")
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
        st.write("Enter monthly WPS/CST cost (in AED):")
        cost = st.number_input("WPS/CST Cost", min_value=0.0, step=10.0, key="wps_cost_ai")
        if st.button("Continue", key="submit_wps_cost_ai"):
            data["wps"]["enabled"] = True
            data["wps"]["cost"] = cost
            st.session_state.chat_stage = "pdc_ask"
            st.rerun()

    # New stages for PDC
    elif stage == "pdc_ask":
        st.write("Do you process Post-Dated Cheques (PDCs)?")
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
        st.write("Enter number of PDCs processed monthly:")
        count = st.number_input("PDC Count", min_value=0, step=1, key="pdc_count_ai")
        if st.button("Continue", key="submit_pdc_count_ai"):
            data["pdc"]["count"] = count
            st.session_state.chat_stage = "pdc_cost"
            st.rerun()

    elif stage == "pdc_cost":
        st.write("Enter cost per PDC (in AED):")
        cost = st.number_input("Cost per PDC", min_value=0.0, step=0.1, key="pdc_cost_ai_item")
        if st.button("Continue", key="submit_pdc_cost_ai_item"):
            data["pdc"]["cost"] = cost
            st.session_state.chat_stage = "inward_fcy_ask"
            st.rerun()

    # New stages for Inward FCY Remittance
    elif stage == "inward_fcy_ask":
        st.write("Do you receive Inward FCY Remittances?")
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
        st.write("Enter number of Inward FCY Remittances monthly:")
        count = st.number_input("Inward FCY Remittance Count", min_value=0, step=1, key="inward_fcy_count_ai")
        if st.button("Continue", key="submit_inward_fcy_count_ai"):
            data["inward_fcy_remittance"]["count"] = count
            st.session_state.chat_stage = "inward_fcy_cost"
            st.rerun()

    elif stage == "inward_fcy_cost":
        st.write("Enter cost per Inward FCY Remittance (in AED):")
        cost = st.number_input("Cost per Inward FCY Remittance", min_value=0.0, step=0.1, key="inward_fcy_cost_ai_item")
        if st.button("Continue", key="submit_inward_fcy_cost_ai_item"):
            data["inward_fcy_remittance"]["cost"] = cost
            st.session_state.chat_stage = "other_costs_ask"
            st.rerun()

    # New stage for Other Costs
    elif stage == "other_costs_ask":
        st.write("Do you have any other monthly costs (e.g., cheque submission, courier, labour, miscellaneous)?")
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
        st.write("Enter total of these other monthly costs (in AED):")
        other_total_cost = st.number_input("Total Other Costs", min_value=0.0, step=1.0, key="other_costs_input_ai")
        if st.button("View Analysis", key="submit_other_costs_ai"):
            data["other_costs_input"] = other_total_cost
            analysis_successful = generate_analysis()
            st.session_state.chat_stage = "analysis" if analysis_successful else "no_savings_found"
            st.rerun()

    elif stage == "analysis":
        st.success("Analysis Complete! View the results in the main panel.")
        # Optionally, add a button here to start a new AI chat analysis
        if st.button("Start New AI Analysis", key="new_ai_analysis_from_success"):
            init_chat_state() # Reset AI chat state
            st.session_state.submitted = False # Ensure manual results are not shown
            st.session_state.show_welcome = True # Show welcome screen
            if 'analysis_results' in st.session_state: # Clean up previous results
                del st.session_state.analysis_results
            st.rerun()

    elif stage == "no_savings_found":
        st.warning("Based on your inputs, no package offers savings over not using one.")
        st.info("You can adjust your inputs or try the Manual mode.")
        if st.button("Try Again with AI", key="try_again_ai"):
            init_chat_state() # Reset AI chat state
            st.session_state.submitted = False
            st.session_state.show_welcome = True
            if 'analysis_results' in st.session_state:
                del st.session_state.analysis_results
            st.rerun()
    
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
        st.session_state.analysis_results = {
            "best": best,
            "savings": savings,
            "results": results,
            "no_pkg_cost": no_pkg_cost,
            "user_data": user_data_for_main_display,
            "tx": tx,
            "tx_cost": tx_cost
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
if "chat_stage" not in st.session_state: # Ensure AI chat state is initialized if needed early
    init_chat_state() # This might be redundant if AI mode button calls it, but safe for direct access cases

# -------------------- UI RENDER STARTS HERE --------------------

# Main content area title
st.title(APP_TITLE) 

# Sidebar UI
with st.sidebar:
    # Display a smaller version of the image and title in the sidebar
    try:
        st.image(SIDEBAR_IMAGE_PATH, width=75) 
    except Exception as e:
        st.warning(f"Sidebar image not found at {SIDEBAR_IMAGE_PATH}. Please place it there. Error: {e}")
    
    # Styled App Title in Sidebar
    st.markdown(f"<h2 style='text-align: center; color: #e4002b; font-weight: bold;'>{APP_TITLE}</h2>", unsafe_allow_html=True)
    
    # Styled Subtitle
    st.markdown("**_Your AI-powered cost optimization assistant._**") 
    
    st.markdown("---")

    st.markdown("### 🧭 Choose Input Mode")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📝 Manual Mode", key="mode_manual", use_container_width=True, help="Enter inputs manually"):
            st.session_state.input_mode = 'Manual'
            st.session_state.submitted = False
            st.session_state.show_welcome = True
            if 'chat_stage' in st.session_state:
                del st.session_state.chat_stage
            if 'messages' in st.session_state:
                del st.session_state.messages
    with col2:
        if st.button("🤖 AI Assistant", key="mode_ai", use_container_width=True, help="Chat with AI to analyze your needs"):
            st.session_state.input_mode = 'AI Assistant'
            st.session_state.submitted = False
            st.session_state.show_welcome = True
            init_chat_state()

    st.markdown("---")

    # Manual Form in Sidebar
    if st.session_state.input_mode == "Manual":
        
        def wps_checkbox_callback():
            # This function will be called when the checkbox state changes
            # It just needs to exist for on_change to trigger a rerun with the new widget value
            pass

        st.markdown("### 📝 Transaction Details")
        
        st.markdown("#### 🌍 International Transfers")
        int_count = st.number_input("Count", 0, key="int_count_manual_form") # Added unique keys
        int_cost = st.number_input("Cost (AED)", 0.0, step=0.1, key="int_cost_manual_form")
        
        st.markdown("#### 🏠 Domestic Transfers")
        dom_count = st.number_input("Count", 0, key="dom_count_manual_form")
        dom_cost = st.number_input("Cost (AED)", 0.0, step=0.1, key="dom_cost_manual_form")
        
        st.markdown("#### 📑 Cheques")
        chq_count = st.number_input("Count", 0, key="chq_count_manual_form")
        chq_cost = st.number_input("Cost (AED)", 0.0, step=0.1, key="chq_cost_manual_form")
        
        st.markdown("#### 💱 Foreign Exchange")
        fx_direction = st.radio("Direction", ["Buy USD", "Sell USD"], key="fx_direction_manual_form")
        fx_amount = st.number_input("Amount (USD)", 0.0, key="fx_amount_manual_form")
        
        if fx_direction == "Buy USD":
            fx_rate = st.number_input("Buy Rate (AED/USD)", value=3.67, min_value=0.0, step=0.0001, format="%.4f", key="fx_buy_rate_manual_form")
        else:
            fx_rate = st.number_input("Sell Rate (AED/USD)", value=3.63, min_value=0.0, step=0.0001, format="%.4f", key="fx_sell_rate_manual_form")
        
        st.markdown("#### 💸 WPS/CST")
        # Use on_change to update session state and trigger a rerun
        st.session_state.manual_wps_enabled = st.checkbox(
            "Enable WPS/CST?", 
            value=st.session_state.manual_wps_enabled, # Control component with session state
            key="manual_form_wps_enabled_chkbx_onchange",
            on_change=wps_checkbox_callback 
        )
        
        manual_wps_cost_val = 0.0 
        if st.session_state.manual_wps_enabled:
            manual_wps_cost_val = st.number_input("WPS/CST Cost (AED)", value=0.0, min_value=0.0, key="manual_form_wps_cost_input_field_onchange")

        st.markdown("#### :memo: PDC Processing") 
        pdc_count = st.number_input("PDC Count", 0, key="pdc_count_manual_form")
        pdc_cost_per_item = st.number_input("Cost per PDC (AED)", 0.0, step=0.1, key="pdc_cost_manual_form")

        st.markdown("#### :inbox_tray: Inward FCY Remittance") 
        inward_fcy_count = st.number_input("Inward FCY Remittance Count", 0, key="inward_fcy_count_manual_form")
        inward_fcy_cost_per_item = st.number_input("Cost per Inward FCY Remittance (AED)", 0.0, step=0.1, key="inward_fcy_cost_manual_form")

        st.markdown("#### :receipt: Other Costs") 
        other_costs_total = st.number_input("Total Other Monthly Costs (AED)", 0.0, help="Cheque submission, courier, labour, miscellaneous costs", key="other_costs_manual_form")
        
        # Using a regular button now, not inside a form for this part
        if st.button("🔍 Analyze", use_container_width=True, key="manual_analyze_button"):
            st.session_state.submitted = True
            st.session_state.show_welcome = False
            user_data = {
                "int_count": int_count,
                "int_cost": int_cost,
                "dom_count": dom_count,
                "dom_cost": dom_cost,
                "chq_count": chq_count,
                "chq_cost": chq_cost,
                "fx_amount": fx_amount,
                "fx_direction": fx_direction,
                "client_fx_rate": fx_rate,
                "wps_enabled": st.session_state.manual_wps_enabled, # Get from session state
                "wps_cost": manual_wps_cost_val, 
                "pdc_count": pdc_count,
                "pdc_cost": pdc_cost_per_item,
                "inward_fcy_count": inward_fcy_count,
                "inward_fcy_cost": inward_fcy_cost_per_item,
                "other_costs_input": other_costs_total
            }
            
            tx = {
                "international": user_data["int_count"],
                "domestic": user_data["dom_count"],
                "cheque": user_data["chq_count"],
                "pdc": user_data["pdc_count"],
                "inward_fcy_remittance": user_data["inward_fcy_count"]
            }
            tx_cost = {
                "international": user_data["int_cost"],
                "domestic": user_data["dom_cost"],
                "cheque": user_data["chq_cost"],
                "pdc": user_data["pdc_cost"],
                "inward_fcy_remittance": user_data["inward_fcy_cost"]
            }
            
            best, savings, results, no_pkg_cost = suggest_best_package(
                tx, tx_cost,
                user_data["fx_amount"],
                user_data["fx_direction"],
                user_data["client_fx_rate"],
                user_data["wps_cost"],
                user_data["other_costs_input"]
            )

            if best:
                st.session_state.analysis_results = {
                    "best": best,
                    "savings": savings,
                    "results": results,
                    "no_pkg_cost": no_pkg_cost,
                    "user_data": user_data,
                    "tx": tx,
                    "tx_cost": tx_cost
                }
                st.rerun() # Rerun to update main page with results
            else:
                st.warning("No suitable package found or an error occurred during analysis.")
                if "analysis_results" in st.session_state: # Clear previous results if any
                    del st.session_state.analysis_results
                st.session_state.submitted = False # Allow re-submission or mode change

    # AI Assistant in Sidebar
    elif st.session_state.input_mode == "AI Assistant":
        st.markdown("### 💬 AI Assistant")
        
        # Initialize chat state if needed
        init_chat_state()
        
        # The process_user_response function now handles all UI rendering for the chat
        process_user_response(None) # Pass None as response, buttons handle interaction

    # Reset Button
    if st.session_state.submitted or (st.session_state.input_mode == "AI Assistant" and len(st.session_state.get("messages", [])) > 0):
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.submitted = False
            st.session_state.show_welcome = True
            if "analysis_results" in st.session_state:
                del st.session_state.analysis_results
            # For AI Assistant mode, reset its specific state
            if st.session_state.input_mode == "AI Assistant":
                init_chat_state() # This will reset chat_stage and transaction_data for AI
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
    
    fx_rate_used = packages[best]["fx_buy_rate"] if user_data["fx_direction"] == "Buy USD" else packages[best]["fx_sell_rate"]
    
    # Build the detailed calculation string
    md_lines = []
    md_lines.append("### 🔢 **Calculation Details**")
    md_lines.append(f"Here's how choosing **{best}** saves you **{savings:.2f} AED**:")
    md_lines.append("\n1. **Transaction Costs Without Any Package**:")
    md_lines.append(f"   - International: {tx['international']} × {tx_cost['international']:.2f} = {tx['international'] * tx_cost['international']:.2f} AED")
    md_lines.append(f"   - Domestic: {tx['domestic']} × {tx_cost['domestic']:.2f} = {tx['domestic'] * tx_cost['domestic']:.2f} AED")
    md_lines.append(f"   - Cheque: {tx['cheque']} × {tx_cost['cheque']:.2f} = {tx['cheque'] * tx_cost['cheque']:.2f} AED")
    md_lines.append(f"   - PDC: {user_data['pdc_count']} × {user_data['pdc_cost']:.2f} = {user_data['pdc_count'] * user_data['pdc_cost']:.2f} AED")
    md_lines.append(f"   - Inward FCY Remittance: {user_data['inward_fcy_count']} × {user_data['inward_fcy_cost']:.2f} = {user_data['inward_fcy_count'] * user_data['inward_fcy_cost']:.2f} AED")
    total_no_pkg_base_txn_costs = tx['international'] * tx_cost['international'] + \
                                  tx['domestic'] * tx_cost['domestic'] + \
                                  tx['cheque'] * tx_cost['cheque'] + \
                                  user_data['pdc_count'] * user_data['pdc_cost'] + \
                                  user_data['inward_fcy_count'] * user_data['inward_fcy_cost']
    md_lines.append(f"   - **Total Base Transaction & Service Costs**: {total_no_pkg_base_txn_costs:.2f} AED")
    
    md_lines.append("\n2. **FX Impact Without Any Package**:")
    md_lines.append(f"   - Direction: {user_data['fx_direction']}")
    md_lines.append(f"   - Client's Rate: {user_data['client_fx_rate']:.4f} AED/USD")
    md_lines.append(f"   - FX Amount: {user_data['fx_amount']:.2f} USD")
    if user_data["fx_direction"] == "Buy USD":
        md_lines.append(f"   - Resulting FX Cost (Buying USD): {user_data['fx_amount'] * user_data['client_fx_rate']:.2f} AED")
    else: # Sell USD
        md_lines.append(f"   - Resulting FX Proceeds (Selling USD at Client's Rate): {user_data['fx_amount'] * user_data['client_fx_rate']:.2f} AED")

    md_lines.append("\n3. **WPS/CST Cost**:")
    md_lines.append(f"   - WPS/CST Enabled: {'Yes' if user_data['wps_enabled'] else 'No'}")
    md_lines.append(f"   - WPS/CST Cost: {user_data['wps_cost']:.2f} AED")
    
    md_lines.append("\n4. **Other Costs (User Input - No Package)**:")
    md_lines.append(f"   - Other Costs: {user_data['other_costs_input']:.2f} AED")

    md_lines.append("\n5. **Total Cost Without Any Package**:")
    md_lines.append(f"   - Total Cost: {no_pkg_cost:.2f} AED")
    
    md_lines.append(f"\n6. **Cost With Best Package ({best})**:")
    md_lines.append(f"   - Package Cost: {packages[best]['cost']:.2f} AED")
    md_lines.append("   - Remaining Transactions & Services (Paid): ")

    # Detailed breakdown for paid transactions based on new package structure
    pkg_rules = packages[best]
    for item_type in ["international", "domestic", "cheque", "pdc", "inward_fcy_remittance"]:
        user_item_count = 0
        client_item_rate = 0.0
        # Get count and cost from appropriate user_data keys
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
        else: # pdc, inward_fcy_remittance are direct keys
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
        
        md_lines.append(f"     - {item_type.replace('_', ' ').title()}: {paid_count} × {rate_applied:.2f} = {cost_of_paid:.2f} AED")
        
    md_lines.append(f"\n   - FX Impact with Package ({best}):")
    md_lines.append(f"     - Direction: {user_data['fx_direction']}")
    md_lines.append(f"     - Package Rate: {fx_rate_used:.4f} AED/USD")
    md_lines.append(f"     - FX Amount: {user_data['fx_amount']:.2f} USD")
    if user_data["fx_direction"] == "Buy USD":
        md_lines.append(f"     - Resulting FX Cost (Buying USD): {user_data['fx_amount'] * fx_rate_used:.2f} AED")
    else: # Sell USD
        pkg_fx_display_value = user_data['fx_amount'] * fx_rate_used
        md_lines.append(f"     - Resulting FX Proceeds (Selling USD at Package Rate): {pkg_fx_display_value:.2f} AED")
        fx_gain_from_package_rate = pkg_fx_display_value - (user_data['fx_amount'] * user_data['client_fx_rate'])
        md_lines.append(f"     - Additional gain from package rate vs client's rate: {fx_gain_from_package_rate:.2f} AED")

    md_lines.append(f"\n   - WPS/CST Cost: {user_data['wps_cost']:.2f} AED")
    
    # Other Costs with Package
    md_lines.append(f"\n   - Other Costs (User Input) with Package ({best}):")
    if packages[best].get("other_costs_apply_input", False):
        md_lines.append(f"     - Other Costs Added: {user_data['other_costs_input']:.2f} AED")
    else:
        md_lines.append(f"     - Other Costs Included/Free with Package: 0.00 AED")

    md_lines.append(f"   - **Total Cost With Best Package**: {results['results'][best]['total_cost']:.2f} AED")
    
    md_lines.append("\n7. **Savings**:")
    md_lines.append(f"   - Total Cost Without Package - Total Cost With Best Package = **{savings:.2f} AED")

    md_lines.append("\n8. **Complimentary Items with this Package**:")
    complimentary = packages[best].get("complimentary_items", [])
    if complimentary:
        for item in complimentary:
            md_lines.append(f"   - {item}")
    else:
        md_lines.append("   - None listed.")
    
    calculation_details_md = "\n".join(md_lines)

    with st.expander("🔍 How We Calculated Your Savings", expanded=False):
        st.markdown(calculation_details_md, unsafe_allow_html=True) # Added unsafe_allow_html for potential HTML in future

    # Graphs and export options
    if best:
        # Graph 1: Cost Comparison
        df_cost = pd.DataFrame({
            "Category": ["Without Package"] + list(results["results"].keys()),
            "Cost (AED)": [no_pkg_cost] + [results["results"][pkg]["total_cost"] for pkg in results["results"]],
        })
        fig_cost = px.express.bar(df_cost, x="Category", y="Cost (AED)", title="💰 Total Cost by Option",
                          color="Category", text_auto=".2f")
        st.plotly_chart(fig_cost, use_container_width=True)

        # Graph 2: Savings Comparison
        df_savings = pd.DataFrame({
            "Package": list(results["results"].keys()),
            "Savings (AED)": [results["results"][pkg]["savings"] for pkg in results["results"]],
        })
        fig_savings = px.express.bar(df_savings, x="Package", y="Savings (AED)", title="💸 Savings by Package",
                             color="Package", text_auto=".2f")
        st.plotly_chart(fig_savings, use_container_width=True)

        # Export Options
        st.markdown("### 📤 Export Results")
        col1, col2 = st.columns(2)
        with col1:
            csv = export_to_csv(results["results"], user_data, best, savings, no_pkg_cost)
            st.download_button("📥 Export to CSV", data=csv, file_name="package_comparison.csv", mime="text/csv")
        with col2:
            pdf_bytes = export_to_pdf(results["results"], user_data, best, savings, no_pkg_cost)
            st.download_button("📄 Export to PDF", data=pdf_bytes, file_name="package_comparison.pdf", mime="application/pdf")
