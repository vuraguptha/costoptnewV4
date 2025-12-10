import pandas as pd
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from config import packages

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
