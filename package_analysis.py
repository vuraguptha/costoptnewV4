import streamlit as st
import openai
import json
from config import packages





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



def generate_narrative_summary(best_pkg, savings, user_data, no_pkg_cost, results):
    """Generates a narrative summary of the analysis using an AI model."""
    if not best_pkg:
        return "No savings were identified with any package based on the provided data."

    # --- COST CALCULATION TO ALIGN WITH BAR CHART ---
    # The bar chart and breakdown cards use the best package's FX cost as a baseline (zero).
    # We must calculate the costs for the narrative using the same logic to ensure consistency.
    best_pkg_fx_cost_baseline = results[best_pkg]["breakdown"].get("Absolute FX Cost", 0)
    
    # Cost for "Without Package" as shown in the bar chart
    no_pkg_display_cost = results["Without Package"]["true_total_cost"] - best_pkg_fx_cost_baseline
    
    # Cost for the best package as shown in the bar chart
    best_pkg_display_cost = results[best_pkg]["true_total_cost"] - best_pkg_fx_cost_baseline


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
            "Chart Cost without any package": f"{no_pkg_display_cost:,.0f} AED",
            "Recommended Package": best_pkg,
            "Chart Cost with this package": f"{best_pkg_display_cost:,.0f} AED",
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
        "VERY IMPORTANT: The costs you mention (e.g., 'Chart Cost without any package') are the same as those displayed in the bar chart and breakdown cards. They represent fees and marginal costs, with the best package's FX rate as a baseline. Do not refer to them as total costs. "
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
