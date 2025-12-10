import os



# -------------------- UI CONFIG & APP NAMING --------------------
APP_TITLE = "Fikra Genie"
APP_SUBTITLE = """Powered by advanced analytics and a deep evaluation of each client's financial profile,<br>it enables data-driven conversations, stronger client alignment, and measurable value creation."""
MAIN_APP_IMAGE_FILENAME = "takhfid_genie_image.png"
ADCB_LOGO_FILENAME = "adcb_logo.png"  # ADCB logo file
WATERMARK_IMAGE_FILENAME = "adcb_watermark.png"

# Directory setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_APP_IMAGE_PATH = os.path.join(SCRIPT_DIR, MAIN_APP_IMAGE_FILENAME)
ADCB_LOGO_PATH = os.path.join(SCRIPT_DIR, ADCB_LOGO_FILENAME)
WATERMARK_IMAGE_PATH = os.path.join(SCRIPT_DIR, WATERMARK_IMAGE_FILENAME)



# -------------------- PACKAGE CONFIG --------------------
packages = {
    "Package Essential": {
        "cost": 275,
        "transactions": {
            "international": {"free_count": 0, "rate_after_free": 30},
            "domestic": {"free_count": 9999, "rate_after_free": None},
            "cheque": {"free_count": 17, "rate_after_free": 1}
        },
        "pdc": {"free_count": 0, "rate_after_free": 25},
        "inward_fcy_remittance": {"free_count": 0, "rate_after_free": 10},
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
        "pdc": {"free_count": 0, "rate_after_free": 25},
        "inward_fcy_remittance": {"free_count": 0, "rate_after_free": 10},
        "other_costs_apply_input": True,
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
    "Package Gold": {
        "cost": 1500,
        "transactions": {
            "international": {"free_count": 75, "rate_after_free": 20},
            "domestic": {"free_count": 9999, "rate_after_free": None}, # Unlimited free, or use client's rate if somehow exceeded
            "cheque": {"free_count": 0, "rate_after_free": 1}
        },
        "pdc": {"free_count": 99999, "rate_after_free": 0},
        "inward_fcy_remittance": {"free_count": 0, "rate_after_free": 10},
        "other_costs_apply_input": True, # User's 'Other Costs' are added to this package's total
        "fx_buy_rate": 3.6760,
        "fx_sell_rate": 3.6700,
        "complimentary_items": ["Free Credit Cards", "Free Debit Cards", "ProCash Soft Token"]
    },
    "Package Platinum": {
        "cost": 2000,
        "transactions": {
            "international": {"free_count": 75, "rate_after_free": 20},
            "domestic": {"free_count": 9999, "rate_after_free": None},
            "cheque": {"free_count": 17, "rate_after_free": 1}
        },
        "pdc": {"free_count": 99999, "rate_after_free": 0}, # Effectively free
        "inward_fcy_remittance": {"free_count": 99999, "rate_after_free": 0}, # Effectively free
        "other_costs_apply_input": False, # User's 'Other Costs' are NOT added (free with package)
        "fx_buy_rate": 3.6740,
        "fx_sell_rate": 3.6710,
        "complimentary_items": ["Free Credit Cards", "Free Debit Cards", "ProCash Soft Token"]
    },
    "Package Platinum plus": {
        "cost": 2500,
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



# -------------------- ACTIVESAVER CONFIG --------------------
# ADCB ActiveSaver tiered interest rates
savings_pack_plus_slabs = [
    {'min_balance': 0, 'max_balance': 50000, 'interest_rate': 0.4, 'description': 'Basic Tier'},
    {'min_balance': 50000, 'max_balance': 200000, 'interest_rate': 1.0, 'description': 'Silver Tier'},
    {'min_balance': 200000, 'max_balance': 2000000, 'interest_rate': 1.75, 'description': 'Gold Tier'},
    {'min_balance': 2000000, 'max_balance': 10000000, 'interest_rate': 2.25, 'description': 'Platinum Tier'},
    {'min_balance': 10000000, 'max_balance': 20000000, 'interest_rate': 2, 'description': 'Diamond Tier'},
    {'min_balance': 20000000, 'max_balance': None, 'interest_rate': 0.2, 'description': 'Elite Tier'}
]

# Define slabs for AED and USD
AED_SLABS = [
    {'min_balance': 0, 'max_balance': 50000, 'interest_rate': 0.4, 'description': 'Basic Tier'},
    {'min_balance': 50000, 'max_balance': 200000, 'interest_rate': 1.0, 'description': 'Silver Tier'},
    {'min_balance': 200000, 'max_balance': 2000000, 'interest_rate': 1.75, 'description': 'Gold Tier'},
    {'min_balance': 2000000, 'max_balance': 10000000, 'interest_rate': 2.25, 'description': 'Platinum Tier'},
    {'min_balance': 10000000, 'max_balance': 20000000, 'interest_rate': 2, 'description': 'Diamond Tier'},
    {'min_balance': 20000000, 'max_balance': None, 'interest_rate': 0.2, 'description': 'Elite Tier'}
]
USD_SLABS = [
    {'min_balance': 0, 'max_balance': 15000, 'interest_rate': 0.2, 'description': 'Basic Tier'},
    {'min_balance': 15000, 'max_balance': 50000, 'interest_rate': 0.6, 'description': 'Silver Tier'},
    {'min_balance': 50000, 'max_balance': 500000, 'interest_rate': 1, 'description': 'Gold Tier'},
    {'min_balance': 500000, 'max_balance': 3000000, 'interest_rate': 1.5, 'description': 'Platinum Tier'},
    {'min_balance': 3000000, 'max_balance': 6000000, 'interest_rate': 2, 'description': 'Diamond Tier'},
    {'min_balance': 6000000, 'max_balance': None, 'interest_rate': 0.2, 'description': 'Elite Tier'}
]

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

# -------------------- CHAT STAGES --------------------
CHAT_STAGES = {
    "welcome": "Hi! I'm your AI Banking Assistant. Is your client currently making domestic transfers as part of their recurring business expenditures??",
    "domestic_count": "Enter number of domestic transfers:",
    "domestic_cost": "Enter cost per domestic transfer, in AED:",
    "international_ask": "Do you make international transfers?",
    "international_count": "Enter number of international transfers:",
    "international_cost": "Enter cost per international transfer, in AED:",
    "cheque_ask": "Do you process cheques?",
    "cheque_count": "Enter number of cheques:",
    "cheque_cost": "Enter cost per cheque, in AED:",
    "fx_ask": "Do you need foreign exchange?",
    "fx_amount": "Enter FX amount in USD:",
    "fx_direction": "Are you buying or selling USD?",
    "fx_rate": "Enter your {direction} rate (AED/USD):",
    "wps_ask": "Do you use WPS or CST (Wages Protection System or Corporate Self Transfer)?",
    "wps_cost": "Enter monthly WPS or CST cost, in AED:",
    "pdc_ask": "Do you process Post-Dated Cheques (PDCs)?",
    "pdc_count": "Enter number of PDCs processed monthly:",
    "pdc_cost": "Enter cost per PDC, in AED:",
    "inward_fcy_ask": "Do you receive Inward FCY Remittances?",
    "inward_fcy_count": "Enter number of Inward FCY Remittances monthly:",
    "inward_fcy_cost": "Enter cost per Inward FCY Remittance, in AED:",
    "other_costs_ask": "Do you have any other monthly costs such as cheque submission, courier, or miscellaneous fees?",
    "other_costs_input": "Enter total of these other monthly costs, in AED:",
    "analysis": "Analysis Complete! View the results in the main panel.",
    "no_savings_found": "Based on your inputs, no package offers savings over not using one. You can adjust your inputs or try the Manual mode."
} 