from config import savings_pack_plus_slabs, USD_SLABS, AED_SLABS



class SavingsInterestCalculator:
    """
    A class to calculate interest rates and amounts based on tiered balance slabs.
    Each slab has a different interest rate applied to the balance within that range.
    Now supports dynamic balance changes over time periods.
    """
    
    def __init__(self, slabs_config):
        """
        Initialize the calculator with tiered slabs configuration.
        
        Args:
            slabs_config (list): List of dictionaries containing slab configurations.
                Each slab should have:
                - 'min_balance': Minimum balance for this slab (inclusive)
                - 'max_balance': Maximum balance for this slab (exclusive, None for unlimited)
                - 'interest_rate': Annual interest rate as percentage (e.g., 2.5 for 2.5%)
                - 'description': Optional description of the slab
        """
        self.slabs = self._validate_and_sort_slabs(slabs_config)
    
    def _validate_and_sort_slabs(self, slabs_config):
        """
        Validate and sort slabs by minimum balance to ensure proper order.
        
        Args:
            slabs_config (list): Raw slabs configuration
            
        Returns:
            list: Validated and sorted slabs
        """
        if not slabs_config:
            raise ValueError("Slabs configuration cannot be empty")
        
        # Validate each slab
        for i, slab in enumerate(slabs_config):
            required_keys = ['min_balance', 'max_balance', 'interest_rate']
            missing_keys = [key for key in required_keys if key not in slab]
            if missing_keys:
                raise ValueError(f"Slab {i} missing required keys: {missing_keys}")
            
            if slab['min_balance'] < 0 or slab['interest_rate'] < 0:
                raise ValueError(f"Slab {i} has negative values")
            
            if slab['max_balance'] is not None and slab['max_balance'] <= slab['min_balance']:
                raise ValueError(f"Slab {i} max_balance must be greater than min_balance")
        
        # Sort by minimum balance
        sorted_slabs = sorted(slabs_config, key=lambda x: x['min_balance'])
        
        # Check for gaps or overlaps
        for i in range(len(sorted_slabs) - 1):
            current_slab = sorted_slabs[i]
            next_slab = sorted_slabs[i + 1]
            
            if current_slab['max_balance'] is not None:
                if current_slab['max_balance'] != next_slab['min_balance']:
                    raise ValueError(f"Gap or overlap between slabs {i} and {i+1}")
        
        return sorted_slabs
    
    def calculate_interest_simple(self, balance_days_dict, compounding_frequency='monthly'):
        """
        Calculate interest for a simple balance-days dictionary.
        
        Args:
            balance_days_dict (dict): Dictionary with balance as key and days as value.
                Example: {25000: 10, 75000: 15, 150000: 20}
            compounding_frequency (str): 'daily', 'monthly', 'quarterly', 'annually'
            
        Returns:
            dict: Detailed interest calculation results
        """
        # Input validation
        if not isinstance(balance_days_dict, dict):
            raise ValueError(f"Input must be a dictionary, got {type(balance_days_dict)}")
        
        if not balance_days_dict:
            raise ValueError("Balance-days dictionary cannot be empty")
        
        # Convert dictionary to the format expected by calculate_interest_dynamic
        balance_periods = []
        for balance, days in balance_days_dict.items():
            # Validate balance and days
            if not isinstance(balance, (int, float)) or not isinstance(days, (int, float)):
                raise ValueError(f"Balance and days must be numbers, got balance={type(balance)}, days={type(days)}")
            
            if balance < 0 or days <= 0:
                raise ValueError(f"Invalid balance ({balance}) or days ({days}) - both must be positive")
            
            # Convert to proper types and create description safely
            balance_float = float(balance)
            days_int = int(days)
            
            balance_periods.append({
                'balance': balance_float,
                'days': days_int,
                'description': f'{balance_float:,.0f} AED for {days_int} days'
            })
        
        try:
            return self.calculate_interest_dynamic(balance_periods, compounding_frequency)
        except Exception as e:
            raise ValueError(f"Error in calculate_interest_dynamic: {str(e)}")
    
    def calculate_interest_dynamic(self, balance_periods, compounding_frequency='monthly'):
        """
        Calculate interest for dynamic balance changes over time periods.
        
        Args:
            balance_periods (list): List of dictionaries containing balance periods.
                Each period should have:
                - 'balance': Account balance for this period
                - 'days': Number of days this balance is maintained
                - 'description': Optional description of the period
            compounding_frequency (str): 'daily', 'monthly', 'quarterly', 'annually'
            
        Returns:
            dict: Detailed interest calculation results
        """
        if not balance_periods:
            raise ValueError("Balance periods cannot be empty")
        
        total_interest = 0
        total_days = 0
        period_breakdown = []
        
        for i, period in enumerate(balance_periods):
            if 'balance' not in period or 'days' not in period:
                raise ValueError(f"Period {i} missing required keys: balance, days")
            
            balance = period['balance']
            days = period['days']
            
            if balance < 0 or days <= 0:
                raise ValueError(f"Period {i} has invalid balance or days")
            
            # Calculate interest for this period
            period_result = self.calculate_interest(balance, days, compounding_frequency)
            
            period_breakdown.append({
                'period_number': i + 1,
                'balance': balance,
                'days': days,
                'interest': period_result['total_interest'],
                'effective_rate': period_result['effective_annual_rate'],
                'description': period.get('description', f'Period {i + 1}'),
                'slab_breakdown': period_result['slab_breakdown']
            })
            
            total_interest += period_result['total_interest']
            total_days += days
        
        # Calculate overall effective annual rate
        total_balance_days = sum(p['balance'] * p['days'] for p in balance_periods)
        overall_effective_rate = (total_interest / total_balance_days * 365 * 100) if total_balance_days > 0 else 0
        
        return {
            'total_interest': total_interest,
            'total_days': total_days,
            'overall_effective_rate': overall_effective_rate,
            'compounding_frequency': compounding_frequency,
            'period_breakdown': period_breakdown,
            'total_balance_days': total_balance_days,
            'average_balance': total_balance_days / total_days if total_days > 0 else 0
        }
    
    def calculate_interest(self, balance, days=365, compounding_frequency='monthly'):
        """
        Calculate interest for a given balance based on the single-tier logic (not slabbed).
        The entire balance gets the rate of the tier in which it falls.
        """
        if balance < 0:
            raise ValueError("Balance cannot be negative")
        
        # Find the slab/tier where the balance falls
        applicable_slab = None
        for slab in self.slabs:
            min_bal = slab['min_balance']
            max_bal = slab['max_balance'] if slab['max_balance'] is not None else float('inf')
            if balance >= min_bal and (max_bal is None or balance < max_bal):
                applicable_slab = slab
                break
        if not applicable_slab:
            raise ValueError("No applicable slab found for the given balance")
        
        annual_rate = applicable_slab['interest_rate'] / 100
        period_ratio = days / 365
        interest = balance * annual_rate * period_ratio
        
        slab_breakdown = [{
            'slab_range': f"{applicable_slab['min_balance']:,.2f} - {applicable_slab['max_balance'] if applicable_slab['max_balance'] is not None else '∞'}",
            'balance_in_slab': balance,
            'interest_rate': applicable_slab['interest_rate'],
            'interest_amount': interest,
            'description': applicable_slab.get('description', '')
        }]
        
        effective_annual_rate = (interest / balance * 365 / days * 100) if balance > 0 else 0
        
        return {
            'total_balance': balance,
            'total_interest': interest,
            'effective_annual_rate': effective_annual_rate,
            'calculation_period_days': days,
            'compounding_frequency': compounding_frequency,
            'slab_breakdown': slab_breakdown,
        }

def analyze_activesaver_benefit(balance_days_dict, package_cost, compounding_frequency='monthly'):
    """
    Analyze how much ActiveSaver interest can offset package costs.
    
    Args:
        balance_days_dict (dict): Dictionary with balance as key and days as value
        package_cost (float): Monthly package cost
        compounding_frequency (str): Compounding frequency
        
    Returns:
        dict: ActiveSaver analysis results
    """
    try:
        # Initialize calculator
        calculator = SavingsInterestCalculator(savings_pack_plus_slabs)
        
        # Calculate interest
        interest_result = calculator.calculate_interest_simple(balance_days_dict, compounding_frequency)
        annual_interest = interest_result['total_interest']
        monthly_interest = annual_interest / 12
        
        # Calculate net cost after interest
        net_monthly_cost = max(0, package_cost - monthly_interest)
        offset_percentage = (monthly_interest / package_cost * 100) if package_cost > 0 else 0
        
        return {
            'package_cost': package_cost,
            'monthly_interest': monthly_interest,
            'annual_interest': annual_interest,
            'net_monthly_cost': net_monthly_cost,
            'offset_percentage': offset_percentage,
            'interest_details': interest_result,
            'balance_days_dict': balance_days_dict,
            'is_fully_covered': monthly_interest >= package_cost
        }
    except Exception as e:
        return {
            'error': str(e),
            'package_cost': package_cost,
            'balance_days_dict': balance_days_dict
        }

def get_activesaver_slabs(currency="AED"):
    """
    Get the appropriate ActiveSaver slabs for the given currency.
    
    Args:
        currency (str): Currency code ("AED" or "USD")
        
    Returns:
        list: Slabs configuration for the currency
    """
    if currency == "AED":
        return AED_SLABS
    elif currency == "USD":
        return USD_SLABS
    else:
        raise ValueError(f"Unsupported currency: {currency}. Supported currencies: AED, USD") 