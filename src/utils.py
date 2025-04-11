import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ======================= Utility Functions =======================

def round_numbers(lst, num_decimal_places=3):
    """Round numbers in a list to a specified number of decimal places."""
    return [round(num, num_decimal_places) for num in lst]

def range_with_floats(start, stop, step):
    """Generate a range of float values."""
    return np.arange(start, stop, step)

def get_formatter(powerlimits=(-4, 4)):
    """Configure a scientific notation formatter for plot axes."""
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits(powerlimits)
    return formatter