import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from scipy.stats import norm

# Black-Scholes formulas for European options:
# For a European call option:
# C(S, t) = S * N(d1) - K * exp(-r * (T - t)) * N(d2)
# where:
# d1 = [ln(S/K) + (r + 0.5 * sigma^2) * (T - t)] / (sigma * sqrt(T - t))
# d2 = d1 - sigma * sqrt(T - t)
#
# For a European put option:
# P(S, t) = K * exp(-r * (T - t)) * N(-d2) - S * N(-d1)

# Initialize storage for lines to be able to modify their visibility later
lines = []
bs_curve_lines = []  # For storing Black-Scholes price curves

# Toggle visibility of all lines except the total portfolio payoff
def hide_lines(event):
    is_any_visible = any(line.get_visible() for line in lines[:-1])
    for line in lines[:-1]:
        line.set_visible(not is_any_visible)
    plt.draw()

# Toggle visibility of BS price curves only, keeping dots visible
def hide_bs_curves(event):
    is_any_visible = any(line.get_visible() for line in bs_curve_lines)
    for line in bs_curve_lines:
        line.set_visible(not is_any_visible)
    plt.draw()

# Black-Scholes formula for a European call option
def black_scholes_call_price(S, K, r, sigma, T):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price, d1, d2

# Black-Scholes formula for a European put option
def black_scholes_put_price(S, K, r, sigma, T):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price, d1, d2

# Function to calculate and plot derivative portfolio payoff, BS call and put price curves, and BS price points
def plot_payoff(stock_positions, r, sigma, T, S_given):
    global lines, bs_curve_lines
    
    # Create a new figure for each plot
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(left=0.1, right=0.6, bottom=0.3)  # Adjusted for space on the right for text and sliders

    # Create an array for underlying stock prices, starting from 0
    S = np.linspace(0.01, 200, 1000)

    # Initialize total payoff
    total_payoff = np.zeros_like(S)
    
    # Loop through each position and calculate payoff
    new_lines = []
    for position in stock_positions:
        type_ = position['type']
        quantity = position['quantity']
        strike = position.get('strike', 0)
        
        if type_ == 'stock':
            payoff = quantity * S
        elif type_ == 'call':
            payoff = quantity * np.maximum(S - strike, 0)
        elif type_ == 'put':
            payoff = quantity * np.maximum(strike - S, 0)
        elif type_ == 'bond':
            payoff = quantity * np.ones_like(S) * strike * np.exp(-r * T)
        else:
            raise ValueError("Invalid position type. Must be 'stock', 'call', 'put', or 'bond'.")
        
        total_payoff += payoff
        line, = ax.plot(S, payoff, '--', label=f'{type_.capitalize()} Payoff (Qty: {quantity}, Strike: {strike})')
        new_lines.append(line)
    
    # Plot total payoff
    total_line, = ax.plot(S, total_payoff, label='Total Portfolio Payoff', linewidth=2)
    new_lines.append(total_line)
    
    # Store the substituted formula strings for later display
    substituted_formulas = []

    # Add general Black-Scholes formulas at the top of the text block
    general_formulas = (
        "General BS Call: C(S,t) = S * N(d1) - K * exp(-r * (T - t)) * N(d2)\n"
        "General BS Put: P(S,t) = K * exp(-r * (T - t)) * N(-d2) - S * N(-d1)\n"
        "d1 = [ln(S/K) + (r + 0.5 * sigma^2) * (T - t)] / (sigma * sqrt(T - t))\n"
        "d2 = d1 - sigma * sqrt(T - t)"
    )
    substituted_formulas.append(general_formulas)
    
    # Function to plot BS call and put price curves and points for the given S_0 value
    def plot_bs_prices(S_0):
        # Loop through each position to plot BS call and put price curves and points
        for position in stock_positions:
            if position['type'] == 'call':
                strike = position['strike']
                
                # Calculate the full Black-Scholes call price curve
                bs_call_prices = [black_scholes_call_price(s, strike, r, sigma, T)[0] for s in S]
                bs_call_curve, = ax.plot(S, bs_call_prices, 'r-', label=f'BS Call Price Curve (K={strike})')
                bs_curve_lines.append(bs_call_curve)  # Add to curve list
                
                # Calculate BS price at S_0 for call option and plot it
                bs_price, d1, d2 = black_scholes_call_price(S_0, strike, r, sigma, T)
                ax.plot(S_0, bs_price, 'ro', label=f'BS Call Price at S={S_0}, K={strike}: {bs_price:.2f}')
                
                formula_str = (
                    f"Call: S={S_0}, K={strike}, r={r}, sigma={sigma}, T={T}\n"
                    f"d1 = [ln({S_0}/{strike}) + ({r} + 0.5 * {sigma}^2) * {T}] / ({sigma} * sqrt({T})) = {d1:.2f}\n"
                    f"d2 = {d1:.2f} - {sigma} * sqrt({T}) = {d2:.2f}\n"
                    f"BS Call Price = {S_0} * N({d1:.2f}) - {strike} * exp(-{r} * {T}) * N({d2:.2f}) = {bs_price:.2f}"
                )
                substituted_formulas.append(formula_str)

            elif position['type'] == 'put':
                strike = position['strike']
                
                # Calculate the full Black-Scholes put price curve
                bs_put_prices = [black_scholes_put_price(s, strike, r, sigma, T)[0] for s in S]
                bs_put_curve, = ax.plot(S, bs_put_prices, 'b-', label=f'BS Put Price Curve (K={strike})')
                bs_curve_lines.append(bs_put_curve)  # Add to curve list
                
                # Calculate BS price at S_0 for put option and plot it
                bs_price, d1, d2 = black_scholes_put_price(S_0, strike, r, sigma, T)
                ax.plot(S_0, bs_price, 'bo', label=f'BS Put Price at S={S_0}, K={strike}: {bs_price:.2f}')
                
                formula_str = (
                    f"Put: S={S_0}, K={strike}, r={r}, sigma={sigma}, T={T}\n"
                    f"d1 = [ln({S_0}/{strike}) + ({r} + 0.5 * {sigma}^2) * {T}] / ({sigma} * sqrt({T})) = {d1:.2f}\n"
                    f"d2 = {d1:.2f} - {sigma} * sqrt({T}) = {d2:.2f}\n"
                    f"BS Put Price = {strike} * exp(-{r} * {T}) * N(-{d2:.2f}) - {S_0} * N(-{d1:.2f}) = {bs_price:.2f}"
                )
                substituted_formulas.append(formula_str)

        # Redraw the plot and legend after updating
        ax.legend(fontsize='x-small')
        ax.grid()
        ax.set_xlabel('Stock Price at Expiration')
        ax.set_ylabel('Payoff / Option Price')
        ax.set_title(f'Derivative Portfolio Payoff and BS Option Price Point (r={r}, sigma={sigma}, T={T})')
        fig.canvas.draw_idle()

    # Initialize the plot with the default S_given
    plot_bs_prices(S_given)

    # Button for toggling visibility of payoff lines
    ax_button_toggle_lines = plt.axes([0.1, 0.02, 0.1, 0.04])
    button_toggle_lines = Button(ax_button_toggle_lines, 'Toggle Lines')
    button_toggle_lines.on_clicked(hide_lines)

    # Button for toggling visibility of BS price curves
    ax_button_toggle_bs = plt.axes([0.22, 0.02, 0.1, 0.04])
    button_toggle_bs = Button(ax_button_toggle_bs, 'Toggle BS Curves')
    button_toggle_bs.on_clicked(hide_bs_curves)

    # Combine all text into a single block in the top right corner
    full_text = "\n\n".join(substituted_formulas)
    text_size = 7 if len(substituted_formulas) > 3 else 8  # Adjust text size for readability
    fig.text(0.65, 0.7, full_text, fontsize=text_size, va='top', ha='left', wrap=True)

    # Update lines to the new lines
    lines.clear()
    lines.extend(new_lines)

    plt.show()

# Main command loop for user inputs
def main():
    r = float(input("Enter the initial risk-free rate (r): "))
    sigma = float(input("Enter the initial volatility (sigma): "))
    T = float(input("Enter the time to maturity (T): "))
    
    stock_positions = []
    print("\nEnter the details for up to 5 instruments (or type 'done' to finish):")
    for i in range(5):
        position_type = input(f"Enter position type for instrument {i + 1} (stock, call, put, bond) or 'done' to finish: ").lower()
        if position_type == 'done':
            break
        quantity = float(input(f"Enter quantity for {position_type}: "))
        strike = 0
        if position_type in ['call', 'put', 'bond']:
            strike = float(input(f"Enter strike price for {position_type}: "))
        stock_positions.append({'type': position_type, 'quantity': quantity, 'strike': strike})

    while True:
        S_given = float(input("Enter the initial stock price (S) for which to calculate the BS price point: "))
        plot_payoff(stock_positions, r, sigma, T, S_given)

        another = input("\nWould you like to create another plot? (y/n): ").strip().lower()
        if another != 'y':
            print("Exiting the program.")
            break

        keep_values = input("\nWould you like to keep the same values for r, sigma, and T? (y/n): ").strip().lower()
        if keep_values != 'y':
            r = float(input("Enter the new risk-free rate (r): "))
            sigma = float(input("Enter the new volatility (sigma): "))
            T = float(input("Enter the new time to maturity (T): "))

        print("\nEnter the updated strike prices for each instrument (or press Enter to keep the current values):")
        for i, position in enumerate(stock_positions):
            if position['type'] in ['call', 'put', 'bond']:
                new_strike = input(f"Enter new strike price for {position['type']} (current: {position['strike']}): ")
                if new_strike:
                    position['strike'] = float(new_strike)

if __name__ == "__main__":
    main()
