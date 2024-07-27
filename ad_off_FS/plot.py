
import matplotlib.pyplot as plt
from preprocess import AD_off, meanRR, maxHR, meanHR, RMSSD, SDHR, SDNN, HF, LF, total_power
import sys

def plot_variable_against_AD_off(selected_feature):

    fig, ax = plt.subplots(figsize=(6, 6))
    
    x_data = eval(selected_feature).to_numpy().flatten()
    y_data = AD_off.to_numpy().flatten()
    
    ax.scatter(x_data, y_data)
    ax.set_xlabel(f'{selected_feature} (unit)')
    ax.set_ylabel('AD_off (s)')
    ax.set_title(f'{selected_feature} vs AD_off')

    plt.show()

def main():
    independent_vars = ['meanRR', 'maxHR', 'meanHR', 'RMSSD', 'SDHR', 'SDNN', 'HF', 'LF', 'total_power']

    print("Available features: ", independent_vars)
    selected_feature = input("Select a feature to plot against AD_off: ")

    if selected_feature not in independent_vars:
        print("Invalid selection.")
        sys.exit(1)
    
    plot_variable_against_AD_off(selected_feature)

if __name__ == '__main__':
    main()
