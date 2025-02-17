import csv
import matplotlib.pyplot as plt
import numpy as np

def read_csv(file_path):
    x_axis = []
    y_axis = []
    print('hello')
    try:
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            for i, row in enumerate(reader):
                if i < 21:  # Skip the first 21 lines
                    continue
                print(f"Line {i+1}: {row}")  # Debugging statement
                x_axis.append(float(row[0]))  # First column for x-axis
                y_axis.append(float(row[1]))  # Second column for y-axis
    except Exception as e:
        print(f"Error reading file: {e}")
    return x_axis, y_axis

def find_x_at_y(x_axis, y_axis, y_value):
    x_axis = np.array(x_axis)
    y_axis = np.array(y_axis)
    return np.interp(y_value, y_axis, x_axis)

# Example usage
file_path = '.\\250130Filter.csv'
x_axis, y_axis = read_csv(file_path)
#print("X-axis:", x_axis)
#print("Y-axis:", y_axis)

# Plotting the data
plt.plot(x_axis, y_axis)
plt.xscale('log')  # Set x-axis to logarithmic scale
plt.xlabel('Frekvens [Hz]')
plt.ylabel('Magnitude [dB]')
plt.title('Lavpassfilterets frekvensrespons')
plt.grid()

# Add vertical line at x=23
plt.axvline(x=23, color='r', linestyle='--', label='x=23Hz')

# Add horizontal line at y=-3
plt.axhline(y=-3, color='b', linestyle='--', label='y=-3dB')

plt.legend()
plt.show()

# Find x-value at a given y-value
y_value = -3  # Example y-value
x_value_at_y = find_x_at_y(x_axis, y_axis, y_value)
print(f"The x-value at y = {y_value} is {x_value_at_y}")
