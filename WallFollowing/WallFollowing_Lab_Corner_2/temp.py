"""
Author       : Hanqing Qi
Date         : 2023-09-08 18:54:28
LastEditors  : Hanqing Qi
LastEditTime : 2023-09-08 18:54:29
FilePath     : /WallFollowing_Lab_Corner_2/temp.py
Description  : 
"""
import matplotlib.pyplot as plt
# Define the filename
filename = 'record_data.txt'

# Initialize an empty list to store x, y values
data_points = []

# Open the file and read its content
with open(filename, 'r') as file:
    for line in file:
        # Split the line into values using comma as a delimiter
        values = line.split(',')
        
        # Extract x and y values
        x = float(values[1].strip())
        y = float(values[2].strip())
        
        # Append the x, y tuple to the data_points list
        data_points.append((x, y))

fig, ax = plt.subplots()
xdata, ydata = [], []
ax.set_xlim(-2, 4)
ax.set_ylim(-3, 2)
ln, = ax.plot(xdata, ydata, 'r-')

# Initial draw to show the empty plot
plt.draw()
plt.show(block=False)

for i in range(len(data_points)):  # Simulating the receipt of 100 data points
    x, y = data_points[i]
    xdata.append(x)
    ydata.append(y)
    ln.set_data(xdata, ydata)
    
    plt.draw()
    plt.pause(0.001)  # Pause for a short duration before next update

plt.show(block=True)
