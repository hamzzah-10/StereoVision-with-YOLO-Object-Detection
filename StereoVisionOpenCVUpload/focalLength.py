import numpy as np

# Load the projection matrix P1
P1 = np.load('P1.npy')

# Extract focal lengths
f_x = P1[0, 0]  # Focal length in pixels (x-axis)
f_y = P1[1, 1]  # Focal length in pixels (y-axis)

print("Focal length (x-axis):", f_x)
print("Focal length (y-axis):", f_y)
