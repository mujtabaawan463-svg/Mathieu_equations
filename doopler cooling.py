import numpy as np
import matplotlib.pyplot as plt

# Define the angle range from 0 to 90 degrees
theta_deg = np.linspace(0, 180, 500)
theta_rad = np.deg2rad(theta_deg)

# Normalized cooling rate Beta/Beta_0 = cos^2(theta)
cooling_rate = np.cos(theta_rad)**2

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(theta_deg, cooling_rate, label=r'$\beta(\theta) \propto \cos^2(\theta)$', color='blue', linewidth=2)

# Highlighting key points
plt.scatter([0, 45, 90], [1, 0.5, 0], color='red')
plt.annotate('Max Cooling (0°)', (0, 1), textcoords="offset points", xytext=(10,0), ha='left')
plt.annotate('45° (50%)', (45, 0.5), textcoords="offset points", xytext=(5,5), ha='left')

plt.annotate('Zero Cooling (90°)', (90, 0), textcoords="offset points", xytext=(-70,10), ha='center')


plt.xlabel('Angle $\\theta$ (degrees)')
plt.ylabel(r'Cooling Rate $\beta / \beta_0$')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

plt.show()

