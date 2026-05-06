import numpy as np
import matplotlib.pyplot as plt


A = 1.0
B = 0.0
Omega_rf = 2*np.pi*1     
t = np.linspace(0, 50, 5000)
def ion_motion(t, q):
    omega_x = np.sqrt(1 + q**2/2 + 5*q**5/128)
    return A * np.cos(omega_x * t + B) * (1 - (q/2)*np.cos(Omega_rf * t))
q_values = [0.2, 0.45, 0.75, 0.908]
for q in q_values:
    x_t = ion_motion(t, q)
    plt.figure()
    plt.plot(t, x_t)
    plt.xlabel("Time (arb. units)")
    plt.ylabel("Position")
    plt.title(f"Ion Motion for q = {q}")
    plt.grid(True)
    plt.show()


