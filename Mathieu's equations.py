import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parameters
a = 0.2
alpha = -1.1
q = q =0.4
theta = np.pi/4



def system(t, Y):
    x, vx, y, vy = Y
    
    cos_term = np.cos(2*t)
    
    dxdt = vx
    dvxdt = -a*x - 2*q*cos_term
    dydt = vy
    dvydt = alpha*a*y - 2*q*cos_term
    
    return [dxdt, dvxdt, dydt, dvydt]

# Initial conditions
Y0 = [1, 0, 0, 0]

# Time span
t_span = (0, 100)
t_eval = np.linspace(*t_span, 5000)

sol = solve_ivp(system, t_span, Y0, t_eval=t_eval)

# Plot
plt.figure(figsize=(10,6))
plt.plot(sol.t, sol.y[0], label='x(t)')
#plt.plot(sol.t, sol.y[2], label='y(t)')
plt.xlabel('τ')
ax = plt.gca()
ax.spines['bottom'].set_position('zero')
plt.ylabel('Amplitude')
plt.ylim(1, -1)
plt.legend()
plt.title('Solution of Coupled Parametric System')
plt.show()
