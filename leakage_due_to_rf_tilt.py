import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.constants import physical_constants

amu = physical_constants["atomic mass constant"][0]
e_charge = physical_constants["elementary charge"][0]

class MultidimensionalQ:
    def __init__(self, V, m, Omega, r0, theta):
        self.V = V
        self.m = m
        self.Omega = Omega
        self.r0 = r0
        self.theta = theta
        self.q = 2 * e_charge * V / (m * Omega**2 * r0**2)

    def matrix(self):
        return self.q * np.array([[1, 0],
                                  [0, -1]])

class CoupledMathieuSystem:
    def __init__(self, omega_x, alpha, theta, Qmatrix, Omega):
        self.Q = Qmatrix
        self.Omega = Omega
        self.theta = theta
        c = np.cos(2 * self.theta)
        s = np.sin(2 * self.theta)
        q_scalar = np.max(np.abs(Qmatrix))
        
        self.a = (2 * omega_x / Omega)**2 - q_scalar**2 / 2
        

        self.A = self.a * np.array([[alpha * c, s],
                                    [s, -alpha * c]])

    def matrix_mathieu_eq(self, tau, y):
        x = y[:2]
        dx = y[2:]
        M = self.A + 2 * self.Q * np.cos(2 * tau)
        d2x = - M @ x
        return np.concatenate([dx, d2x])

    def integrate(self, state0, tau_span, npts=8000):
        t_eval = np.linspace(tau_span[0], tau_span[1], npts)
        return solve_ivp(self.matrix_mathieu_eq,
                         tau_span,
                         state0,
                         t_eval=t_eval,
                         method="RK45",
                         rtol=1e-8)

class PaulTrapSimulator:
    def __init__(self, V, m, Omega, r0, theta, omega_x, alpha):
        qsys = MultidimensionalQ(V, m, Omega, r0, theta)
        self.q = qsys.q

        print(f"Mathieu q = {self.q:.4f}")
        self.system = CoupledMathieuSystem(
            omega_x,
            alpha,
            theta,
            qsys.matrix(),
            Omega
        )

    def simulate(self, state0, tau_span=(0, 200)):
        sol = self.system.integrate(state0, tau_span)
        return sol.t, sol.y

    def get_leakage(self, x, y):
        return np.max(np.abs(y)) / np.max(np.abs(x))


V = 110                     
m = 171 * amu               
Omega = 2 * np.pi * 50e6    
r0 = 80e-6
theta = 45 

theta_rad = np.deg2rad(theta)
omega_x = 2 * np.pi * 2e6   
alpha = -1.1

state0 = [1.0, 0.0, 0.0, 0.0]


sim = PaulTrapSimulator(V, m, Omega, r0, theta_rad, omega_x, alpha)
tau, sol = sim.simulate(state0)

x, y = sol[0], sol[1]
leakage = sim.get_leakage(x, y)

print(f"Leakage = {leakage:.4f}")


plt.figure(figsize=(10, 5))
plt.plot(tau, x, label="x(τ) - Lab X", alpha=0.8)
plt.plot(tau, y, label="y(τ) - Lab Y", alpha=0.8)
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlabel("Dimensionless Time (τ)")
plt.ylabel("Displacement")
plt.title(f"Mathieu Solution with Principal Axis Rotation\nθ = {theta}°, Leakage = {leakage:.3f}")
plt.show()

theta_list_deg = np.linspace(0, 180, 100)
leakage_list = []

for theta_deg in theta_list_deg:
    theta_rad = np.deg2rad(theta_deg)
    sim = PaulTrapSimulator(V, m, Omega, r0, theta_rad, omega_x, alpha)
    tau, sol = sim.simulate(state0)
    x, y = sol[0], sol[1]
    leakage = sim.get_leakage(x, y)
    leakage_list.append(leakage)


plt.figure(figsize=(8,4))
plt.plot(theta_list_deg, leakage_list, marker='o')
plt.xlabel("Theta (degrees)")
plt.ylabel("Leakage")
plt.title("Leakage vs θ")
plt.grid()
plt.show()
