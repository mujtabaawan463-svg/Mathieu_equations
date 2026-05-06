import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigvalsh
from scipy.interpolate import interp1d


def get_true_beta(q_target, a_target=0):
    """ Finds the exact beta for a given q using matrix eigenvalues. we do this by guessing beta and 
    then inter1d function with a to guess a corresponds to that beta for which it has stable solution """
    size = 4  
    betas = np.linspace(0, 1, 500) 
    a_results = []

    for b in betas:
        n_values = np.arange(-size, size + 1)
        diag = (2 * n_values + b)**2
        off_diag = np.ones(2 * size) * q_target
        matrix = np.diag(diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)
        a_results.append(np.min(eigvalsh(matrix)))

 
    f_interp = interp1d(a_results, betas, kind='cubic')
    return float(f_interp(a_target))


q_axis = np.linspace(0.01, 0.8, 50)
truth = [get_true_beta(q)**2 for q in q_axis] 

First_order_approx = 0.5 * q_axis**2
second_order_approx  = 0.5 * q_axis**2 + (25/128) * q_axis**4  



plt.figure(figsize=(10, 6))
First_order_approx_error = np.abs(truth - First_order_approx)
print(f"Max error in First_order_approx: {np.max(First_order_approx_error):.6f}")
Second_order_approx_error = np.abs(truth - second_order_approx)
print(f"Max error in Second_order_approx_error: {np.max(Second_order_approx_error):.6f}")


plt.plot(q_axis, truth, 'k-', lw=2, label='Numerical Truth (Hill Matrix)')
plt.plot(q_axis, First_order_approx, 'r--', label='First_order_approx')
plt.plot(q_axis, second_order_approx, 'g:', label='second_order_approx')


plt.xlabel('q parameter', fontsize=16)
plt.ylabel(r'$\beta^2$', fontsize=16)
#plt.title('Comparison of Mathieu Equation Approximations', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()