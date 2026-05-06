import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigvalsh

def get_stability_boundary(q_range, beta_val, size=10):
    """
    Calculates the 'a' parameter boundary for a given 'q' and 'beta'.
    beta=0 is the lower bound, beta=1 is the upper bound.
    """
    a_boundary = []
    n_values = np.arange(-size, size + 1)
    
    for q in q_range:
        diag = (2 * n_values + beta_val)**2
        off_diag = np.ones(2 * size) * q
        matrix = np.diag(diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)
        
        
        evals = eigvalsh(matrix)
        a_boundary.append(np.min(evals))
        
    return np.array(a_boundary)


q_vals = np.linspace(0, 1, 300)
a_lower = get_stability_boundary(q_vals, beta_val=0)
a_upper = get_stability_boundary(q_vals, beta_val=1) 
a_radial_lower = -2 * get_stability_boundary(q_vals/2, beta_val=0)
a_radial_upper = -2 * get_stability_boundary(q_vals/2, beta_val=1)



plt.figure(figsize=(10, 8))
plt.plot(q_vals, a_lower, 'b--', alpha=0.3, label='Z-Stability Boundary')
plt.plot(q_vals, a_upper, 'b--', alpha=0.3)
plt.plot(q_vals, a_radial_lower, 'r--', alpha=0.3, label='R-Stability Boundary')
stability_mask = (a_lower < a_radial_lower)
plt.fill_between(q_vals, a_lower, a_radial_lower, 
                 where=stability_mask, color='blue', alpha=0.6, 
                 label='3D Stable Region (Paul Trap)')

plt.axhline(0, color='black', lw=1)
plt.xlim(0, 1)
plt.ylim(-0.4, 0.4)
plt.xlabel('Parameter $q$', fontsize=12)
plt.ylabel('Parameter $a$', fontsize=12)
plt.title('Ion Trap Stability Diagram ($a$-$q$ Space)', fontsize=14)
plt.legend(loc='upper right')
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()


