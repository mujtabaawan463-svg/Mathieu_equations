import bempp_cl.api as bp
import numpy as np
import matplotlib.pyplot as plt

class DCPotentialSolver:
    def __init__(self, msh_file):
        self.msh_file = msh_file
        self.grid = None
        self.space = None
        self.sigmas = []          
        self.dc_potentials = []   
        self.tags = None
        self.import_grid()

    def import_grid(self):
        """Loads the grid from the msh file."""
        print(f"Loading mesh from {self.msh_file}...")
        self.grid = bp.import_grid(self.msh_file)
        
        print(f"Grid loaded with {self.grid.number_of_elements} elements")
        print(f"Number of vertices: {self.grid.number_of_vertices}")
        
        # Get domain indices (electrode tags)
        if hasattr(self.grid, 'domain_indices'):
            self.tags = np.unique(self.grid.domain_indices)
            print(f"Found electrode tags: {self.tags}")
        else:
            # If no domain indices, create a single tag for entire mesh
            print("Warning: No domain indices found. Treating entire mesh as one electrode.")
            self.tags = np.array([1])
        
        print(f"Number of unique electrode tags: {len(self.tags)}")

    def solve_all_electrodes(self, V=1.0):
        """
        Solves the Laplace equation for each electrode tag individually.
        """
        if self.grid is None:
            raise RuntimeError("Grid not loaded.")
        
        print(f"\nBuilding basis functions for {len(self.tags)} electrodes...")
        print(f"Each solved with {V}V")
        
        # Use DP0 space (piecewise constant) for charge density
        self.space = bp.function_space(self.grid, "DP", 0)
        print(f"Function space created with {self.space.dofs} degrees of freedom")
        
        self.sigmas = []
        self.electrode_tags = []
        
        # Pre-assemble operators (more efficient)
        print("Assembling boundary operators...")
        slp_op = bp.operators.boundary.laplace.single_layer(
            self.space, self.space, self.space, assembler="dense"
        )
        dlp_op = bp.operators.boundary.laplace.double_layer(
            self.space, self.space, self.space, assembler="dense"
        )
        identity_op = bp.operators.boundary.sparse.identity(
            self.space, self.space, self.space
        )
        
        # Pre-assemble combined operator (0.5*I + K)
        combined_op = 0.5 * identity_op + dlp_op
        
        for tag in self.tags:
            print(f"  Solving for electrode tag: {tag}...")
            
            # Create Dirichlet data as a numpy array
            dirichlet_data = np.zeros(self.grid.number_of_elements)
            
            # Find elements belonging to this tag
            if hasattr(self.grid, 'domain_indices'):
                element_tags = self.grid.domain_indices
                dirichlet_data[element_tags == tag] = V
            else:
                # If no domain indices, apply to all elements
                dirichlet_data[:] = V
            
            # Create GridFunction from array
            dirichlet_grid_fun = bp.GridFunction(self.space, coefficients=dirichlet_data)
            
            # Compute RHS: (0.5*I + K) * u_D
            rhs = combined_op * dirichlet_grid_fun
            
            # Solve for surface charge density sigma
            try:
                sigma, info = bp.linalg.gmres(slp_op, rhs, tol=1e-6, restart=100, maxit=1000)
                
                if info != 0:
                    print(f"    Warning: Tag {tag} GMRES did not converge (info={info})")
                else:
                    print(f"    Converged successfully")
                
                self.sigmas.append(sigma)
                self.electrode_tags.append(tag)
            except Exception as e:
                print(f"    Error solving for tag {tag}: {e}")
                # Add zero sigma as placeholder
                zero_sigma = bp.GridFunction(self.space, coefficients=np.zeros(self.space.dofs))
                self.sigmas.append(zero_sigma)
                self.electrode_tags.append(tag)
        
        print(f"Basis construction complete. {len(self.sigmas)} charge densities computed.")
        return self.sigmas

    def potential_at_points(self, points):
        """
        Calculates the basis potential for each sigma at the given point(s).
        """
        if not self.sigmas:
            raise RuntimeError("No charge densities found. Run solve_all_electrodes first.")
        
        if points.shape[0] != 3:
            raise ValueError(f"Points must have shape (3, N), got {points.shape}")
        
        print(f"\nEvaluating potential at {points.shape[1]} point(s)...")
        
        # Single layer potential operator for evaluation
        slp_pot = bp.operators.potential.laplace.single_layer(
            self.space, points
        )
        
        self.dc_potentials = []
        for i, sigma in enumerate(self.sigmas):
            try:
                V_basis = slp_pot * sigma
                # Convert to numpy array and flatten
                V_basis_array = np.array(V_basis).flatten()
                self.dc_potentials.append(V_basis_array)
                print(f"  Electrode {self.electrode_tags[i]}: basis potential range = [{V_basis_array.min():.3f}, {V_basis_array.max():.3f}]")
            except Exception as e:
                print(f"  Error evaluating potential for electrode {i}: {e}")
                self.dc_potentials.append(np.zeros(points.shape[1]))
        
        # Convert to numpy array: (n_electrodes, n_points)
        self.dc_potentials = np.array(self.dc_potentials)
        
        return self.dc_potentials

    def compute_potential(self, applied_voltages, points=None):
        """
        Applies linear superposition: V_total = sum(V_applied_i * Basis_Potential_i)
        """
        if points is not None:
            self.potential_at_points(points)
        
        if len(applied_voltages) != len(self.dc_potentials):
            raise ValueError(
                f"Expected {len(self.dc_potentials)} voltages, got {len(applied_voltages)}."
            )
        
        # Linear superposition
        applied_voltages = np.array(applied_voltages)
        V_total = np.dot(applied_voltages, self.dc_potentials)
        
        return V_total

    def map_potential_plane(self, x_range, y_range, z_fixed, applied_voltages, resolution=50):
        """Maps potential on a 2D plane."""
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x, y)
        
        # Create points array (3, N)
        points = np.vstack([X.ravel(), Y.ravel(), np.full(X.size, z_fixed)])
        
        # Compute potential
        V = self.compute_potential(applied_voltages, points)
        V_grid = V.reshape(resolution, resolution)
        
        return X, Y, V_grid

    def visualize_potential_plane(self, X, Y, V_grid, title="Electrostatic Potential"):
        """Visualizes potential on a 2D plane."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Contour plot
        contour = axes[0].contourf(X*1e6, Y*1e6, V_grid, levels=50, cmap='viridis')
        axes[0].set_xlabel('X (μm)')
        axes[0].set_ylabel('Y (μm)')
        axes[0].set_title(f'{title} - Contour')
        plt.colorbar(contour, ax=axes[0], label='Potential (V)')
        
        # Surface plot
        surf = axes[1].contourf(X*1e6, Y*1e6, V_grid, levels=50, cmap='plasma')
        axes[1].set_xlabel('X (μm)')
        axes[1].set_ylabel('Y (μm)')
        axes[1].set_title(f'{title} - Surface')
        plt.colorbar(surf, ax=axes[1], label='Potential (V)')
        
        plt.tight_layout()
        plt.show()


# ============================================================================
# SIMPLIFIED TEST VERSION (Without @bp.real_callable)
# ============================================================================

def test_simple_potential():
    """Test with a simple sphere to verify bempp works"""
    print("Testing bempp with a simple sphere...")
    
    # Create a simple sphere mesh
    grid = bp.shapes.sphere(h=0.1)  # h is mesh size
    print(f"Sphere mesh: {grid.number_of_elements} elements")
    
    # Create function space
    space = bp.function_space(grid, "DP", 0)
    
    # Create Dirichlet data (constant potential on sphere)
    dirichlet_coefficients = np.ones(space.dofs)  # 1V on entire surface
    dirichlet_fun = bp.GridFunction(space, coefficients=dirichlet_coefficients)
    
    # Assemble operators
    slp = bp.operators.boundary.laplace.single_layer(space, space, space, assembler="dense")
    dlp = bp.operators.boundary.laplace.double_layer(space, space, space, assembler="dense")
    identity = bp.operators.boundary.sparse.identity(space, space, space)
    
    # RHS: (0.5*I + K) * u_D
    rhs = (0.5 * identity + dlp) * dirichlet_fun
    
    # Solve
    sigma, info = bp.linalg.gmres(slp, rhs, tol=1e-6)
    print(f"GMRES converged with info={info}")
    
    # Evaluate potential at a point
    points = np.array([[0.0], [0.0], [2.0]])  # Point outside sphere
    slp_pot = bp.operators.potential.laplace.single_layer(space, points)
    potential = slp_pot * sigma
    
    print(f"Potential at (0,0,2): {potential[0]:.6f} V")
    print("Simple test successful!")
    
    return True


if __name__ == "__main__":
    # First test with simple sphere to verify bempp works
    test_simple_potential()
    
    msh_path = "Fivewiretrap.msh"
    
    # Create solver
    solver = DCPotentialSolver(msh_path)
    
    # Define electrode voltages based on number of tags
    n_electrodes = len(solver.tags)
    print(f"\nNumber of electrodes found: {n_electrodes}")
    
    if n_electrodes == 5:
        electrode_voltages = [10.0, 10.0, 0.0, -10.0, -10.0]
    elif n_electrodes == 3:
        electrode_voltages = [10.0, 0.0, -10.0]
    else:
        # Default: alternate positive and negative
        electrode_voltages = [10.0 if i % 2 == 0 else -10.0 for i in range(n_electrodes)]
    
    print(f"Using voltages: {electrode_voltages}")
    
    # Solve for basis functions
    try:
        solver.solve_all_electrodes(V=1.0)
        
        # Evaluate at center point
        eval_point = np.array([[0.0], [0.0], [50e-6]])
        V_total = solver.compute_potential(electrode_voltages, eval_point)
        print(f"\nTotal Potential at center (0,0,50μm): {V_total[0]:.6f} V")
        
        # Map potential plane
        print("\nMapping potential in X-Y plane...")
        X, Y, V_grid = solver.map_potential_plane(
            x_range=(-200e-6, 200e-6),
            y_range=(-200e-6, 200e-6),
            z_fixed=50e-6,
            applied_voltages=electrode_voltages,
            resolution=50
        )
        
        solver.visualize_potential_plane(X, Y, V_grid, title="5-Wire Ion Trap Potential")
        
    except Exception as e:
        print(f"\nError during solution: {e}")
        import traceback
        traceback.print_exc()