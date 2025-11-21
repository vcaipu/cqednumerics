import jax
import jax.numpy as jnp
import numpy as np
import skfem as fem
import unittest
from FEMSystem import FEMSystem

jax.config.update("jax_enable_x64", True)

class TestFEMSystemIntegrals(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Setup a high-quality mesh for accurate integration testing
        # Unit square from (0,0) to (1,1)
        # We use P1 elements but high integration order to test the quadrature logic
        cls.mesh = fem.MeshTri.init_sqsymmetric().refined(4) 
        cls.element = fem.ElementTriP1()
        cls.intorder = 5 # High order quadrature
        
        cls.femsystem = FEMSystem(cls.mesh, cls.element, cls.intorder)
        print(f"\nMesh initialized with {cls.mesh.nelements} elements.")

    def test_constant_integration(self):
        # Integral of 1 over unit square should be 1.0
        def func(x):
            return jnp.ones_like(x[0])
            
        result = self.femsystem.integrate_function(func)
        print(f"Constant (1.0): Expected=1.0, Got={result:.8f}")
        self.assertAlmostEqual(result, 1.0, places=6)

    def test_linear_integration_x(self):
        # Integral of x over unit square: int_0^1 x dx * int_0^1 dy = 0.5 * 1 = 0.5
        def func(x):
            return x[0]
            
        result = self.femsystem.integrate_function(func)
        print(f"Linear (x): Expected=0.5, Got={result:.8f}")
        self.assertAlmostEqual(result, 0.5, places=6)

    def test_linear_integration_y(self):
        # Integral of y over unit square: 0.5
        def func(x):
            return x[1]
            
        result = self.femsystem.integrate_function(func)
        print(f"Linear (y): Expected=0.5, Got={result:.8f}")
        self.assertAlmostEqual(result, 0.5, places=6)

    def test_quadratic_integration(self):
        # Integral of x^2 + y^2
        # int_0^1 x^2 dx = 1/3. Total = 1/3 + 1/3 = 2/3 = 0.666666...
        def func(x):
            return x[0]**2 + x[1]**2
            
        result = self.femsystem.integrate_function(func)
        print(f"Quadratic (x^2+y^2): Expected={2/3:.8f}, Got={result:.8f}")
        self.assertAlmostEqual(result, 2/3, places=6)

    def test_trigonometric_integration(self):
        # Integral of sin(pi*x)*sin(pi*y)
        # (2/pi) * (2/pi) = 4/pi^2
        def func(x):
            return jnp.sin(jnp.pi * x[0]) * jnp.sin(jnp.pi * x[1])
            
        expected = 4.0 / (np.pi**2)
        result = self.femsystem.integrate_function(func)
        print(f"Trigonometric (sin(pi*x)sin(pi*y)): Expected={expected:.8f}, Got={result:.8f}")
        self.assertAlmostEqual(result, expected, places=6)

    def test_exponential_integration(self):
        # Integral of exp(x) over unit square
        # (e^1 - e^0) * 1 = e - 1
        def func(x):
            return jnp.exp(x[0])
            
        expected = np.e - 1.0
        result = self.femsystem.integrate_function(func)
        print(f"Exponential (exp(x)): Expected={expected:.8f}, Got={result:.8f}")
        self.assertAlmostEqual(result, expected, places=6)
        
    def test_gaussian_integration(self):
        # Integral of a Gaussian peak centered at 0.5, 0.5
        # exp(-((x-0.5)^2 + (y-0.5)^2) / (2*sigma^2))
        # We use a small sigma so it's fully contained in domain, but not too small that mesh misses it
        sigma = 0.1
        def func(x):
            return jnp.exp(-((x[0]-0.5)**2 + (x[1]-0.5)**2) / (2 * sigma**2))
        
        # Analytical integral over R^2 is 2*pi*sigma^2.
        # Since 0.5 +/- 5*sigma is within [0,1], the integral over square is very close to R^2 integral.
        expected = 2 * np.pi * sigma**2
        
        result = self.femsystem.integrate_function(func)
        print(f"Gaussian (sigma={sigma}): Expected~={expected:.8f}, Got={result:.8f}")
        self.assertAlmostEqual(result, expected, places=4) # Slightly lower precision due to boundary cutoff/mesh resolution

if __name__ == '__main__':
    unittest.main()
