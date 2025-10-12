#from numba import njit
from scipy.optimize import minimize
import jax.numpy as jnp
from matplotlib import pyplot as plt
from scipy.linalg import eig_banded
import jax
from jax import jit
from tqdm import tqdm



# first derivative approximation via finite difference for various orders of accuracy

@jit
def d2(phi, dx): 
    pad = jnp.concatenate((jnp.array([0]), phi, jnp.array([0])))
    return (pad[2:] - pad[:-2]) / (2*dx)

@jit
def d4(phi, dx): 
    pad = jnp.concatenate((jnp.array([0,0]), phi, jnp.array([0,0])))
    return ((-1/12)*pad[4:] + (2/3)*pad[3:-1] - (2/3)*pad[1:-3] + (1/12)*pad[:-4])/dx

@jit
def d6(phi, dx): 
    pad = jnp.concatenate((jnp.array([0,0,0]), phi, jnp.array([0,0,0])))
    return ((1/60)*pad[6:] + (-3/20)*pad[5:-1] + (3/4)*pad[4:-2] + 0*pad[3:-3] + (-3/4)*pad[2:-4] + (3/20)*pad[1:-5] + (-1/60)*pad[:-6])/dx

@jit
def d8(phi, dx): 
    pad = jnp.concatenate((jnp.array([0,0,0,0]), phi, jnp.array([0,0,0,0])))
    return ((-1/280)*pad[8:] + (4/105)*pad[7:-1] + (-1/5)*pad[6:-2] + (4/5)*pad[5:-3] + 0*pad[4:-4] + (-4/5)*pad[3:-5] + (1/5)*pad[2:-6] + (-4/105)*pad[1:-7] + (1/280)*pad[:-8])/dx

d = d8

# integration along a axis 0 or 1

@jit
def i0(phi, dx):
    return jnp.trapezoid(phi, dx = dx, axis = 0)

@jit
def i1(phi, dx):
    return jnp.trapezoid(phi, dx = dx, axis = 1)

class solver:
    def __init__(self, xmax, nx, L, a, N, initial_guess, learning_rate, steps):
        # This is the constructor (initializer)
        self.xmax = xmax
        self.nx = nx
        self.L = L
        self.a = a
        self.N = N
        self.initial_guess = initial_guess
        self.learning_rate = learning_rate
        self.steps = steps
        self.x = jnp.linspace(-xmax, xmax, nx)
        self.dx = self.x[1]-self.x[0]
        self.X = self.x.reshape((nx,1))
        self.absdiff = jnp.abs(self.X - self.X.T) 
        self.theta = jnp.where(jnp.abs(jnp.abs(self.x)-(L+a)/2)<a/2, 1.0, 0.0)
        self.U_ext = i1(self.absdiff * self.theta.reshape((1,nx)), self.dx)
        self.S = self.N / 2
        self.m = jnp.arange(-self.S, self.S + 1)
        self.dim = len(self.m)
        self.coeff_plus  = jnp.sqrt(self.S*(self.S+1) - self.m*(self.m-1))  # S+
        self.coeff_minus = jnp.sqrt(self.S*(self.S+1) - self.m*(self.m+1))  # S-

    @jit
    def eps(self, phi):
        return i0(4*d(phi, self.dx)**2 + self.U_ext*phi**2, self.dx)

    @jit
    def U(self, phi_i, phi_j, phi_k, phi_l):
        left  = (phi_i*phi_l)
        right = (phi_j*phi_k).reshape(1,self.nx) 
        return -(self.a/self.N) * i0(left*i1(self.absdiff*right, self.dx), self.dx)

    @jit
    def alpha(self, phi):
        return self.U(phi, phi, phi, phi)

    @jit
    def beta(self, phi_p, phi_m):
        return self.U(phi_p, phi_m, phi_m, phi_p)

    @jit
    def gamma(self, phi_p, phi_m):
        return self.U(phi_p, phi_m, phi_p, phi_m)

    @jit
    def unpack(self, phis_c):
        phi_p, phi_m, c = phis_c[:self.nx//2], phis_c[self.nx//2:self.nx], phis_c[self.nx:]
        phi_p = jnp.concatenate((jnp.flip(phi_p), phi_p))
        phi_m = jnp.concatenate((-jnp.flip(phi_m), phi_m))
        phi_p = phi_p / jnp.sqrt(i0(phi_p**2, self.dx))
        phi_m = phi_m / jnp.sqrt(i0(phi_m**2, self.dx))
        c = c / jnp.sqrt(jnp.sum(c**2))
        return phi_p, phi_m, c

    @jit
    def Sz(self, c):
        return self.m*c
    
    @jit
    def Sx(self, c):
        c_plus  = jnp.concatenate([jnp.array([0.0]), c[:-1]])  # c_{m-1}
        c_minus = jnp.concatenate([c[1:], jnp.array([0.0])])  # c_{m+1}
        return 0.5 * (self.coeff_plus * c_plus + self.coeff_minus * c_minus)

    @jit
    def H(self, phis_c):
        phi_p, phi_m, c = self.unpack(phis_c)
        Eps_p = self.eps(phi_p)
        Eps_m = self.eps(phi_m)
        Alpha_p = self.alpha(phi_p)
        Alpha_m = self.alpha(phi_m)
        Beta = self.beta(phi_p, phi_m)
        Gamma = self.gamma(phi_p, phi_m)
        E0 = ((Eps_p + Eps_m)/2 + (self.N/2-1)*(Alpha_p + Alpha_m)/2 + (self.N/2)*Beta - Gamma) # * N

        x1 = 0
        x2 = 4*Gamma/self.N
        z1 = (Eps_p - Eps_m + (self.N-1)*(Alpha_p - Alpha_m))/self.N
        z2 = (Alpha_p + Alpha_m - 2*Beta)/self.N

        Sx_c = self.Sx(c)
        Sx2_c = self.Sx(Sx_c)
        Sz_c = self.Sz(c)
        Sz2_c = self.Sz(Sz_c)

        return E0 + jnp.dot(c, x1*Sx_c + x2*Sx2_c + z1*Sz_c + z2*Sz2_c)


  
    def solve(self):
        grad = jax.grad(self.H)
        lr = self.learning_rate
        steps = self.steps
        x = self.initial_guess
        x = jnp.concatenate((self.theta[self.nx//2:], self.theta[self.nx//2:], jnp.zeros(self.dim-1), jnp.ones(1)))
        for _ in range(steps):
            g = grad(x)     # compute gradient
            x = x - lr * g    # update step
        phi_p, phi_m, v = self.unpack(x)
        return phi_p, phi_m, v
    
if __name__ == "__main__":
    xmax = 100
    nx = 2000
    a = 10
    N = 100
    Ls = jnp.linspace(0, 40, 10)
    cmap = plt.get_cmap('viridis', len(Ls))

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    coefficients = []
    left_modes = []
    right_modes = []
    learning_rate = 1
    steps = 1000
    initial_guess = jnp.ones(nx) # None # np.ones(nx) # np.random.rand(nx) # 
    for i, L in tqdm(enumerate(Ls)):
        print(i)
        phi_p, phi_m, v = solver(xmax, nx, L, a, N, initial_guess, learning_rate, steps).solve()
        phi_R, phi_L = (phi_p + phi_m)/jnp.sqrt(2), (phi_p - phi_m)/jnp.sqrt(2)
        phi_R *= jnp.sign(phi_R[jnp.argmax(jnp.abs(phi_R))])
        phi_L *= jnp.sign(phi_L[jnp.argmax(jnp.abs(phi_L))])
        coefficients.append(v)
        left_modes.append(phi_L)
        right_modes.append(phi_R)
        initial_guess = jnp.concatenate((phi_p[nx//2:], phi_m[nx//2:], v))
        
        axs[0].plot(jnp.linspace(-xmax, xmax, nx), phi_m, color=cmap(i), label=f"L={L:.1f}")
        axs[0].plot(jnp.linspace(-xmax, xmax, nx), phi_p, color=cmap(i), label=f"L={L:.1f}", alpha = 0.1)

        axs[1].plot(jnp.arange(-N/2, N/2 + 1), v.reshape(len(v)), color=cmap(i), label = f"L={L:.1f}", alpha=1)

        axs[2].scatter(i, v[0])

plt.show()