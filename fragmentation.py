from numba import njit
from scipy.optimize import minimize
import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import eig_banded

#initial_guess = result.x
def solve(xmax, nx, L, a, N, initial_guess = None, **kwargs):
    x = np.linspace(-xmax, xmax, nx)
    dx = x[1]-x[0]
    X = x.reshape((nx,1))
    absdiff = np.abs(X - X.T) 
    theta = np.where(np.abs(np.abs(x)-(L+a)/2)<a/2, 1, 0)
    
    #@njit
    def d(phi, dx): 
        pad = np.concatenate((np.array([0]), phi, np.array([0])))
        return (pad[2:] - pad[:-2]) / (2*dx)
    #@njit
    def i(phi, dx, axis):
        #return np.trapz(phi, x, axis=axis)
        return np.sum(phi, axis)*dx
  
    U_ext = i(absdiff * theta.reshape((1,nx)), dx, 1)

    #@njit
    def eps(phi):
        return i(4*d(phi, dx)**2 + U_ext*phi**2, dx, 0)

    #@njit
    def U(phi_i, phi_j, phi_k, phi_l):
        left  = (phi_i*phi_l)
        right = (phi_j*phi_k).reshape(1,nx) 
        return -(a/N) * i(left*i(absdiff*right, dx, axis=1), dx, 0)

    #@njit
    def alpha(phi):
        return U(phi, phi, phi, phi)

    #@njit
    def beta(phi_p, phi_m):
        return U(phi_p, phi_m, phi_m, phi_p)

    #@njit
    def gamma(phi_p, phi_m):
        return U(phi_p, phi_m, phi_p, phi_m)

    #@njit
    def unpack(phis):
        phi_p, phi_m = phis[:nx//2], phis[nx//2:]
        phi_p = np.concatenate((np.flip(phi_p), phi_p))
        phi_m = np.concatenate((-np.flip(phi_m), phi_m))
        phi_p = phi_p / np.sqrt(i(phi_p**2, dx, 0))
        phi_m = phi_m / np.sqrt(i(phi_m**2, dx, 0))
        return phi_p, phi_m

    def eig(a, b, c, N, eigvals_only=True):
        #Compute the lowest eigenvalue of H = a S_z + b S_z^2 + c S_x^2 for a spin S = N/2
        S = N / 2
        m = np.arange(-S, S + 1)
        dim = len(m)
        Sp2 = np.sqrt((S*(S+1)) - (m[0:-2]+1)*(m[0:-2]+2)) * np.sqrt((S*(S+1)) - (m[0:-2])*(m[0:-2]+1))
        #Sm2 = np.sqrt((S*(S+1)) - (m[2:]-1)*(m[2:]-2)) * np.sqrt((S*(S+1)) - (m[2:])*(m[2:]-1))
        Spm = S*(S+1) - m*(m-1)
        Smp = S*(S+1) - m*(m+1)
        diag = a*m + b*m**2 + c*(Spm + Smp)/4
        off2 = c*Sp2/4
        Ab = np.zeros((3, dim))
        Ab[0,:] = diag
        Ab[2,:-2] = off2
        return eig_banded(Ab, lower=True, eigvals_only=eigvals_only, select = 'i', select_range = (0, 0))[int(not eigvals_only)]


    S = N / 2
    m = np.arange(-S, S + 1)
    dim = len(m)
    Sp2 = np.sqrt((S*(S+1)) - (m[0:-2]+1)*(m[0:-2]+2)) * np.sqrt((S*(S+1)) - (m[0:-2])*(m[0:-2]+1))
    #Sm2 = np.sqrt((S*(S+1)) - (m[2:]-1)*(m[2:]-2)) * np.sqrt((S*(S+1)) - (m[2:])*(m[2:]-1))
    Spm = S*(S+1) - m*(m-1)
    Smp = S*(S+1) - m*(m+1)

    def eig(x1, x2, z1, z2, N, eigvals_only=True):
        #Compute the lowest eigenvalue of H = x1 S_x + x2 S_x^2 + z1 S_z + z2 S_z^2 for a spin S = N/2
        diag = z1*m + z2*m**2 + x2*(Spm + Smp)/4
        off1 = x1*np.sqrt(S*(S+1) - m[0:-1]*(m[0:-1]+1))/2
        off2 = x2*Sp2/4
        Ab = np.zeros((3, dim))
        Ab[0,:] = diag
        Ab[1,:-1] = off1
        Ab[2,:-2] = off2
        return eig_banded(Ab, lower=True, eigvals_only=eigvals_only, select = 'i', select_range = (0, 0))[int(not eigvals_only)]


    def H(phis):
        phi_p, phi_m = unpack(phis)
        Eps_p = eps(phi_p)
        Eps_m = eps(phi_m)
        Alpha_p = alpha(phi_p)
        Alpha_m = alpha(phi_m)
        Beta = beta(phi_p, phi_m)
        Gamma = gamma(phi_p, phi_m)
        E0 = ((Eps_p + Eps_m)/2 + (N/2-1)*(Alpha_p + Alpha_m)/2 + (N/2)*Beta - Gamma) # * N

        x1 = 0
        x2 = 4*Gamma/N
        z1 = (Eps_p - Eps_m + (N-1)*(Alpha_p - Alpha_m))/N
        z2 = (Alpha_p + Alpha_m - 2*Beta)/N

        #x2 = 0
        #z1 = 0
        #z2 = 0

        return E0 + eig(x1, x2, z1, z2, N)
        return E0 + eig(x1, x2, z1, z2, N)
        return E0 + eig((Eps_p - Eps_m + (N-1)*(Alpha_p - Alpha_m))/N, (Alpha_p + Alpha_m - 2*Beta)/N, 4*Gamma/N, N)

    #initial_guess = np.concatenate((theta[nx//2:], theta[nx//2:]))
    if initial_guess is None:
        initial_guess = np.ones(nx)
        initial_guess = np.concatenate((theta[nx//2:], theta[nx//2:]))
    
    result = minimize(H, initial_guess, **kwargs)
    phi_p, phi_m = unpack(result.x)

    def vec(phis):
        phi_p, phi_m = unpack(phis)
        Eps_p = eps(phi_p)
        Eps_m = eps(phi_m)
        Alpha_p = alpha(phi_p)
        Alpha_m = alpha(phi_m)
        Beta = beta(phi_p, phi_m)
        Gamma = gamma(phi_p, phi_m)
        a = (Eps_p - Eps_m + (N-1)*(Alpha_p - Alpha_m))/N
        b = (Alpha_p + Alpha_m - 2*Beta)/N
        c = 4*Gamma/N
        x1 = 0
        x2 = 4*Gamma/N
        z1 = (Eps_p - Eps_m + (N-1)*(Alpha_p - Alpha_m))/N
        z2 = (Alpha_p + Alpha_m - 2*Beta)/N
        #x2 = 0
        #z2 = 0
        #return eig_helper(0, c, a, b, N, eigvals_only=False)
        return eig(z1, z2, x1, x2, N, eigvals_only=False)
        #return eig((Eps_p - Eps_m + (N-1)*(Alpha_p - Alpha_m))/N, (Alpha_p + Alpha_m - 2*Beta)/N, 4*Gamma/N, N, eigvals_only=False)

    v = vec(result.x)
    v = v / np.sqrt(np.sum(v**2)) * np.sign(v[len(v)//2])
    #plt.plot(x, U_ext/max(U_ext))
    #print(result)
    #plt.plot(x, theta)
    #plt.plot(x, phi_p / max(phi_p), label='phi_+')
    #plt.plot(x, phi_m / max(phi_m), label='phi_-')
    #plt.show()
    #plt.bar(np.arange(len(v)), v.reshape(len(v)))
    return result, phi_p, phi_m, v


from tqdm import tqdm
import numpy as np
from IPython.display import clear_output
from multiprocessing import Pool
import os

if __name__ == "__main__":
    xmax = 60
    nx = 600
    a = 20
    N = 50
    Ls = np.linspace(0, 10, 40)

    def solve_wrapper(L):
        print('here')
        return solve(xmax, nx, L, a, N, method='L-BFGS-B', options={'maxfun': 500000, 'maxiter': 1000, 'ftol': 1e-12})

    # Get the number of available CPUs
    num_processes = os.cpu_count()
    print(f"Number of processes: {num_processes}")
    with Pool(processes=num_processes) as p:  # Using the number of CPUs
        results = list(tqdm(p.map(solve_wrapper, Ls), total=len(Ls)))
    
    cmap = plt.get_cmap('viridis', len(Ls))
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    with Pool(processes=None) as p:
        results = list(tqdm(p.imap(solve_wrapper, Ls), total=len(Ls)))
    coefficients = []
    left_modes = []
    right_modes = []
    for i in range(len(Ls)):
        result, phi_p, phi_m, v = results[i]
        assert(result.success)
        phi_R, phi_L = (phi_p + phi_m)/np.sqrt(2), (phi_p - phi_m)/np.sqrt(2)
        phi_R *= np.sign(phi_R[np.argmax(np.abs(phi_R))])
        phi_L *= np.sign(phi_L[np.argmax(np.abs(phi_L))])
        coefficients.append(v)
        left_modes.append(phi_L)
        right_modes.append(phi_R)
        axs[0].plot(np.linspace(-xmax, xmax, nx), phi_R, color=cmap(i), label=f"L={L:.1f}")
        axs[0].plot(np.linspace(-xmax, xmax, nx), phi_L, color=cmap(i), label=f"L={L:.1f}", alpha = 0.1)
        axs[1].bar(np.arange(len(v)), v.reshape(len(v)), color=cmap(i), label = f"L={L:.1f}", alpha=0.5)

plt.show()