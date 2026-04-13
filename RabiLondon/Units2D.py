import numpy as np
class Units2D:
    fco = {}

    def __init__(self,xi=None,kappa=None): 

        # Of a COOPER PAIR, in SI UNITS, get effective mass if needed
        M_CP = 1.82e-30 #kg
        Q_CP = -3.204e-19 #Coulombs

        # Fuandmental Constants, in SI UNITS
        mu_0 = 1.25663706212e-6 #Henry per meter / Newton per Ampere squared
        epsilon_0 = 8.8541878128e-12 #Farads per meter
        h_bar = 1.0545718e-34 #Joules seconds

        # Fundamental constants object
        fco = {
            "M_CP": M_CP,
            "Q_CP": Q_CP,
            "mu_0": mu_0,
            "epsilon_0": epsilon_0,
            "h_bar": h_bar
        }
        self.fco = fco

        # Change FCO if needed
        if (xi is not None) and (kappa is not None):
            m_eff = self.get_m_eff(kappa,xi)
            print(f"Effective mass: {m_eff} kg | {m_eff/M_CP} times (2m_e)")
            M_CP = m_eff
            fco = {
                "M_CP": M_CP,
                "Q_CP": Q_CP,
                "mu_0": mu_0,
                "epsilon_0": epsilon_0,
                "h_bar": h_bar
            }
            self.fco = fco

    # Run before everything else, and effective mass 
    def get_m_eff(self,kappa,xi):
        m,q,mu_0,epsilon_0,hbar = self.fco["M_CP"],self.fco["Q_CP"],self.fco["mu_0"],self.fco["epsilon_0"],self.fco["h_bar"]
        c = 1/np.sqrt(mu_0*epsilon_0)
        m_eff = hbar / (2*c)* kappa / xi

        return m_eff
    
    
    # n_s from \xi
    def n_sfromxi(self,xi):
        m,q,mu_0,epsilon_0,hbar = self.fco["M_CP"],self.fco["Q_CP"],self.fco["mu_0"],self.fco["epsilon_0"],self.fco["h_bar"]
        c = 1/np.sqrt(mu_0*epsilon_0)
        h = hbar * 2 * np.pi

        lambda_c = h / (m*c)
        n_s = m/(mu_0*q**2) * (lambda_c / (4*np.pi *xi**2))**2
        return n_s

    def n_sfromlondon(self,lambda_L):
        m,q,mu_0,epsilon_0,hbar = self.fco["M_CP"],self.fco["Q_CP"],self.fco["mu_0"],self.fco["epsilon_0"],self.fco["h_bar"]

        n_s = m / (mu_0 * q**2 * lambda_L**2)
        return n_s

    def london_from_n_s(self,n_s):
        m,q,mu_0,epsilon_0,hbar = self.fco["M_CP"],self.fco["Q_CP"],self.fco["mu_0"],self.fco["epsilon_0"],self.fco["h_bar"]
    
        return np.sqrt(m/(mu_0 * n_s * q**2))
    

    def xifromn_s(self,n_s):
        m,q,mu_0,epsilon_0,hbar = self.fco["M_CP"],self.fco["Q_CP"],self.fco["mu_0"],self.fco["epsilon_0"],self.fco["h_bar"]
        c = 1/np.sqrt(mu_0*epsilon_0)
        h = hbar * 2 * np.pi

        lambda_c = h / (m*c)
        lambda_L = np.sqrt(m/(mu_0*n_s*q**2))
        return np.sqrt(lambda_c * lambda_L / (4*np.pi))
    
    def compute_dz(self,n_s,N,area):
        m,q,mu_0,epsilon_0,hbar = self.fco["M_CP"],self.fco["Q_CP"],self.fco["mu_0"],self.fco["epsilon_0"],self.fco["h_bar"]
        c = 1/np.sqrt(mu_0*epsilon_0)
        h = hbar * 2 * np.pi

        lambda_c = h / (m*c)
        res_dz = (N/area) * (mu_0*q**2/m)**(3/4) * (4*np.pi/lambda_c)**(3/2) * (1/n_s)**(1/4)
        return res_dz

        # Compute value for n_s
    def compute_values_from_deltaZ(self,deltaZ,N,area):
        m,q,mu_0,epsilon_0,hbar = self.fco["M_CP"],self.fco["Q_CP"],self.fco["mu_0"],self.fco["epsilon_0"],self.fco["h_bar"]
        c = 1/np.sqrt(mu_0*epsilon_0)
        h = hbar * 2 * np.pi

        lambda_c = h / (m*c)

        n_s = ((N/area)**4) * (mu_0*q**2/m)**3 * (4*np.pi/lambda_c)**6 * (1/(deltaZ**4))
        lambda_L = np.sqrt(m/(mu_0*n_s*q**2))
        xi = np.sqrt(lambda_L * lambda_c / (4*np.pi))
        c_bar = c / xi
        return n_s, lambda_L, lambda_c, xi, c_bar
    
    def convert_energy_to_rad_s(self,E_nondim,xi):
        m,q,mu_0,epsilon_0,hbar = self.fco["M_CP"],self.fco["Q_CP"],self.fco["mu_0"],self.fco["epsilon_0"],self.fco["h_bar"]
        n_s = self.n_sfromxi(xi)
        E_in_joules = (q**2) * n_s * (xi**2) / (2*epsilon_0) * E_nondim
        E_in_rad_s = E_in_joules / hbar
        return E_in_rad_s
    
    def kappa_cbar_from_xi(self,xi):
        m,q,mu_0,epsilon_0,hbar = self.fco["M_CP"],self.fco["Q_CP"],self.fco["mu_0"],self.fco["epsilon_0"],self.fco["h_bar"]
        c = 1/np.sqrt(mu_0*epsilon_0)

        n_s = self.n_sfromxi(xi)
        lambda_L = self.london_from_n_s(n_s)
        kappa = lambda_L / xi
        c_bar = c / xi
        return kappa, c_bar