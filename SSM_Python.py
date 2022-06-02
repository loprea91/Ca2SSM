# Imports
import numpy as np
from numba import jit

@jit(nopython=True, fastmath=True)
def SPDE_solver(ICs = [0.16, 30.0, 0.64, 0.88, 0.61, 0.12, 13.0, 0.0],  # Initial conditions
              dt = 0.01,                                        # time step
              dx = 0.01,                                        # space step
              tmax = 900.0,                                       # total time
              m = 5,                                            # spatial discretizations
              tau_c = 0.01,                                     # Ornstein-Uhlenbeck time constant
              D = 0.4,                                          # Diffusion for IP3
              Vs = 0.0301,                                      # Max conductane SOCE (0.0301)
              rho = 0.00005,                                    # Diffusion for PDE
              v3 = 120,
              vr = 18,
              seed = 10,
              alpha = 1.0,
              beta = 1.0,
              v_ip3 = 0.88,
              v_pmca = 0.6,
              vncx = 1.4,
              **kwargs 
              ):      
    
    # Random seed
    np.random.seed(seed)
     
    # Error class
    #class StabilityError(Exception):
    #    pass
    
    # Simulation parameters
    tspan = np.arange(0.0,(tmax-dt),dt)       
    n = len(tspan)                                                                                
    r = rho*dt/(dx**2)
    #if r >= 0.5:
    #    raise StabilityError('The value rho*dt/(dx**2) must be < 0.5 to ensure numerical stability!')
    
    # Model Parameters
    eta = 0.35
    Va = -80
    P = 96.5
    R = 8.314
    temp = 300
    VFRT = Va * P / (R * temp)
    cao = 12
    nao = 14
    ksat = 0.25
    kmcao = 1.3
    kmnao = 97.63
    kmnai = 12.3
    kmcai = 0.0026
    kn = 0.1
    hc3 = 2
    k3 = 0.3
    v2 = 0.5
    ip3 = 0.2
    d5 = 0.08234
    k_pmca = 0.8
    kb1 = 0.2573
    fi = 0.01
    gamma = 9
    fe = 0.025
    a2 = 0.2
    d1 = 0.13
    d2 = 1.049
    d3 = 0.9434
    Ks = 50
    tau_soc = 30
    ka = 0.01920
    kb = 0.2573
    kc = 0.0571
    kca = 5
    kna = 5
    tau_o = 10
    ktau = 1
    minf = ip3 / (ip3 + d1)
    
    # Functions
    def hinf(Ca): 
        return (d2 * (ip3 + d1) / (ip3 + d3)) / ((d2 * (ip3 + d1) / (ip3 + d3)) + Ca)
    
    def ninf(Ca): 
        return Ca / (Ca + d5)
    
    def ninfi(Ca): 
        return 1 / (1 + (kn / Ca)**2)
    
    def Kx(Ca, Na): 
        return kmcao * (Na**3) + (kmnao**3) * Ca + (kmnai**3) * cao * (1 + Ca / kmcai) + kmcai * (nao**3) * (1 + (Na**3) / (kmnai**3)) + (Na**3) * cao + (nao**3) * Ca
    
    def soc_inf(Cer): 
        return (Ks**4) / ((Ks**4) + (Cer**4))
    
    def winf(Ca): 
        return (ka / (Ca**4) + 1 + (Ca**3) / kb) / (1 / kc + ka / (Ca**4) + 1 + (Ca**3) / kb)
    
    def xinf(Ca, Na): 
        return 1 - 1 / ((1 + (Ca / kca)**2) * (1 + (kna / Na)**2))
    
    def Jip3(Ca, Cer, h): 
        return v_ip3 * (minf**3) * (ninf(Ca)**3) * (h**3) * (Cer - Ca)
    
    def Jserca(Ca): 
        return v3 * (Ca**hc3) / ((k3**hc3) + (Ca**hc3))
    
    def Jleak(Ca, Cer): 
        return v2 * (Cer - Ca)
    
    def Jryr(Ca, Cer, w): 
        return vr * w * (1 + (Ca**3) / kb1) / (ka / (Ca**4) + 1 + (Ca**3) / kb1) * (Cer - Ca)
    
    def Jsoc(s): 
        return Vs * s
    
    def Jncx(Ca, x, Na): 
        return ninfi(Ca) * x * (vncx * (np.exp(eta * VFRT) * (Na**3) * cao -np.exp((eta - 1) * VFRT) * (nao**3) * Ca) / (Kx(Ca, Na) * (1 + ksat * (np.exp((eta - 1) * VFRT)))))
    
    def Jpmca(Ca): 
        return (v_pmca * (Ca**2) / ((Ca**2) + (k_pmca**2)))
    
    def dCa_dt(Ca, Cer, h, s, w, x, Na):
        return fi * (alpha * (Jip3(Ca, Cer, h) + Jleak(Ca, Cer) + Jryr(Ca, Cer, w) + Jsoc(s) + Jncx(Ca, x, Na)) + beta * (- Jpmca(Ca) - Jserca(Ca)))
    
    # Initialization of state variables & noise
    Ca, Cer, h, s, w, x, Na, eta_u = [np.zeros((n, m)) for i in range(8)]
    Ca[0, :]    = ICs[0]
    Cer[0, :]   = ICs[1]
    h[0, :]     = ICs[2]
    s[0, :]     = ICs[3]
    w[0, :]     = ICs[4]
    x[0, :]     = ICs[5]     
    Na[0, :]    = ICs[6]        
    eta_u[0, :] = ICs[7]
    noise_term  = np.random.randn(n, m)
    
    # FTCS Scheme  (∂u/∂t = ∂²u/∂x² + f(u(t, x)) 
    for k in range(n-1):
        # BCs: Neumann (∂C/∂t = 0) at x₁ and xₘ
        Ca[k + 1, 0]   = 2 * r * Ca[k, 1]     + (1 - 2*r) * Ca[k, 0]   + dCa_dt(Ca[k, 0], Cer[k, 0], h[k, 0], s[k, 0], w[k, 0], x[k, 0], Na[k, 0]) * dt
        Ca[k + 1, m-1] = 2 * r * Ca[k, m-2]   + (1 - 2*r) * Ca[k, m-1] + dCa_dt(Ca[k, m-1], Cer[k, m-1], h[k, m-1], s[k, m-1], w[k, m-1], x[k, m-1], Na[k, m-1]) * dt
        for i in range(m):
            if i > 0 and i < m-1:
                Ca[k + 1, i] = r*Ca[k, i-1] + (1 - 2*r)*Ca[k, i] + r*Ca[k, i+1] + dCa_dt(Ca[k, i], Cer[k, i], h[k, i], s[k, i], w[k, i], x[k, i], Na[k, i]) * dt
            Cer[k + 1, i]   = Cer[k, i] + (-gamma * fe * (alpha * (Jip3(Ca[k, i], Cer[k, i],  h[k, i]) + Jleak(Ca[k, i], Cer[k, i]) + Jryr(Ca[k, i], Cer[k, i], w[k, i])) - beta * Jserca(Ca[k, i]))) * dt
            h[k + 1, i]     = h[k, i] + ((hinf(Ca[k, i]) - h[k, i]) / (1 / (a2 * ((d2 * (ip3 + d1) / (ip3 + d3)) + Ca[k, i]))) + eta_u[k, i]) * dt
            s[k + 1, i]     = s[k, i] + ((soc_inf(Cer[k, i]) - s[k, i]) / tau_soc) * dt
            w[k + 1, i]     = w[k, i] + ((winf(Ca[k, i]) -w[k, i]) / (winf(Ca[k, i]) / kc)) * dt
            x[k + 1, i]     = x[k, i] + ((xinf(Ca[k, i], Na[k, i]) - x[k, i]) / (0.25 + tau_o / (1 + (Ca[k, i] / ktau)))) * dt
            Na[k + 1, i]    = Na[k, i] + (-3 * Jncx(Ca[k, i], x[k, i], Na[k, i])) * dt
            eta_u[k + 1, i] = eta_u[k, i] + (-eta_u[k, i]/tau_c) * dt + np.sqrt(2*D/tau_c) * noise_term[k, i]
                      
    return Ca, Cer, h, s, w, x, Na, eta_u, noise_term

    

