module Ca2Model

using Random, LoopVectorization

export OL_Solver

    function OL_Solver(;u0::Array{Float64} = [0.16, 30.0, 0.64, 0.88, 0.61, 0.12, 13.0, 0.0],  # Initial conditions
                        dt::Float64 = 0.01,                                                    # time step
                        dx::Float64 = 0.01,                                                    # space step
                        tmax::Float64 = 900.0,                                                 # total time
                        m::Int = 5,                                                            # spatial discretizations
                        tau_c::Float64 = 0.01,                                                 # Ornstein-Uhlenbeck time constant
                        D::Float64 = 1.2,                                                      # Diffusion for IP3
                        Vs::Float64 = 0.0301,                                                  # Max conductane SOCE (0.0301)
                        rho::Float64 = 0.00005,                                                # Diffusion for PDE
                       )::Tuple{Matrix{Float64}, Matrix{Float64}, Matrix{Float64}, Matrix{Float64},
                                Matrix{Float64}, Matrix{Float64}, Matrix{Float64}, Matrix{Float64}}      

        # Constants
        vncx = 1.4
        eta = 0.35
        Va = -80.0
        P = 96.5
        R = 8.314
        temp = 300.0
        VFRT = Va * P / (R * temp)
        cao = 12.0
        nao = 14.0
        ksat = 0.25
        kmcao = 1.3
        kmnao = 97.63
        kmnai = 12.3
        kmcai = 0.0026
        kn = 0.1
        v3 = 120.0
        hc3 = 2.0
        k3 = 0.3
        v2 = 0.5
        v_ip3 = 0.88
        ip3 = 0.2
        d5 = 0.08234
        k_pmca = 0.8
        v_pmca = 0.6
        vr = 18.0
        kb1 = 0.2573
        fi = 0.01
        γ = 9.0
        fe = 0.025
        a2 = 0.2
        d1 = 0.13
        d2 = 1.049
        d3 = 0.9434
        Ks = 50.0
        tau_soc = 30.0
        ka = 0.01920
        kb = 0.2573
        kc = 0.0571
        kca = 5.0
        kna = 5.0
        tau_o = 10.0
        ktau = 1.0
        minf = ip3 / (ip3 + d1);

        # Helper functions
        function hinf(Ca)
            (d2 * (ip3 + d1) / (ip3 + d3)) / ((d2 * (ip3 + d1) / (ip3 + d3)) + Ca)
        end
    
        function ninf(Ca)
            Ca / (Ca + d5)
        end
    
        function ninfi(Ca)
            1 / (1 + (kn / Ca)^2)
        end
    
        function Kx(Ca, Na)
            kmcao * (Na^3) + (kmnao^3) * Ca + (kmnai^3) * cao * (1 + Ca / kmcai) + kmcai * (nao^3) * (1 + (Na^3) / (kmnai^3)) + (Na^3) * cao + (nao^3) * Ca
        end
    
        function soc_inf(Cer)
            (Ks^4) / ((Ks^4) + (Cer^4))
        end
    
        function winf(Ca)
            (ka / (Ca^4) + 1 + (Ca^3) / kb) / (1 / kc + ka / (Ca^4) + 1 + (Ca^3) / kb)
        end
    
        function xinf(Ca, Na)
            1 - 1 / ((1 + (Ca / kca)^2) * (1 + (kna / Na)^2))
        end
    
        function Jip3(Ca, Cer, h)
            v_ip3 * (minf^3) * (ninf(Ca)^3) * (h^3) * (Cer - Ca)
        end
    
        function Jserca(Ca)
            v3 * (Ca^hc3) / ((k3^hc3) + (Ca^hc3))
        end
    
        function Jleak(Ca, Cer)
            v2 * (Cer - Ca)
        end
    
        function Jryr(Ca, Cer, w)
             vr * w * (1 + (Ca^3) / kb1) / (ka / (Ca^4) + 1 + (Ca^3) / kb1) * (Cer - Ca)
        end
    
        function Jsoc(s)
            Vs * s
        end
    
        function Jncx(Ca, x, Na)
            ninfi(Ca) * x * (vncx * (exp(eta * VFRT) * (Na^3) * cao -exp((eta - 1) * VFRT) * (nao^3) * Ca) / (Kx(Ca, Na) * (1 + ksat * (exp((eta - 1) * VFRT)))))
        end
    
        function Jpmca(Ca)
            (v_pmca * (Ca^2) / ((Ca^2) + (k_pmca^2)))
        end
    
        function Ca_dot(Ca, Cer, h, s, w, x, Na)
            fi * (Jip3(Ca, Cer, h) - Jserca(Ca) + Jleak(Ca, Cer) + Jryr(Ca, Cer, w) + Jsoc(s) + Jncx(Ca, x, Na) - Jpmca(Ca))
        end

        # Simulation parameters
        t = 0:dt:tmax         # time base
        n = length(t)         # number of time points
        r = rho * dt/(dx^2)     # r < 0.5 to ensure stability
        r ≤ 0.5 || error("Error: ensure r <= 0.5 for numerical stability.") 
    
        # Initialization
        Ca, Cer, h, s, w, x, Na, η = [Matrix{Float64}(undef, m, n) for i in 1:8]
        Ca[:, 1]  .= u0[1]
        Cer[:, 1] .= u0[2]
        h[:, 1]   .= u0[3]
        s[:, 1]   .= u0[4]
        w[:, 1]   .= u0[5]
        x[:, 1]   .= u0[6]
        Na[:, 1]  .= u0[7]
        η[:, 1]   .= u0[8]
        Random.seed!(10);
        noise_term = randn(m, n)
         
        @avx     for k in 1:(n - 1)
                       Ca[1, k + 1] = 2 * r * Ca[2, k]     + (1 - 2 *r) * Ca[1, k] + Ca_dot(Ca[1, k], Cer[1, k], h[1, k], s[1, k], w[1, k], x[1, k], Na[1, k]) * dt
                       Ca[m, k + 1] = 2 * r * Ca[m - 1, k] + (1 - 2 *r) * Ca[m, k] + Ca_dot(Ca[m, k], Cer[m, k], h[m, k], s[m, k], w[m, k], x[m, k], Na[m, k]) * dt
                 for i in 1:m         
                        Ca[i, k + 1] = ifelse((1<i)&(i<m), r * Ca[i - 1, k] + (1 - 2 * r) * Ca[i, k] + r * Ca[i + 1, k] + Ca_dot(Ca[i, k], Cer[i, k], h[i, k], s[i, k], w[i, k], x[i, k], Na[i, k]) * dt, Ca[i, k + 1])
                        Cer[i, k + 1]    = Cer[i, k] + (-γ * fe * (Jip3(Ca[i, k], Cer[i, k], h[i, k]) - Jserca(Ca[i, k]) + Jleak(Ca[i, k], Cer[i, k]) + Jryr(Ca[i, k], Cer[i, k], w[i, k]))) * dt
                        h[i, k + 1]      = h[i, k] + ((hinf(Ca[i, k]) - h[i, k]) / (1 / (a2 * ((d2 * (ip3 + d1) / (ip3 + d3)) + Ca[i, k]))) + η[i, k]) * dt
                        s[i, k + 1]      = s[i, k] + ((soc_inf(Cer[i, k]) - s[i, k]) / tau_soc) * dt
                        w[i, k + 1]      = w[i, k] + ((winf(Ca[i, k]) -w[i, k]) / (winf(Ca[i, k]) / kc)) * dt
                        x[i, k + 1]      = x[i, k] + ((xinf(Ca[i, k], Na[i, k]) - x[i, k]) / (0.25 + tau_o / (1 + (Ca[i, k] / ktau)))) * dt
                        Na[i, k + 1]     = Na[i, k] + (-3 * Jncx(Ca[i, k], x[i, k], Na[i, k])) * dt
                        η[i, k + 1]      = η[i, k] + (-η[i, k]/tau_c) * dt + sqrt(2 *D/tau_c) * noise_term[i, k]
                end 
            end
   
    return Ca, Cer, h, s, w, x, Na, η
    end
end