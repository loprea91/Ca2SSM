def OL_solver(ICs, trange, pars):
    import numpy as np
    from scipy import integrate
    p = dict(Vs=.1,
             vncx=1.4,
             eta=0.35,
             Va=-80,
             P=96.5,
             Ra=8.314,
             temp=300,
             cao=12,
             nao=14,
             ksat=0.25,
             kmcao=1.3,
             kmnao=97.63,
             kmnai=12.3,
             kmcai=0.0026,
             kn=0.1,
             v3=120,
             hc3=2,
             k3=0.3,
             v2=0.5,
             v_ip3=0.88,
             ip3=0.2,
             d5=0.08234,
             k_pmca=0.8,
             v_pmca=0.6,
             vr=18,
             kb1=0.2573,
             fi=0.01,
             gammaa=9,
             fe=0.025,
             a2=0.2,
             d1=0.13,
             d2=1.049,
             d3=0.9434,
             Ks=50,
             tau_soc=30,
             ka=0.01920,
             kb=0.2573,
             kc=0.0571,
             kca=5,
             kna=5,
             tau_o=10,
             ktau=1,
             in_amp=0,
             in_start=100,
             in_per=50,
             in_dur=10)
    
    p.update(pars)

    def OL_ODE(t, y):
        # State variables
        Ca, Cer, h, soc, w, x, Na = y
        
        inp = p['in_amp'] if (p['in_start'] <= t and np.mod(t - p['in_start'], p['in_per']) < p['in_dur']) else 0

        hinf = (p['d2']*(p['ip3']+p['d1'])/(p['ip3']+p['d3']))/((p['d2']*(p['ip3']+p['d1'])/(p['ip3']+p['d3']))+Ca)
        minf = p['ip3']/(p['ip3'] + p['d1'])
        ninf = Ca/(Ca+p['d5'])
        ninfi = 1/(1+(p['kn']/Ca)**2)
        Kx = p['kmcao']*(Na**3)+(p['kmnao']**3)*Ca+(p['kmnai']**3)*p['cao']*(1+Ca/p['kmcai'])
        +p['kmcai']*(p['nao']**3)*(1+(Na**3)/(p['kmnai']**3))+(Na**3)*p['cao']+(p['nao']**3)*Ca
        VFRT = p['Va']*p['P']/(p['Ra']*p['temp'])
        soc_inf = (p['Ks']**4)/((p['Ks']**4)+(Cer**4))
        winf = (p['ka']/(Ca**4)+1+(Ca**3)/p['kb'])/(1/p['kc']+p['ka']/(Ca**4)+1+(Ca**3)/p['kb'])
        xinf = 1-1/((1+(Ca/p['kca'])**2)*(1+(p['kna']/Na)**2))
        Jip3 = p['v_ip3']*(minf**3)*(ninf**3)*(h**3)*(Cer-Ca)
        Jserca = p['v3']*(Ca**p['hc3'])/((p['k3']**p['hc3'])+(Ca**p['hc3']))
        Jleak = p['v2']*(Cer-Ca)
        Jryr = p['vr']*w*(1+(Ca**3)/p['kb1'])/(p['ka']/(Ca**4)+1+(Ca**3)/p['kb1'])*(Cer-Ca)
        Jsoce = p['Vs']*soc
        Jncx = ninfi*x*(p['vncx']*(np.exp(p['eta']*VFRT)*(Na**3)*p['cao']-np.exp((p['eta']-1)*VFRT)*(p['nao']**3)*Ca)/(Kx*(1+p['ksat']*(np.exp((p['eta']-1)*VFRT)))))
        Jpmca = (p['v_pmca']*(Ca**2)/((Ca**2)+(p['k_pmca']**2)))

        # ODEs
        Ca_dt = 2*(p['fi']*(Jip3-Jserca+Jleak+Jryr+Jsoce+Jncx-Jpmca))
        Cer_dt = 2*(-p['gammaa']*p['fe']*(Jip3-Jserca+Jleak+Jryr))
        h_dt = 2*((hinf-h)/(1/(p['a2']*((p['d2']*(p['ip3']+p['d1'])/(p['ip3']+p['d3']))+Ca))))
        s_dt = 2*((soc_inf-soc)/p['tau_soc'])
        w_dt = 2*((winf-w)/(winf/p['kc']))
        x_dt = 2*((xinf-x)/(0.25+p['tau_o']/(1+(Ca/p['ktau']))))
        Na_dt = 2*(-3*Jncx)

        return np.array([Ca_dt + inp, Cer_dt, h_dt, s_dt, w_dt, x_dt, Na_dt])

    tseries = integrate.solve_ivp(OL_ODE, trange, ICs,  method = 'RK45', max_step=0.01)
    return tseries
