function [Ca, Cer, h, s, w, x, Na, eta_u] = OL_solver(varargin)
% Solves the 7D stochastic PDE Ca2+ model from (Jiang et al, 2022) using
% FTCS for the spatial term and Euler-Murayama for stochastic
% Input arguments:
%     u0           Initial conditions  
%                  default: [0.16, 30, 0.64, 0.88, 0.61, 0.12, 13, 0])
%     dt           Time step (default: 0.01)
%     dx           Space step (default: 0.01)
%     tmax         Total time (default: 900s)
%     m            Spatial discretizations (default: 5)
%     tau_c        Ornstein-Uhlenbeck time constant (default: 0.01)
%     D            Diffusion for IP3 (default: 1.2)
%     Vs           Max conductane SOCE (default: 0.0301)
%     rho          Diffusion constant for spatial term (default: 0.0301) 
%     vec_flag     Vectorization Boolean (faster if m > ~time30)
% Outputs:
%     Ca           Internal [Ca2+]
%     Cer          Internal ER [Ca2+]
%     h            Slow inactivation variable for IP3
%     s            Slow inactivation variable for SOCE
%     w            Slow inactivation variable for RyR
%     x            Slow inactivation variable for NCX
%     Na           Internal [Na+]
%     eta_u        Noise term (Ornstein-Uhlenbeck process)
%% Parser
ip = inputParser;
ip.addParameter('u0', [0.16, 30, 0.64, 0.88, 0.61, 0.12, 13, 0], ...
                     @(x) isnumeric(x) && isequal(size(x), [1, 8]))
ip.addParameter('dt', 1e-2, @(x) isnumeric(x) && isscalar(x))
ip.addParameter('dx', 1e-2, @(x) isnumeric(x) && isscalar(x))
ip.addParameter('tmax', 900, @(x) isnumeric(x) && isscalar(x))
ip.addParameter('m', 5, @(x) isnumeric(x) && isscalar(x))
ip.addParameter('tau_c', 1e-2, @(x) isnumeric(x) && isscalar(x))
ip.addParameter('D', 1.2, @(x) isnumeric(x) && isscalar(x))
ip.addParameter('Vs', 3.01e-2, @(x) isnumeric(x) && isscalar(x))
ip.addParameter('rho',5e-5, @(x)isnumeric(x) && isscalar(x))
ip.addParameter('vec_flag', 0, @(x) ismember(x, [0, 1]) || x)
ip.parse(varargin{:})
u0 =       ip.Results.u0;        
dt =       ip.Results.dt;     
dx =       ip.Results.dx;       
tmax =     ip.Results.tmax;      
m =        ip.Results.m;        
tau_c =    ip.Results.tau_c;    
D =        ip.Results.D;          
Vs =       ip.Results.Vs;         
rho =      ip.Results.rho;        
vec_flag = ip.Results.vec_flag;      

% Checking stability condition for FTCS method (r < 0.5)
r = rho*dt/(dx^2);
if r >= 0.5 
    error('To ensure numerical stability rho.*dt./(dx.^2) < 0.5!, r = %f', r)
end

% Constants
rng(120)
vncx = 1.4;
eta = 0.35;
Va = -80;
P = 96.5;
R = 8.314;
temp = 300;
VFRT = Va .* P ./ (R .* temp);
cao = 12;
nao = 14;
ksat = 0.25;
kmcao = 1.3;
kmnao = 97.63;
kmnai = 12.3;
kmcai = 0.0026;
kn = 0.1;
v3 = 120;
hc3 = 2;
k3 = 0.3;
v2 = 0.5;
v_ip3 = 0.88;
ip3 = 0.2;
d5 = 0.08234;
k_pmca = 0.8;
v_pmca = 0.6;
vr = 18;
kb1 = 0.2573;
fi = 0.01;
gamma_var = 9;
fe = 0.025;
a2 = 0.2;
d1 = 0.13;
d2 = 1.049;
d3 = 0.9434;
Ks = 50;
tau_soc = 30;
ka = 0.01920;
kb = 0.2573;
kc = 0.0571;
kca = 5;
kna = 5;
tau_o = 10;
ktau = 1;
minf = ip3 ./ (ip3 + d1);

% Simulation parameters
t = 0:dt:tmax;         % time base
n = length(t);         % number of time points

% Initialization
[Ca, Cer, h, s, w, x, Na, eta_u] = deal(zeros(n, m));
Ca(1, :)  = u0(1);
Cer(1, :) = u0(2);
h(1, :)   = u0(3);
s(1, :)   = u0(4);
w(1, :)   = u0(5);
x(1, :)   = u0(6);
Na(1, :)  = u0(7);
eta_u(1, :)   = u0(8);
noise_term = randn(n, m);

% Helper functions
hinf = @(Ca) (d2 .* (ip3 + d1) ./ (ip3 + d3)) ./ ((d2 .* (ip3 + d1) ./ (ip3 + d3)) + Ca);
ninf = @(Ca) Ca ./ (Ca + d5);
ninfi = @(Ca) 1 ./ (1 + (kn ./ Ca).^2);
Kx = @(Ca, Na) kmcao .* (Na.^3) + (kmnao.^3) .* Ca + (kmnai.^3) .* cao .* (1 + Ca ./ kmcai) + kmcai .* (nao.^3) .* (1 + (Na.^3) ./ (kmnai.^3)) + (Na.^3) .* cao + (nao.^3) .* Ca;
soc_inf = @(Cer) (Ks.^4) ./ ((Ks.^4) + (Cer.^4));
winf = @(Ca) (ka ./ (Ca.^4) + 1 + (Ca.^3) ./ kb) ./ (1 ./ kc + ka ./ (Ca.^4) + 1 + (Ca.^3) ./ kb);
xinf = @(Ca, Na) 1 - 1 ./ ((1 + (Ca ./ kca).^2) .* (1 + (kna ./ Na).^2));
Jip3  = @(Ca, Cer, h) v_ip3 .* (minf.^3) .* (ninf(Ca).^3) .* (h.^3) .* (Cer - Ca);
Jserca  = @(Ca) v3 .* (Ca.^hc3) ./ ((k3.^hc3) + (Ca.^hc3));
Jleak = @(Ca, Cer) v2 .* (Cer - Ca);
Jryr = @(Ca, Cer, w) vr .* w .* (1 + (Ca.^3) ./ kb1) ./ (ka ./ (Ca.^4) + 1 + (Ca.^3) ./ kb1) .* (Cer - Ca);
Jsoc = @(s) Vs .* s;
Jncx = @(Ca, x, Na) ninfi(Ca) .* x .* (vncx .* (exp(eta .* VFRT) .* (Na.^3) .* cao -exp((eta - 1) .* VFRT) .* (nao.^3) .* Ca) ./ (Kx(Ca, Na) .* (1 + ksat .* (exp((eta - 1) .* VFRT)))));
Jpmca = @(Ca) (v_pmca .* (Ca.^2) ./ ((Ca.^2) + (k_pmca.^2)));

if vec_flag
        for kk = 1:(n - 1)                                                     
        % BCs: Neumann (∂C/∂t = 0) at x₁ and xₘ
        Ca(kk + 1, 1) = 2 .* r .* Ca(kk, 2)     + (1 - 2.*r) .* Ca(kk, 1) + fi .* (Jip3(Ca(kk, 1), Cer(kk, 1), h(kk, 1)) - Jserca(Ca(kk, 1)) + Jleak(Ca(kk, 1), Cer(kk, 1)) + Jryr(Ca(kk, 1), Cer(kk, 1), w(kk, 1)) + Jsoc(s(kk, 1)) + Jncx(Ca(kk, 1), x(kk, 1), Na(kk, 1)) - Jpmca(Ca(kk, 1))) .* dt;            
        Ca(kk + 1, m) = 2 .* r .* Ca(kk, m - 1) + (1 - 2.*r) .* Ca(kk, m) + fi .* (Jip3(Ca(kk, m), Cer(kk, m), h(kk, m)) - Jserca(Ca(kk, m)) + Jleak(Ca(kk, m), Cer(kk, m)) + Jryr(Ca(kk, m), Cer(kk, m), w(kk, m)) + Jsoc(s(kk, m)) + Jncx(Ca(kk, m), x(kk, m), Na(kk, m)) - Jpmca(Ca(kk, m))) .* dt;   
        Ca(kk + 1, 2:m-1) = r.*Ca(kk, 1:m-2) + (1 - 2.*r).*Ca(kk, 2:m-1) + r.*Ca(kk, 3:m) + fi .* (Jip3(Ca(kk, 2:m-1), Cer(kk, 2:m-1), h(kk, 2:m-1)) - Jserca(Ca(kk, 2:m-1)) + Jleak(Ca(kk, 2:m-1), Cer(kk, 2:m-1)) + Jryr(Ca(kk, 2:m-1), Cer(kk, 2:m-1), w(kk, 2:m-1)) + Jsoc(s(kk, 2:m-1)) + Jncx(Ca(kk, 2:m-1), x(kk, 2:m-1), Na(kk, 2:m-1)) - Jpmca(Ca(kk, 2:m-1))) .* dt;
        Cer(kk + 1, :)   = Cer(kk, :) + (-gamma_var .* fe .* (Jip3(Ca(kk, :), Cer(kk, :),  h(kk, :)) - Jserca(Ca(kk, :)) + Jleak(Ca(kk, :), Cer(kk, :)) + Jryr(Ca(kk, :), Cer(kk, :), w(kk, :)))) .* dt;
        h(kk + 1, :)     = h(kk, :) + ((hinf(Ca(kk, :)) - h(kk, :)) ./ (1 ./ (a2 .* ((d2 .* (ip3 + d1) ./ (ip3 + d3)) + Ca(kk, :)))) + eta_u(kk, :)) .* dt;
        s(kk + 1, :)     = s(kk, :) + ((soc_inf(Cer(kk, :)) - s(kk, :)) ./ tau_soc) .* dt;
        w(kk + 1, :)     = w(kk, :) + ((winf(Ca(kk, :)) -w(kk, :)) ./ (winf(Ca(kk, :)) ./ kc)) .* dt;
        x(kk + 1, :)     = x(kk, :) + ((xinf(Ca(kk, :), Na(kk, :)) - x(kk, :)) ./ (0.25 + tau_o ./ (1 + (Ca(kk, :) ./ ktau)))) .* dt;
        Na(kk + 1, :)    = Na(kk, :) + (-3 .* Jncx(Ca(kk, :), x(kk, :), Na(kk, :))) .* dt;
        eta_u(kk + 1, :) = eta_u(kk, :) + (-eta_u(kk, :)./tau_c) .* dt + sqrt(2.*D./tau_c) .* noise_term(kk, :);
        end
else
    for kk = 1:(n - 1)                                                     
        % BCs: Neumann (∂C/∂t = 0) at x₁ and xₘ
        Ca(kk + 1, 1) = 2 .* r .* Ca(kk, 2)     + (1 - 2.*r) .* Ca(kk, 1) + fi .* (Jip3(Ca(kk, 1), Cer(kk, 1), h(kk, 1)) - Jserca(Ca(kk, 1)) + Jleak(Ca(kk, 1), Cer(kk, 1)) + Jryr(Ca(kk, 1), Cer(kk, 1), w(kk, 1)) + Jsoc(s(kk, 1)) + Jncx(Ca(kk, 1), x(kk, 1), Na(kk, 1)) - Jpmca(Ca(kk, 1))) .* dt;            
        Ca(kk + 1, m) = 2 .* r .* Ca(kk, m - 1) + (1 - 2.*r) .* Ca(kk, m) + fi .* (Jip3(Ca(kk, m), Cer(kk, m), h(kk, m)) - Jserca(Ca(kk, m)) + Jleak(Ca(kk, m), Cer(kk, m)) + Jryr(Ca(kk, m), Cer(kk, m), w(kk, m)) + Jsoc(s(kk, m)) + Jncx(Ca(kk, m), x(kk, m), Na(kk, m)) - Jpmca(Ca(kk, m))) .* dt;      
    
        for ii = 1:m
            if (ii > 1) && (ii < m)
                Ca(kk + 1, ii) = r.*Ca(kk, ii-1) + (1 - 2.*r).*Ca(kk, ii) + r.*Ca(kk, ii+1) + fi .* (Jip3(Ca(kk, ii), Cer(kk, ii), h(kk, ii)) - Jserca(Ca(kk, ii)) + Jleak(Ca(kk, ii), Cer(kk, ii)) + Jryr(Ca(kk, ii), Cer(kk, ii), w(kk, ii)) + Jsoc(s(kk, ii)) + Jncx(Ca(kk, ii), x(kk, ii), Na(kk, ii)) - Jpmca(Ca(kk, ii))) .* dt;
            end    
            Cer(kk + 1, ii)   = Cer(kk, ii) + (-gamma_var .* fe .* (Jip3(Ca(kk, ii), Cer(kk, ii),  h(kk, ii)) - Jserca(Ca(kk, ii)) + Jleak(Ca(kk, ii), Cer(kk, ii)) + Jryr(Ca(kk, ii), Cer(kk, ii), w(kk, ii)))) .* dt;
            h(kk + 1, ii)     = h(kk, ii) + ((hinf(Ca(kk, ii)) - h(kk, ii)) ./ (1 ./ (a2 .* ((d2 .* (ip3 + d1) ./ (ip3 + d3)) + Ca(kk, ii)))) + eta_u(kk, ii)) .* dt;
            s(kk + 1, ii)     = s(kk, ii) + ((soc_inf(Cer(kk, ii)) - s(kk, ii)) ./ tau_soc) .* dt;
            w(kk + 1, ii)     = w(kk, ii) + ((winf(Ca(kk, ii)) -w(kk, ii)) ./ (winf(Ca(kk, ii)) ./ kc)) .* dt;
            x(kk + 1, ii)     = x(kk, ii) + ((xinf(Ca(kk, ii), Na(kk, ii)) - x(kk, ii)) ./ (0.25 + tau_o ./ (1 + (Ca(kk, ii) ./ ktau)))) .* dt;
            Na(kk + 1, ii)    = Na(kk, ii) + (-3 .* Jncx(Ca(kk, ii), x(kk, ii), Na(kk, ii))) .* dt;
            eta_u(kk + 1, ii) = eta_u(kk, ii) + (-eta_u(kk, ii)./tau_c) .* dt + sqrt(2.*D./tau_c) .* noise_term(kk, ii);
        end
    end
end
end