from math import pi, sin, cos, sqrt, radians


## Datos geometricos y de pesos

m = 0.367     #[kg]
g = 9.8       #[m/s]
rho = 1.225     #[kg/m^3]
S = 0.324     #[m^2]
b = 1.2       #[m]
c = S/b       #[m]
AR = b**2/S     #[ND]
S_t = 0.09      #[m^2]
b_t = 0.46      #[m]
AR_t = b_t**2/S_t #[ND]
k_aero = 2.5
c_e = 5
A = radians(26.5)

# xcg = 149.137e-3 #[m]
xcg = 119.137e-3 #[m]
zcg = 5.814e-3 #[m]

xw = 0.09
lw = xcg-xw
zw = -0.05
hw = zcg-zw

xt = 0.570
lt = xcg-xt
zt = 15e-3
ht = zcg-zt

Iyy = 0.008 #[kg*m^2]


### Datos aerodinamicos

CL_alpha = 2*pi*AR/(AR+2)
CL_alpha_t = pi/2*AR_t
Li = 0.0051
CD_0 = 0.018
CD_0t = 0.021
k = 1/pi/AR
k_t = 1/pi/AR_t
# delta_t = -3*pi/180
CT = 0
depsilon = 0.2
alpha_w = 0
dalpha_w = 0
ddelta_t = 0 ## Variable que cambia con la cola, estudiar!!

h0 = 0.4*sin(10*pi/180) #[m]
a0 = 0                  #[rad]
a = -0.5                #[ND]
phi=0                   #rad
dalpha = 0

### Parametros adimensionales

Uc = sqrt(2*m*g/(rho*S))
Lc = c/2
tc = Lc/Uc
Lambda = S_t/S
L = lt/lw
H = ht/hw
RHL = hw/lw
M = 2*m/rho/S/c
X = rho/8*S*c**2*lw/Iyy
h0ad = h0/Lc