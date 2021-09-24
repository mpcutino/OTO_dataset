from scipy.integrate import odeint
from scipy.special import hankel2
from math import atan2
import cmath

from ospa.constants import *
from ospa.flight_state import FlightState


def AleteoBad(f, Ub, h0, a0, phi, a, alpha_m, AR, t0, phase0, t):
    '''
    :param f: Frecuencia
    :param h0: Amplitud de heaving
    :param Ub: Velocidad
    :param a0: Amplitud de pitching
    :param phi: Desfase entre heaving y pitching
    :param a: Punto sobre el que hace pitching (perfil entre -1 y 1)
    :param alpha_m: angulo de ataque medio
    :param AR: Densidad
    :param t: Instante de tiempo
    :param phase0:
    :param t0: tiempo inicial
    '''
    omega = 2 * pi * f
    k = omega / Ub

    zphi = complex(0, phi)
    alpha0 = a0 * cmath.exp(zphi)  # Amplitud de pitching con desfase

    phase = omega * (t - t0) + phase0

    zomega = complex(0, omega)
    zphase = complex(0, phase)
    G0 = 2 * pi * (Ub * alpha0 - zomega * h0 - zomega * (a - 1 / 2) * alpha0)
    Gamma0 = G0 * cmath.exp(zphase)

    ## Sustentacion

    zbessel = complex(hankel2(1, k), hankel2(0, k))
    C = hankel2(1, k) / zbessel
    F = C.real
    G = C.imag

    CL = 2 * pi * ((alpha_m + k * h0 * (G * cos(phase) + F * sin(phase)) + a0 * (
                cos(phase + phi) * (F + G * k * (a - 0.5)) + sin(phase + phi) *
                (-G + F * k * (a - 0.5)))) * AR / (AR + 2) + (
                               k * h0 * k / 2. * cos(phase) + a0 * cos(phase + phi) * a * k ** 2. / 2 - a0 * sin(
                           phase + phi) * k / 2) * AR / (AR + 1))
    CLM = 2 * pi * alpha_m * AR / (AR + 2)

    ## Empuje
    zk = complex(0, k)
    zbessel = complex(hankel2(1, k), hankel2(0, k))
    C1 = 1 / k * cmath.exp(-zk) / zbessel
    F1 = C1.real
    G1 = C1.imag

    cT1 = pi * a0 * k * sin(phase + phi) * (k * h0 * sin(phase) + a * k * a0 * sin(phase + phi) + a0 * cos(phase + phi))
    cT2 = 4 * (a0 * (cos(phase + phi) + k * (a - 0.5) * sin(phase + phi)) + k * h0 * sin(phase)) * (
                F1 * (k * h0 * cos(phase) +
                      a0 * (-2. * sin(phase + phi) + k * (a - 1) * cos(phase + phi))) + G1 * (-k * h0 * sin(phase) +
                                                                                              a0 * (-2. * cos(
                    phase + phi) - k * (a - 1) * sin(phase + phi))) - pi / 2 * a0 * (
                            F * cos(phase + phi) - G * sin(phase + phi)))

    zphase = complex(0, phase)
    zalpha = complex(alpha_m, alpha0 * cmath.exp(zphase))
    CT = -zalpha.real * CL + cT1 * AR / (AR + 1) + cT2 * AR / (AR + 2)
    CTM = -alpha_m * CLM + AR / (AR + 2) * (-2 * (k * h0) ** 2 * G1 - a0 ** 2 * (
                2 * pi * F + 2 * a * k * F1 + 2 * (2 + k ** 2 * (a - 1) * (a - 0.5)) * G1) +
                                            a0 * k * h0 * (-(2 * F1 + k * (4 * a - 3) * G1) * cos(phi) + (
                        2 * pi * F + k * F1 + 6. * G1) * sin(phi)))

    return CL, CT, CTM, CLM


def ode_dydt(delta_t, fa):

    def model(Y, t, dalpha):

        u = Y[0]
        w = Y[1]
        q = Y[2]
        theta = Y[3]
        x = Y[4]
        h = Y[5]

        alpha = atan2(w, u)
        Ub = sqrt(u ** 2 + w ** 2)

        if not fa:
            if alpha < radians(10):
                CLs = CL_alpha * (alpha + alpha_w)
                CLns = CL_alpha * (1.5 * (dalpha + dalpha_w) - 2 * lw / c * q) / Ub
                CL = CLs + CLns
            else:
                CLs = CL_alpha * radians(10)
                CLns = CL_alpha * (0.5 * (dalpha + dalpha_w)) / Ub
                CL = CLs + CLns

            depsilon = 0.2
            CT = 0
            CDi = k * CLs ** 2

        else:
            CL, CT, _, _ = AleteoBad(fa, Ub, h0ad, a0, phi, a, alpha, AR, 0, 0, t)
            depsilon = 0
            CDi = k * CL ** 2
            CLs = CL

        if alpha + delta_t - depsilon * CLs / CL_alpha < radians(25):
            CLts = CL_alpha_t * (alpha + delta_t - depsilon * CLs / CL_alpha)
            if fa:
                CLtns = CL_alpha_t * (1.5 * (dalpha + ddelta_t) - lt / Lc * q) / Ub
            else:
                CLtns = CL_alpha_t * (1.5 * (dalpha + ddelta_t) - 2 * lt / c * q) / Ub
            CLt = CLts + CLtns
        else:
            CLts = CL_alpha_t * radians(25)
            CLtns = CL_alpha_t * (0.5 * (dalpha + ddelta_t)) / Ub
            CLt = CLts + CLtns

        CDit = k_t * CLts ** 2

        Fxw = sin(alpha) * CL - cos(alpha) * (CDi + CD_0 - CT)
        Fxt = Lambda * (sin(alpha) * CLt - cos(alpha) * (CDit + CD_0t))
        Fxb = -cos(alpha) * Li
        Fzw = -cos(alpha) * CL - sin(alpha) * (CDi + CD_0 - CT)
        Fzt = Lambda * (-cos(alpha) * CLt - sin(alpha) * (CDit + CD_0t))
        Fzb = -sin(alpha) * Li

        du = -q * w + 1 / 2 / M * ((u ** 2 + w ** 2) * (Fxw + Fxt + Fxb) - sin(theta))
        dw = q * u + 1 / 2 / M * ((u ** 2 + w ** 2) * (Fzw + Fzt + Fzb) + cos(theta))
        dq = X * (u ** 2 + w ** 2) * (-Fzw - L * Fzt - RHL * (Fxw + H * Fxt))
        dtheta = q
        dx = u * cos(theta) + w * sin(theta)
        dh = -w * cos(theta) + u * sin(theta)

        dalpha = (u * dw - w * du) / Ub**2

        return [du, dw, dq, dtheta, dx, dh]

    return model


def get_next_state(state, action, final_state, h=0.03):
    ''' state: Flight state,
        action: (tail angle, frequency, time step),
        final_state: Final state
        h: model step'''
    
    angle, fa, time = action
    model = ode_dydt(angle, fa)
    tSpan = [h * i for i in range(int(time/h))]
    
    y0, cost0 = [state.u, state.v, state.omega, state.theta, state.x, state.z], state.cost
    # try:
    Y = odeint(model, y0, tSpan, args=(0,))
    # except:
    #     print("odeint esta hervi'o")
    #     return -1

    x_0 = Y[0][-2]
    aux = False
    for y_i in Y:
        if y_i[-2] >= x_0:
            x_0 = y_i[-2]
        else:
            aux = True
            break
    if aux:
        print("Model Error")
        return -1
        
    aux = []
    if final_state.x < Y[-1][-2]:
        for yi in Y:
            if yi[-2] >= final_state.x:
                break
            aux.append(yi)
    everything_ok = len(aux) == 0
    if everything_ok:
        aux = Y
    elif len(aux) == 1:
        # there is only one state, the one that you came from
        return -1
        
    tim = len(aux)/len(Y)
    Y = aux
    P_w, P_t = k_aero*(fa/tc)**3, c_e
    power = (P_t + P_w)*(tim*time*tc)
    
    u, v, omega, theta, x, z = Y[-1]
    new_state = FlightState(angle, fa, u, v, theta, omega, x, z)
    new_state.increment_cost(cost0, power)
    
    return new_state
