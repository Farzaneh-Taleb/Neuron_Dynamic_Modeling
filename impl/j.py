"""
@author Farzaneh.Tlb
5/13/19 4:39 PM
Implementation of .... (Fill this line)
"""

"""
@author Farzaneh.Tlb
5/13/19 10:52 AM
Implementation of .... (Fill this line)
"""
import scipy as sp
import pylab as plt
from scipy.integrate import odeint
import numpy as np
## Full Hodgkin-Huxley Model (copied from Computational Lab 2)

# Constants
C_m = 1  # membrane capacitance, in uF/cm^2
g_Na = 120.0  # maximum conducances, in mS/cm^2
g_K = 36.0
g_L = 0.3
# E_Na = 50.0  # Nernst reversal potentials, in mV
E_Na = 55.0  # Nernst reversal potentials, in mV
# E_K = -77.0
E_K = -72.0
E_L = -54.387
# E_L = 10.6


# Channel gating kinetics
# Functions of membrane voltage
def alpha_m(V): return 0.1 * (V + 40.0) / (1.0 - sp.exp(-(V + 40.0) / 10.0))


def beta_m(V):  return 4.0 * sp.exp(-(V + 65.0) / 18.0)


def alpha_h(V): return 0.07 * sp.exp(-(V + 65.0) / 20.0)


def beta_h(V):  return 1.0 / (1.0 + sp.exp(-(V + 35.0) / 10.0))


def alpha_n(V): return 0.01 * (V + 55.0) / (1.0 - sp.exp(-(V + 55.0) / 10.0))


def beta_n(V):  return 0.125 * sp.exp(-(V + 65) / 80.0)


# Membrane currents (in uA/cm^2)
#  Sodium (Na = element name)
def I_Na(V, m, h): return g_Na * m ** 3 * h * (V - E_Na)


#  Potassium (K = element name)
def I_K(V, n):  return g_K * n ** 4 * (V - E_K)


#  Leak
def I_L(V):     return g_L * (V - E_L)


# External current
def I_inj(x, t):  # step up 10 uA/cm^2 every 100ms for 400ms
     y= x * (t > 20.0) - x * (t > 20.2)
     # y= x * (t > 0.2) - x * (t > 0.4) +  x * (t > 0.6) - x * (t > 0.8) +  x * (t > 0.6) - x * (t > 0.8)
     # y= x * (t > 0.2) - x * (t > 0.4)
     return y
     # return 10*t


# The time to integrate over
t = sp.arange(0.0,100, .01)


# Integrate!
def dALLdt(X, t , C_m):
    V, m, h, n = X

    # calculate membrane potential & activation variables
    dVdt = (I_inj(20 , t) - I_Na(V, m, h) - I_K(V, n) - I_L(V)) / C_m
    dmdt = alpha_m(V) * (1.0 - m) - beta_m(V) * m
    dhdt = alpha_h(V) * (1.0 - h) - beta_h(V) * h
    dndt = alpha_n(V) * (1.0 - n) - beta_n(V) * n
    return dVdt, dmdt, dhdt, dndt

def get_g_k(n):
    return g_K*n**4

def get_g_Na(m , h):
    return g_Na*(m**3)*h



# i = parameter_fitting(0,20)
# print("ii" , i )
# X = odeint(dALLdt, [-65, 0.05, 0.6, 0.32], t)
plt.figure()
# plt.subplot(2, 1, 1)
plt.title('Hodgkin-Huxley Neuron')
plt.ylabel('V (mV)')

i=0
for c in np.arange(0.01,1,0.1):
    i += 1
    plt.subplot(10, 1, i)
    X = odeint(dALLdt, [-60, 0.05, 0.6, 0.3], t, (c,))
    V = X[:, 0]
    m = X[:, 1]
    h = X[:, 2]
    n = X[:, 3]
    ina = I_Na(V, m, h)
    ik = I_K(V, n)
    il = I_L(V)
    plt.plot(t[2000:], V[2000:])


plt.subplot(11, 1, 11)
plt.plot(t[2000:], I_inj(20 , t)[2000:], 'k')
plt.xlabel('t (ms)')
plt.ylabel('$I_{inj}$ ($\\mu{A}/cm^2$)')
plt.ylim(-1, 21)

plt.show()