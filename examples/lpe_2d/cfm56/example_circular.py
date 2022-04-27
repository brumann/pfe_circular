import numpy as np
from pfe import Mesh, Model, Constant, Vector
from pfe.interpolation import Lagrange2
from pfe.models import lpe_2d_circular
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, interp2d, griddata
import scipy.integrate as it
import scipy.optimize as opt
import time
from numba import jit


"""This code is a example of how to use the PFE code for a circular duct
 The geometry chosen is taken from Ovenden's paper https://doi.org/10.2514/6.2004-2945"""


# GEOMETRY
# --------
length = 2
r1 = lambda x: np.where(x>0, np.maximum(0.001, 0.64212 - (0.04777 + 0.98234 * (x / length) ** 2) ** 0.5),
                        0.64212 - (0.04777 ) ** 0.5)
r2 = lambda x:np.where(x<0, 1,  np.where(x > length, 1. - 0.18453 + 0.10158, 1. - 0.18453 * (x / length) ** 2 + 0.10158 *
             (np.exp(-11. * (1. - (x / length))) - np.exp(-11.)) / (1. - np.exp(-11.))))

# Parameters for the extraction
nb_x = 200
nb_r = 100
x_min = 1e-5
x_max = 3-1e-5
x = np.linspace(x_min, x_max, nb_x)


def external_radius(xy):
    return( r2(xy[0,:]) )
def internal_radius(xy):
    return( r1(xy[0,:]) )

# FLOW
# --------
def flow_d0(xy):
    return( np.ones_like(xy[0,:]) )
def flow_u0(xy):
    return(np.zeros_like(xy[0,:]) )
def flow_c0(xy):
    return(np.ones_like(xy[0,:]) )
def flow_v1(xy):
    return(np.zeros_like(xy[0,:]) )

# ACOUSTIC
# --------
num_modes = 10
mode_n = 0
mode_m = 10
omega = 16

mesh = Mesh('duct.msh', num_dim=2)

model = Model()
model.parameters['omega'] = Constant(omega)
model.parameters['m'] = Constant(mode_m)
model.parameters['r2'] = Lagrange2(mesh, external_radius)
model.parameters['r1'] = Lagrange2(mesh, internal_radius)
model.parameters['rho0'] = Lagrange2(mesh, flow_d0)
model.parameters['c0'] = Lagrange2(mesh, flow_c0)
model.parameters['u0'] = Lagrange2(mesh, flow_u0)  #Lagrange2(mesh, lambda x: x[0]*10)
model.parameters['v0'] = Lagrange2(mesh, flow_v1) #Lagrange2(mesh, lambda x: x[1]*20)

model.fields['phi'] = Lagrange2(mesh)
model.fields['R'] = Vector(num_modes)
model.fields['T'] = Vector(num_modes)
A_in = np.zeros((num_modes,))
A_in[mode_n] = 1.0

model.add_term(lpe_2d_circular.Main(mesh.group(0)))
model.add_term(lpe_2d_circular.Wall(mesh.group([1, 3, 4, 5])))
model.add_term(lpe_2d_circular.DuctModes(mesh.group(6), 'R', modes_i=A_in))
model.add_term(lpe_2d_circular.DuctModes(mesh.group(2), 'T'))

model.declare_fields()
model.build()
model.solve()

plt.figure(1)
model.fields['phi'].plot(np.real(model.solution))
plt.xlabel('r')
plt.ylabel('x')
plt.axis('equal')

plt.figure(2)
model.fields['phi'].plot(np.abs(model.solution))
plt.xlabel('r')
plt.ylabel('x')
plt.axis('equal')

plt.show()


print("end ....")
