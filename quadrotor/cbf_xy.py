"""dynamics.py
Simulate Simple Quadrotor Dynamics

`python dynamics.py` to see hovering drone
"""

import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
# from visualize_dynamics import *
from sim_utils import *
from controller import *
import time
from cvxopt import matrix
from cvxopt import solvers
solvers.options['show_progress'] = False
# Physical constants
g = 9.81  # FLU
m = 0.5
L = 0.25
k = 3e-6
b = 1e-7
I = np.diag([5e-3, 5e-3, 10e-3])
kd = 0.001
dt = 0.1
k_1 = 12
k_2 = 4

dist = 0.2

cx = 0
cy = 0
px = 1
py = 1

kxy_3 = 10
kxy_2 = 1
kxy_1 = 1
kxy_0 = 1


maxrpm = 10000
maxthrust = k*np.sum(np.array([maxrpm**2] * 4))
param_dict = {"g": g, "m": m, "L": L, "k": k, "b": b, "I": I,
              "kd": kd, "dt": dt, "maxRPM": maxrpm, "maxthrust": maxthrust}


def init_state():
    """Initialize state dictionary. """
    state = {"x": np.array([0.2, 0.5, 1]),
             "xdot": np.zeros(3,),
             "xdd": np.zeros(3,),
             "theta": np.radians(np.array([0, 0, 0])),  # ! hardcoded
             "thetadot": np.radians(np.array([0, 0, 0]))  # ! hardcoded
             }
    return state

#
# def qp_z(state, u_hat):
#     z = state['x'][2]
#     z_dot = state['xdot'][2]
#     theta = state['theta']
#     matrix_rotation = get_rot_matrix(theta)
#     R_33 = matrix_rotation[2][2]
#     P = matrix([1])
#     q = matrix([0])
#     P = matrix(P, (1, 1), 'd')
#     q = matrix(q, (1, 1), 'd')
#     h = matrix([2 * z * R_33 / m * u_hat - 2 * z * g - 2 * z_dot ** 2 + k_1 * (- 2 * z * z_dot) + k_2 * (1 - z ** 2)])
#     G = matrix([2 * z * R_33 / m])
#     sol = solvers.qp(P,q,G,h)
#     sol['x']
#     u_bar = sol['x']
#     u = u_bar + u_hat
#
#     return u[0]


def alpha(state, i):
    [x, y, z] = state['x']
    return np.array([[(x - cx) ** i / px ** 4, (y - cy) ** i / py ** 4]])


def Lf0h(state):
    [x, y, z] = state['x']
    return 1 - (x - cx) ** 4 / px ** 4 - (y - cy) ** 4 / py ** 4


def Lf1h(state):

    [xd, yd, zd] = state['xdot']
    alpha_3 = alpha(state, 3)
    hd_mat = - 4 * np.dot(alpha_3, np.array([[xd],
                                      [yd]]))
    return hd_mat[0][0]


def Lf2h(state):

    [xd, yd, zd] = state['xdot']
    [xdd, ydd, zdd] = state['xdd']
    alpha_3 = alpha(state, 3)
    alpha_2 = alpha(state, 2)
    hdd_mat = - 4 * np.dot(alpha_3, np.array([[xdd],
                                             [ydd]]))\
              - 12 * np.dot(alpha_2, np.array([[xd ** 2],
                                                [yd ** 2]]))
    return hdd_mat[0][0]


def Lf3h(state, F):

    [xd, yd, zd] = state['xdot']
    [xdd, ydd, zdd] = state['xdd']
    [xddd, yddd] = xydddt(state, F)
    alpha_3 = alpha(state, 3)
    alpha_2 = alpha(state, 2)
    alpha_1 = alpha(state, 1)
    hddd_mat = - 4 * np.dot(alpha_3, np.array([[xddd],
                                             [yddd]]))\
              - 36 * np.linalg.multi_dot([alpha_2, np.array([[xd, 0],
                                                [0, yd]]),np.array([[xdd ** 2],
                                                                     [ydd ** 2]])])\
                  - 24 * np.dot(alpha_1, np.array([[xd ** 3],
                                                [yd ** 3]]))

    return hddd_mat[0][0]


def Lf4h(state, F):
    [xd, yd, zd] = state['xdot']
    [xdd, ydd, zdd] = state['xdd']
    [xddd, yddd] = xydddt(state, F)
    alpha_3 = alpha(state, 3)
    alpha_2 = alpha(state, 2)
    alpha_1 = alpha(state, 1)
    alpha_0 = alpha(state, 0)
    J = get_J(state)
    L4hf_mat = 4 * F / m * np.dot(alpha_3, J)\
              - 48 * np.linalg.multi_dot([alpha_2, np.array([[xd, 0],
                                                [0, yd]]),np.array([[xddd],
                                                                     [yddd]])])\
                - 36 * np.dot(alpha_2, np.array([[xdd ** 2],
                                                [ydd ** 2]]))\
                 - 144 * np.linalg.multi_dot([alpha_1, np.array([[xd, 0],
                                                [0, yd]]),np.array([[xdd],
                                                                     [ydd]])])\
                  - 24 * np.dot(alpha_0, np.array([[xd ** 4],
                                                [yd ** 4]]))
    return L4hf_mat[0][0]



# def hdd(state):
#
#     [x, y, z] = state['x']
#     [xd, yd, zd] = state['xdot']
#     [xdd, ydd, zdd] = state['xdd']
#
#     return - 4 * (x - cx) ** 3 / px ** 4 * xdd - 12 * (x - cx) ** 2 / px ** 4 * xd ** 2\
#             - 4 * (y - cy) ** 3 / py ** 4 * ydd - 12 * (y - cy) ** 2 / py ** 4 * yd ** 2


def r_3_xy_dot(state):

    [phi, the, psi] = state['theta']
    [ophi, othe, opsi] = state['thetadot']
    cphi = np.cos(phi)
    sphi = np.sin(phi)
    cthe = np.cos(the)
    sthe = np.sin(the)
    cpsi = np.cos(psi)
    spsi = np.sin(psi)
    return np.array([[-spsi * opsi * sthe + cpsi * cthe * othe - sthe * othe * sphi * spsi + cthe * cphi * ophi * spsi + cthe * cphi * ophi * spsi],
                     [cpsi * opsi * sthe + spsi * cthe * othe + spsi * opsi * cthe * sphi + cpsi * sthe * othe * sphi - cpsi * cthe * cphi * ophi]])


# def xydddt(state, F):
#
#     R_3_xy_dot = r_3_xy_dot(state)
#     # print(R_3_xy_dot)
#     [xddd, yddd] = [-F / m * R_3_xy_dot[0][0], -F / m * R_3_xy_dot[1][0]]
#     return [xddd, yddd]

def xydddt(state, F):

    A = get_A(state)
    V = get_V(state)
    R = get_rot_matrix(state['theta'])
    R_33 = R[2][2]
    # print(R_3_xy_dot)
    mat = -F / m * R_33 * np.dot(V, A)
    return [mat[0][0], mat[1][0]]


def hddd(state):
    [x, y, z] = state['x']
    [xd, yd, zd] = state['xdot']
    [xdd, ydd, zdd] = state['xdd']
    [xddd, yddd] = xydddt(state)
    return -4 * (x - cx) ** 3 / px ** 4 * xddd - 36 * (x - cx) ** 2 / px ** 4 * xd * xdd - 24 * (x - cx) / px ** 4 * xd ** 3\
            -4 * (y - cy) ** 3 / py ** 4 * yddd - 36 * (y - cy) ** 2 / py ** 4 * yd * ydd - 24 * (y - cy) / py ** 4 * yd ** 3



def get_V(state):
    theta = state['theta']
    R = get_rot_matrix(theta)
    W = np.array([[R[1][0], -R[0][0]],
                  [R[1][1], -R[0][1]]])
    V = np.linalg.inv(W)
    return V


def get_A(state):

    theta = state['theta']
    R = get_rot_matrix(theta)
    W = np.array([[R[1][0], -R[0][0]],
                  [R[1][1], -R[0][1]]])
    R_33 = R[2][2]
    R_3_xy_dot = r_3_xy_dot(state)
    A = 1 / R_33 * np.dot(W, R_3_xy_dot)
    # print('A', A)
    return A


def get_v_dot(state):

    theta = state['theta']
    [phi, the, psi] = state['theta']
    [ophi, othe, opsi] = state['theta']
    cphi = np.cos(phi)
    sphi = np.sin(phi)
    cthe = np.cos(the)
    sthe = np.sin(the)
    cpsi = np.cos(psi)
    spsi = np.cos(psi)
    R = get_rot_matrix(theta)
    W = np.array([[R[1][0], -R[0][0]],
                  [R[1][1], -R[0][1]]])
    W_inv = np.linalg.inv(W)
    R_21_dot = -sthe * othe * spsi + cthe * cpsi * opsi - spsi * opsi * sphi * sthe + cpsi * cphi * ophi * sthe + cpsi * sphi * cthe * othe
    R_11_dot = -spsi * opsi * cthe - cpsi * sthe * othe - cphi * ophi * spsi * sthe - sphi * cpsi * opsi * sthe - sphi * spsi * cthe * othe
    R_22_dot = -sphi * ophi * cpsi - cphi * spsi * opsi
    R_12_dot = sphi * ophi * spsi - cphi * cpsi * opsi
    W_dot = np.array([[R_21_dot, -R_11_dot],
                      [R_22_dot, -R_12_dot]])
    V_dot = np.linalg.multi_dot([-W_inv, W_dot, W_inv])

    return V_dot


def get_J1(state):

    [phi, the, psi] = state['theta']
    [ophi, othe, opsi] = state['thetadot']
    cphi = np.cos(phi)
    sphi = np.sin(phi)
    cthe = np.cos(the)
    sthe = np.sin(the)
    A = get_A(state)
    V = get_V(state)
    R_33_dot = - sphi * ophi * cthe - cphi * sthe * othe
    J = R_33_dot * np.dot(V, A)
    # print('J1', J)

    return J


def get_J2(state):
    theta = state['theta']
    R = get_rot_matrix(theta)
    R_33 = R[2][2]
    V_dot = get_v_dot(state)
    A = get_A(state)
    # print('A', A)
    J = R_33 * np.dot(V_dot, A)
    # print('J2', J)
    return J


def get_J3(state):
    I_x = I[0][0]
    I_y = I[1][1]
    I_z = I[2][2]
    [ophi, othe, opsi] = state['thetadot']
    theta = state['theta']
    R = get_rot_matrix(theta)
    R_33 = R[2][2]
    V = get_V(state)
    I_omega_mat = np.array([[(I_y - I_z) / I_x * othe * opsi],
                            [(I_z - I_x) / I_y * ophi * opsi]])

    J = R_33 * np.dot(V, I_omega_mat)
    # print('J3', J)
    return J


def get_J(state):

    J1 = get_J1(state)
    J2 = get_J2(state)
    J3 = get_J3(state)
    J = J1 + J2 + J3
    # print('J', J)
    return J


def get_L(state):
    I_x = I[0][0]
    I_y = I[1][1]
    theta = state['theta']
    R = get_rot_matrix(theta)
    R_33 = R[2][2]
    V = get_V(state)
    I_inv = np.array([[1 / I_x, 0],
                      [0, 1 / I_y]])
    L = R_33 * np.dot(V, I_inv)

    return L


def LgLf3h_uhat(state, u):
    F = u[0]
    u_hat = np.array([[u[1]],
                      [u[2]]])
    L = get_L(state)
    alpha_3 = alpha(state, 3)

    mul_mat = 4 * F / m * np.linalg.multi_dot([alpha_3, L, u_hat])

    return mul_mat[0][0]




def qp_xy(state, u):
    F = u[0]
    u_hat = np.array([[u[1]],
                      [u[2]]])
    h = Lf0h(state)
    dh = Lf1h(state)
    ddh = Lf2h(state)
    dddh = Lf3h(state, F)
    lf4h = Lf4h(state, F)

    lglf3h_hat = LgLf3h_uhat(state, u)
    print('hs:', h, dh, ddh, dddh, lf4h, lglf3h_hat)
    H_qp = lf4h + lglf3h_hat + kxy_3 * dddh + kxy_2 * ddh + kxy_1 * dh + kxy_0 * h
    # if H_qp < 0:
    #     H_qp = -H_qp
    alpha_3 = alpha(state, 3)
    G_qp = - 4 * F / m * np.dot(alpha_3, L)
    P = matrix([[1, 0], [0, 1]])
    Q = matrix([0, 0])
    P = matrix(P, (2, 2), 'd')
    Q = matrix(Q, (2, 1), 'd')
    H = matrix([H_qp])
    G = matrix(G_qp)
    # print('H', H)
    # print('G', G)
    # print('u_norm', u[1], u[2])
    sol = solvers.qp(P,Q,G,H)
    [ux_bar, uy_bar] = sol['x']
    ux = ux_bar + u_hat[0][0]
    uy = uy_bar + u_hat[1][0]
    u_new = u

    # print('u_cbf', [ux, uy])
    # print('x', state['x'])
    # print('xd', state['xdot'])
    # print('xdd', state['xdd'])
    # print('theta', state['theta'])
    # print('omega', state['thetadot'])
    return ux, uy



class QuadDynamics:
    def __init__(self):
        self.param_dict = param_dict

    def step_dynamics(self, state, u):

        # Compute angular velocity vector from angular velocities
        omega = self.thetadot2omega(state["thetadot"], state["theta"])

        # Compute linear and angular accelerations given input and state
        a = self.calc_acc(u, state["theta"], state["xdot"], m, g, k, kd)
        omegadot = self.calc_ang_acc(u, omega, I, L, b, k)

        # Compute next state
        omega = omega + dt * omegadot
        thetadot = self.omega2thetadot(omega, state["theta"])
        theta = state["theta"] + dt * state["thetadot"]
        xdot = state["xdot"] - dt * a
        x = state["x"] + dt * xdot

        # Update state dictionary
        state["x"] = x
        state["xdot"] = xdot
        state["xdd"] = a
        state["theta"] = theta
        state["thetadot"] = thetadot

        return state

    def compute_thrust(self, u, k):


        u = np.clip(u, 0, self.param_dict["maxRPM"]**2)
        T = np.array([0, 0, k*np.sum(u)])
        # print("u", u)
        # print("T", T)

        return T

    def calc_torque(self, u, L, b, k):

        tau = np.array([
            L * k * (u[0]-u[2]),
            L * k * (u[1]-u[3]),
            b * (u[0]-u[1] + u[2]-u[3])
        ])

        return tau

    def calc_acc(self, u, theta, xdot, m, g, k, kd):

        gravity = np.array([0, 0, g])
        R = get_rot_matrix(theta)
        thrust = np.array([0, 0, u[0]])
        # thrust = self.compute_thrust(u, k)
        T = np.dot(R, thrust)
        Fd = -kd * xdot
        a = gravity - 1/m * T
        return a

    def calc_ang_acc(self, u, omega, I, L, b, k):

        # Calculate torque given control input and physical constants
        # tau = self.calc_torque(u, L, b, k)
        tau = np.array([u[1], u[2], u[3]])
        # Calculate body frame angular acceleration using Euler's equation
        omegaddot = np.dot(np.linalg.inv(
            I), (tau - np.cross(omega, np.dot(I, omega))))

        return omegaddot

    def omega2thetadot(self, omega, theta):
        phi = theta[0]
        the = theta[1]
        psi = theta[2]

        mult_matrix = np.array(
            [
                [np.cos(the), 0, -np.cos(phi) * np.sin(the)],
                [0, 1, np.sin(phi)],
                [np.sin(the), 0, np.cos(the)*np.cos(phi)]
            ]

        )
        #
        mult_inv = np.linalg.inv(mult_matrix)
        thetadot = np.dot(mult_inv, omega)
        return thetadot

    def thetadot2omega(self, thetadot, theta):

        phi = theta[0]
        the = theta[1]
        psi = theta[2]

        mult_matrix = np.array(
            [
                [np.cos(the), 0, -np.cos(phi) * np.sin(the)],
                [0, 1, np.sin(phi)],
                [np.sin(the), 0, np.cos(the)*np.cos(psi)]
            ]

        )

        w = np.dot(mult_matrix, thetadot)

        return w


def basic_input():
    """Return arbritrary input to test simulator"""
    return np.power(np.array([950, 700, 700, 700]), 2)


class QuadHistory():
    """Keeps track of quadrotor history for plotting."""

    def __init__(self):
        self.hist_theta = []
        self.hist_des_theta = []
        self.hist_thetadot = []
        self.hist_xdot = [[0, 0, 0]]
        self.hist_xdotdot = []
        self.hist_x = []
        self.hist_y = []
        self.hist_z = []
        self.hist_pos = []
        self.hist_des_xdot = []
        self.hist_des_x = []

    def update_history(self, state, des_theta_deg_i, des_xdot_i, des_x_i, dt):
        """Appends current state and desired theta for plotting."""
        x = state["x"]
        xdot = state["xdot"]
        xdotdot = (xdot - np.array(self.hist_xdot[-1])) / dt
        self.hist_x.append(x[0])
        self.hist_y.append(x[1])
        self.hist_z.append(x[2])
        self.hist_theta.append(np.degrees(state["theta"]))
        self.hist_thetadot.append(np.degrees(state["thetadot"]))
        # if des_xdot_i is None:
        #     des_xdot_i = [0,0,0]
        # if des_x_i is None:
        #     des_x_i = [0, 0, 0]
        self.hist_des_theta.append(des_theta_deg_i)
        self.hist_xdot.append(state["xdot"])
        self.hist_des_xdot.append(des_xdot_i)
        self.hist_des_x.append(des_x_i)
        self.hist_pos.append(x)

        self.hist_xdotdot.append(state["xdd"])


def main():

    t_start = time.time()

    # Set desired position


    # Initialize Robot State
    state = init_state()

    # Initialize quadrotor history tracker
    quad_hist = QuadHistory()

    # Initialize visualization
    fig, (ax1, ax2, ax3) = plt.subplots(3)

    # Initialize controller errors
    integral_p_err = None
    integral_v_err = None

    # Initialize quad dynamics
    quad_dyn = QuadDynamics()

    sim_iter = 300
    xs = list()
    ys = list()
    zs = list()
    # Step through simulation
    for i in range(sim_iter):
        print('==================================================')
        des_pos = np.array([2 * np.sin(i * dt), 0.5, 0])
        des_vel, integral_p_err = pi_position_control(
            state, des_pos, integral_p_err)
        des_thrust, des_theta, integral_v_err = pi_velocity_control(
            state, des_vel, integral_v_err)  # attitude control
        des_theta_deg = np.degrees(des_theta)  # for logging
        # u = pi_attitude_control(
        #     state, des_theta, des_thrust, param_dict)
        # print(u)
        w_square = pi_attitude_control(
            state, des_theta, des_thrust, param_dict)  # attitude control
        # Step dynamcis and update state dict
        # Torque = k * (w_square[0] + w_square[1] + w_square[2] + w_square[3])
        Torque = k * np.sum(w_square)
        tau = np.array([
            L * k * (w_square[0]-w_square[2]),
            L * k * (w_square[1]-w_square[3]),
            b * (w_square[0]-w_square[1] + w_square[2]-w_square[3])
        ])

        u = np.array([Torque, tau[0], tau[1], tau[2]])
        [x, y, z] = state['x']
        # if 1 - abs(x) < dist:
        ux, uy = qp_xy(state, u)
        u[1] = ux
        u[2] = uy
        lglf3h_u = LgLf3h_uhat(state, u)
        l4fh = Lf4h(state, u[0])
        print('4 order', lglf3h_u + l4fh)
        # [xdddt, ydddt] = xydddt(state, u[0])
        # print([xdddt, ydddt])
        state = quad_dyn.step_dynamics(state, u)
        # print('x', state['x'])
        # print('xd', state['xdot'])
        # print('xdd', state['xdd'])
        # print('theta', state['theta'])
        # print('omega', state['thetadot'])
        xs.append(state['x'][0])
        ys.append(state['x'][1])
        zs.append(state['x'][2])
        # print(state['x'])
        # update history for plotting
        quad_hist.update_history(state, des_theta_deg, des_vel, des_pos, dt)
    print(xs)



if __name__ == '__main__':
    main()
