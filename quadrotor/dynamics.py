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

maxrpm = 10000
maxthrust = k*np.sum(np.array([maxrpm**2] * 4))
param_dict = {"g": g, "m": m, "L": L, "k": k, "b": b, "I": I,
              "kd": kd, "dt": dt, "maxRPM": maxrpm, "maxthrust": maxthrust}


def init_state():
    """Initialize state dictionary. """
    state = {"x": np.array([5, 5, 0]),
             "xdot": np.zeros(3,),
             "xdd": np.zeros(3,),
             "theta": np.radians(np.array([0, 0, 0])),  # ! hardcoded
             "thetadot": np.radians(np.array([0, 0, 0]))  # ! hardcoded
             }
    return state


def qp_z(state, u_hat):
    z = state['x'][2]
    z_dot = state['xdot'][2]
    theta = state['theta']
    matrix_rotation = get_rot_matrix(theta)
    R_33 = matrix_rotation[2][2]
    P = matrix([1])
    q = matrix([0])
    P = matrix(P, (1, 1), 'd')
    q = matrix(q, (1, 1), 'd')
    h = matrix([2 * z * R_33 / m * u_hat - 2 * z * g - 2 * z_dot ** 2 + k_1 * (- 2 * z * z_dot) + k_2 * (1 - z ** 2)])
    G = matrix([2 * z * R_33 / m])
    sol = solvers.qp(P,q,G,h)
    sol['x']
    u_bar = sol['x']
    u = u_bar + u_hat

    return u[0]


class QuadDynamics:
    def __init__(self):
        self.param_dict = param_dict

    def step_dynamics(self, state, u):
        """Step dynamics given current state and input. Updates state dict.
        
        Parameters
        ----------
        state : dict 
            contains current x, xdot, theta, thetadot

        u : (4, ) np.ndarray
            control input - (angular velocity)^squared of motors (rad^2/s^2)

        Updates
        -------
        state : dict 
            updates with next x, xdot, xdd, theta, thetadot  
        """
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
        """Compute total thrust (in body frame) given control input and thrust coefficient. Used in calc_acc().
        Clips if above maximum rpm (10000).

        thrust = k * sum(u)
        
        Parameters
        ----------
        u : (4, ) np.ndarray
            control input - (angular velocity)^squared of motors (rad^2/s^2)

        k : float
            thrust coefficient

        Returns
        -------
        T : (3, ) np.ndarray
            thrust in body frame
        """

        u = np.clip(u, 0, self.param_dict["maxRPM"]**2)
        T = np.array([0, 0, k*np.sum(u)])
        # print("u", u)
        # print("T", T)

        return T

    def calc_torque(self, u, L, b, k):
        """Compute torque (body-frame), given control input, and coefficients. Used in calc_ang_acc()
        
        Parameters
        ----------
        u : (4, ) np.ndarray
            control input - (angular velocity)^squared of motors (rad^2/s^2)
        L : float
            distance from center of quadcopter to any propellers, to find torque (m).
        
        b : float # TODO: description
        
        k : float
            thrust coefficient

        Returns
        -------
        tau : (3,) np.ndarray
            torque in body frame (Nm)

        """
        tau = np.array([
            L * k * (u[0]-u[2]),
            L * k * (u[1]-u[3]),
            b * (u[0]-u[1] + u[2]-u[3])
        ])

        return tau

    def calc_acc(self, u, theta, xdot, m, g, k, kd):
        """Computes linear acceleration (in inertial frame) given control input, gravity, thrust and drag.
        a = g + T_b+Fd/m

        Parameters
        ----------
        u : (4, ) np.ndarray
            control input - (angular velocity)^squared of motors (rad^2/s^2)
        theta : (3, ) np.ndarray 
            rpy angle in body frame (radian) 
        xdot : (3, ) np.ndarray
            linear velocity in body frame (m/s), for drag calc 
        m : float
            mass of quadrotor (kg)
        g : float
            gravitational acceleration (m/s^2)
        k : float
            thrust coefficient
        kd : float
            drag coefficient

        Returns
        -------
        a : (3, ) np.ndarray 
            linear acceleration in inertial frame (m/s^2)
        """
        gravity = np.array([0, 0, g])
        R = get_rot_matrix(theta)
        thrust = np.array([0, 0, u[0]])
        # thrust = self.compute_thrust(u, k)
        T = np.dot(R, thrust)
        Fd = -kd * xdot
        a = gravity - 1/m * T
        return a

    def calc_ang_acc(self, u, omega, I, L, b, k):
        """Computes angular acceleration (in body frame) given control input, angular velocity vector, inertial matrix.
        
        omegaddot = inv(I) * (torque - w x (Iw))

        Parameters
        ----------
        u : (4, ) np.ndarray
            control input - (angular velocity)^squared of motors (rad^2/s^2)
        omega : (3, ) np.ndarray 
            angular velcoity vector in body frame
        I : (3, 3) np.ndarray 
            inertia matrix
        L : float
            distance from center of quadcopter to any propellers, to find torque (m).
        b : float # TODO: description
        k : float
            thrust coefficient


        Returns
        -------
        omegaddot : (3, ) np.ndarray
            rotational acceleration in body frame #TODO: units
        """
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
        """Compute angle rate from angular velocity vector and euler angle.

        Uses Tait Bryan's z-y-x/yaw-pitch-roll.

        Parameters
        ----------

        omega: (3, ) np.ndarray
            angular velocity vector

        theta: (3, ) np.ndarray
            euler angles in body frame (roll, pitch, yaw)

        Returns
        ---------
        thetadot: (3, ) np.ndarray
            time derivative of euler angles (roll rate, pitch rate, yaw rate)
        """
        # mult_matrix = np.array(
        #     [
        #         [1, 0, -np.sin(theta[1])],
        #         [0, np.cos(theta[0]), np.cos(theta[1])*np.sin(theta[0])],
        #         [0, -np.sin(theta[0]), np.cos(theta[1])*np.cos(theta[0])]
        #     ], dtype='float')
        #


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
        """Compute angular velocity vector from euler angle and associated rates.
        
        Uses Tait Bryan's z-y-x/yaw-pitch-roll. 

        Parameters
        ----------
        
        thetadot: (3, ) np.ndarray
            time derivative of euler angles (roll rate, pitch rate, yaw rate)

        theta: (3, ) np.ndarray
            euler angles in body frame (roll, pitch, yaw)

        Returns
        ---------
        w: (3, ) np.ndarray
            angular velocity vector (in body frame)
        
        """
        phi = theta[0]
        the = theta[1]
        psi = theta[2]

        # mult_matrix = np.array(
        #     [
        #         [1, 0, -np.sin(pitch)],
        #         [0, np.cos(roll), np.cos(pitch)*np.sin(roll)],
        #         [0, -np.sin(roll), np.cos(pitch)*np.cos(roll)]
        #     ])
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
        des_pos = np.array([3, -5, 2 * np.sin(i * dt)])
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
        z = state['x'][2]
        if 1 - abs(z) < dist:
            F_cbf = qp_z(state, Torque)
            u[0] = F_cbf
        state = quad_dyn.step_dynamics(state, u)
        xs.append(state['x'][0])
        ys.append(state['x'][1])
        zs.append(state['x'][2])
        print(state['x'])
        # update history for plotting
        quad_hist.update_history(state, des_theta_deg, des_vel, des_pos, dt)


##  calculate norminal trajectory
    zs_nom = list()
    zs_ref = list()
    state = init_state()
    for j in range(sim_iter):
        des_pos = np.array([3, -5, 2 * np.sin(j * dt)])
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
            L * k * (w_square[0] - w_square[2]),
            L * k * (w_square[1] - w_square[3]),
            b * (w_square[0] - w_square[1] + w_square[2] - w_square[3])
        ])

        u = np.array([Torque, tau[0], tau[1], tau[2]])
        state = quad_dyn.step_dynamics(state, u)

        zs_nom.append(state['x'][2])
        zs_ref.append(2 * np.sin(j * dt))
        print(state['x'])
        # update history for plotting
        quad_hist.update_history(state, des_theta_deg, des_vel, des_pos, dt)


    ax1.plot(xs)
    ax1.set_ylabel('x')
    ax2.plot(ys)
    ax2.set_ylabel('y')
    ax3.plot(zs)
    ax3.set_ylabel('z')
    ax3.plot(zs_nom, color='r')
    # ax3.plot(zs_ref, color='g')
    plt.tight_layout()
    plt.grid()
    plt.show()

if __name__ == '__main__':
    main()
