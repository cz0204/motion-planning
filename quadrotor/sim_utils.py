import numpy as np

# def get_rot_matrix(angles):
#     [phi, theta, psi] = angles
#     cphi = np.cos(phi)
#     sphi = np.sin(phi)
#     cthe = np.cos(theta)
#     sthe = np.sin(theta)
#     cpsi = np.cos(psi)
#     spsi = np.sin(psi)
#
#     rot_mat = np.array([[cthe * cpsi, sphi * sthe * cpsi - cphi * spsi, cphi * sthe * cpsi + sphi * spsi],
#                         [cthe * spsi, sphi * sthe * spsi + cphi *
#                             cpsi, cphi * sthe * spsi - sphi * cpsi],
#                         [-sthe,       cthe * sphi,                      cthe * cphi]])
#     return rot_mat


def get_rot_matrix(angles):
    [phi, the, psi] = angles
    cphi = np.cos(phi)
    sphi = np.sin(phi)
    cthe = np.cos(the)
    sthe = np.sin(the)
    cpsi = np.cos(psi)
    spsi = np.sin(psi)

    rot_mat = np.array([[cthe * cpsi - sphi * sthe * spsi, -cphi * spsi,  cpsi * sthe + cthe * sphi * spsi],
                        [cthe * spsi + cpsi * sphi * sthe, cphi * cpsi, spsi * sthe - cpsi * cthe * sphi],
                        [-cphi * sthe, sphi,  cphi * cthe]])
    return rot_mat
