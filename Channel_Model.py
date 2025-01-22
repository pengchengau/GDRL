import numpy as np
from scipy.constants import speed_of_light as c
import matlab.engine

def ChannelModel(uf, du):
    fkl = 2.9e9 # The carrier frequency for LEO
    fkn = 15e9 # The carrier frequency for HAPS
    lamdal = c/fkl # The wavelength for LEO
    lamdan = c/fkn # The wavelength for HAPS
    Gul = 10 # The antenna gain which is the multiply of user gain and LEO n gain
    Gun = 10 # The antenna gain which is the multiply of user gain and HAPS n gain
    Mn_x = 6 # The x-axis antenna number for HAPS
    Mn_y = 6 # The y-axis antenna number for HAPS
    Ml_x = 12 # The x-axis antenna number for LEO
    Ml_y = 12 # The y-axis antenna number for LEO
    dn_x = lamdan # The antenna spacing in x-axis for HAPS
    dn_y = lamdan # The antenna spacing in y-axis for HAPS
    dl_x = lamdal # The antenna spacing in x-axis for LEO
    dl_y = lamdal # The antenna spacing in y-axis for LEO
    vertheta = np.pi/6 # The angle in paper MIMO satellite vertheta
    theta_x = np.pi/4 # The angle in paper MIMO satellite theta
    theta_y = np.arcsin(np.sin(theta_x)*np.sin(vertheta)/np.cos(vertheta))
    eng = matlab.engine.start_matlab()
    gukl = eng.groundtospace(fkl, vertheta) #The channel for ground to space except array response vector
    gukn = eng.groundtospace(fkn, vertheta) #The channel for ground to air except array response vector
    if uf == 1 or uf == 2 or uf == 5: # user to LEO with 2 represent throughput oriented and 5 represent delay oriented
        n_u_l = a_mimo(np.sin(theta_y)*np.cos(theta_x), Ml_x, dl_x, fkl) * a_mimo(np.cos(theta_y), Ml_y, dl_y, fkl)
        hu =  np.sqrt(Gul * (c/(4*np.pi*(du)*(fkl)))) * gukl * n_u_l
    else: # user to HAPS with 3 represent throughput oriented and 6 represent delay oriented
        n_u_n = a_mimo(np.sin(theta_y)*np.cos(theta_x), Mn_x, dn_x, fkn) * a_mimo(np.cos(theta_y), Mn_y, dn_y, fkn)
        hu = np.sqrt(Gun * (c/(4*np.pi*(du)*(fkn)))) * gukn * n_u_n
    return hu

def a_mimo(x, y, d_star, fk):
    # x is the cos and sin of the angle theta, y is the number of antennas
    # in each direction, d_star is the antenna spacing in each direction
    n_u = (1/np.sqrt(y)) * np.exp(-1j*2*np.pi*c*d_star/(fk*y) * np.outer(np.arange(y), x))
    return n_u


def fspl(d, lambda_):
    fspl_db = 20 * np.log10(d) + 20 * np.log10(4*np.pi/lambda_)
    return fspl_db