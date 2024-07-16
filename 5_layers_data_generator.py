import math
import cmath
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import random
import multiprocessing
import time
from functools import partial
from multiprocessing import Pool
import tqdm

nm = 1e-9
ln2 = 0.69
mu = 800 * nm
sigma = (50 / (2 * ln2) ** 0.5) * nm
wavelength_vector = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 1000)

""" Parameters """

WITH_CHIRP = False

second = 1
fs = second * 1e-15
ps = second * 1e-12
meter = 1
nm = meter * 1e-9
um = meter * 1e-6
mm = meter * 1e-3

#######################################################################
# PART 7- Plots: plots of time, omega and lambda########################
#######################################################################

pad = 500  # 500
N_samples = 2 ** 10  # 10000
c_const = 3e8 * meter / second
n_ref = 1

dt = (2 * pad * fs) / N_samples
t_vec = np.arange(-pad * fs, pad * fs + dt, dt)

f_vec = np.fft.fftshift(np.fft.fftfreq(t_vec.shape[0], d=dt))
omega_vec = (2 * np.pi * f_vec)
lambda_vec = (2 * np.pi * c_const / omega_vec)


def non_zero(x_vec, y_vec, min_num=0.001):
    i = 0
    j = len(x_vec) - 1
    while y_vec[i] < min_num:
        i = i + 1
    while y_vec[j] < min_num:
        j = j - 1
    # k = min(i , len(x_vec) - j)
    return x_vec[i:j], y_vec[i:j]


def t_spectrum_vec(NDlist, vec):
    lis = []
    for i in range(vec.shape[0]):
        t = transmittance(NDlist, vec[i])
        lis.append(t)
    return lis


def r_spectrum_vec(NDlist, vec):
    lis = []
    for i in range(vec.shape[0]):
        r = reflectance(NDlist, vec[i])
        lis.append(r)
    return lis


def generate_pulse(NDlist, materials_dict):
    width_time = 12 * fs
    carrier_wavelength = 800 * nm
    chirp_factor = 0 / ((pad * fs) ** 2) * np.pi * 2
    print(f'width time {width_time} carrier {carrier_wavelength} chirp is {chirp_factor}')

    omega_0 = np.divide((2 * np.pi * c_const), (n_ref * carrier_wavelength))

    input_wave_amp = np.exp(- 2 * np.log(2) * np.square(t_vec / (width_time)))

    input_wave_phase = 0
    input_wave_phase = input_wave_phase + omega_0 * t_vec
    if WITH_CHIRP:
        input_wave_phase = input_wave_phase + chirp_factor * np.square(t_vec)
    input_wave = input_wave_amp * np.exp(1j * input_wave_phase)

    input_wave_spectrum = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(input_wave)))
    spectrum_intensity = np.square(np.abs(input_wave_spectrum))
    max_val = np.max(spectrum_intensity)
    spectrum_th = 0.01
    non_zero_omega = np.array(np.where(spectrum_intensity / max_val > spectrum_th))

    ''' reflected = r*input_wave_spectrum and transmitted = t*input_wave_spectrum'''

    ''' inverse fourier transform of reflected and transmitted '''
    input_wave_back = np.fft.ifftshift(np.fft.ifft(np.fft.fftshift(input_wave_spectrum)))

    plt.figure('Time Domain Analysis')
    plt.subplot(3, 2, 1)
    plt.title("Input: Amplitude as function of Time")
    plt.xlabel("Time t")
    plt.ylabel("Amplitude")
    x_in_time, y_in_time = non_zero(t_vec, np.abs(input_wave) ** 2)
    plt.plot(x_in_time, y_in_time)

    plt.subplot(3, 2, 3)
    plt.title("Input: Amplitude as function of frequency")
    plt.xlabel("Frequency \u03C9")
    plt.ylabel("Amplitude")
    x_in_omega, y_in_omega = non_zero(omega_vec, np.abs(input_wave_spectrum) ** 2)
    plt.plot(x_in_omega, y_in_omega)

    plt.subplot(3, 2, 5)
    plt.title("Input: Amplitude as function of Wavelength")
    plt.xlabel("Wavelength \u03BB")
    plt.ylabel("Amplitude")
    x_in_lambda, y_in_lambda = non_zero((lambda_vec), np.abs(input_wave_spectrum) ** 2)
    plt.plot(x_in_lambda, y_in_lambda)

    reflected_lambda = r_spectrum_vec(NDlist, lambda_vec) * input_wave_spectrum
    transmitted_lambda = t_spectrum_vec(NDlist, lambda_vec) * input_wave_spectrum
    output_wave_reflected = np.fft.ifftshift(np.fft.ifft(np.fft.fftshift(reflected_lambda)))
    output_wave_transmitted = np.fft.ifftshift(np.fft.ifft(np.fft.fftshift(transmitted_lambda)))

    plt.subplot(3, 2, 6)
    plt.title("Output: Amplitude as function of Wavelength")
    plt.xlabel("Wavelength \u03BB")
    plt.ylabel("Amplitude")
    x_out_lambda_t, y_out_lambda_t = non_zero((lambda_vec), np.abs(transmitted_lambda) ** 2)
    x_out_lambda_r, y_out_lambda_r = non_zero((lambda_vec), np.abs(reflected_lambda) ** 2)
    plt.plot(x_out_lambda_t, y_out_lambda_t)
    plt.plot(x_out_lambda_r, y_out_lambda_r)

    plt.subplot(3, 2, 4)
    plt.title("Output: Amplitude as function of frequency")
    plt.xlabel("Frequency \u03C9")
    plt.ylabel("Amplitude")
    x_out_omega_t, y_out_omega_t = non_zero((omega_vec), np.abs(transmitted_lambda) ** 2)
    x_out_omega_r, y_out_omega_r = non_zero((omega_vec), np.abs(reflected_lambda) ** 2)
    plt.plot(x_out_omega_t, y_out_omega_t)
    plt.plot(x_out_omega_r, y_out_omega_r)

    plt.subplot(3, 2, 2)
    plt.title("Output: Amplitude as function of Time")
    plt.xlabel("Time t")
    plt.ylabel("Amplitude")
    x_out_time_t, y_out_time_t = non_zero((t_vec), np.abs(output_wave_transmitted) ** 2)
    x_out_time_r, y_out_time_r = non_zero((t_vec), np.abs(output_wave_reflected) ** 2)
    plt.plot(x_out_time_t, y_out_time_t)
    plt.plot(x_out_time_r, y_out_time_r)

    # plt.subplot(3, 2, 2)
    # plt.title("Output: Amplitude as function of Time")
    # plt.xlabel("Time t")
    # plt.ylabel("Amplitude")

    ''' 4 - reflected and transmitted in lambda, 5 - reflected and transmitted in time, 6 - structure picture'''
    # plt.subplot(2, 3, 4)
    # plt.plot(t_vec, np.abs(input_wave_back) ** 2)
    # plt.subplot(2, 3, 5)
    # plt.plot(lambda_vec[non_zero_omega], reflected_spectrum)
    # plt.subplot(2, 3, 6)
    # plt.plot(lambda_vec[non_zero_omega], input_wave_spectrum[non_zero_omega], '*b')

    plt.show()

    # A = (np.abs(output_wave_transmitted)) ** 2
    # B = (np.abs(output_wave_reflected)) ** 2
    # print(sum(A) + sum(B))
    # print(sum(np.abs(input_wave) ** 2))


#######################################################################
# End PART 7- Plots: plots of time, omega and lambda###################
#######################################################################

#######################################################################
# PART 8- correlate and histogram#######################################
#######################################################################
def generate_vector(NDlist, materials_dict):
    width_time = 12 * fs
    carrier_wavelength = 800 * nm
    chirp_factor = 0 / ((pad * fs) ** 2) * np.pi * 2
    print(f'width time {width_time} carrier {carrier_wavelength} chirp is {chirp_factor}')

    omega_0 = np.divide((2 * np.pi * c_const), (n_ref * carrier_wavelength))

    input_wave_amp = np.exp(- 2 * np.log(2) * np.square(t_vec / (width_time)))

    input_wave_phase = 0
    input_wave_phase = input_wave_phase + omega_0 * t_vec
    if WITH_CHIRP:
        input_wave_phase = input_wave_phase + chirp_factor * np.square(t_vec)
    input_wave = input_wave_amp * np.exp(1j * input_wave_phase)

    input_wave_spectrum = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(input_wave)))
    spectrum_intensity = np.square(np.abs(input_wave_spectrum))
    max_val = np.max(spectrum_intensity)
    spectrum_th = 0.01

    reflected_lambda = r_spectrum_vec(NDlist, lambda_vec) * input_wave_spectrum
    transmitted_lambda = t_spectrum_vec(NDlist, lambda_vec) * input_wave_spectrum
    output_wave_reflected = np.fft.ifftshift(np.fft.ifft(np.fft.fftshift(reflected_lambda)))
    output_wave_transmitted = np.fft.ifftshift(np.fft.ifft(np.fft.fftshift(transmitted_lambda)))

    return (np.abs(output_wave_transmitted)) ** 2, (np.abs(output_wave_reflected)) ** 2


def plots1000(materials_dict, num=1000):
    vec_of_vectors_t = []
    vec_of_vectors_r = []
    index = 0
    while t_vec[index] < 10**-13:
        index = index + 1
    for i in range(num):
        NDlist = random_NDlist(materials_dict)
        a, b = generate_vector(NDlist, materials_dict)
        a = a[:index]
        b = b[:index]
        vec_of_vectors_t.append(a)
        vec_of_vectors_r.append(b)
    mat_t = np.abs(np.corrcoef(vec_of_vectors_t))
    mat_r = np.abs(np.corrcoef(vec_of_vectors_r))
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.title("T: correlate")
    plt.imshow(mat_t, cmap='gray')
    plt.subplot(2, 2, 2)
    plt.title("T: histogram, mean: " + str(np.round(np.average(np.average(mat_t)), 2)))
    a = mat_to_vec(mat_t)
    plt.hist(a)
    plt.subplot(2, 2, 3)
    plt.title("R: correlate")
    plt.imshow(mat_r, cmap='gray')
    plt.subplot(2, 2, 4)
    plt.title("R: histogram, mean: " + str(np.round(np.average(np.average(mat_r)), 2)))
    b = mat_to_vec(mat_r)
    plt.hist(b)
    plt.show()
    print('t')
    print(np.average(np.average(mat_t)))
    print('r')
    print(np.average(np.average(mat_r)))

    outfile = 'correls'
    np.savez(outfile, T_histogram=a, R_histogram=b, T_matrix=mat_t, R_matrix=mat_r)
    plt.show()

def generate_vector_par(NDlist, materials_dict):
    T, R = generate_vector(NDlist, materials_dict)
    return [T, R]

def examples(materials_dict, num = 100000):
    # with open('data_20layers_100.npy', 'rb') as f:
    #     vec_structure = np.load(f)
    #     vec_t = np.load(f)
    #     vec_r = np.load(f)
    # structure = list(vec_structure)
    # vectorsT = list(vec_t)
    # vectorsR = list(vec_r)
    structure = []
    vectorsT = []
    vectorsR = []
    # vectorsT_old = list.copy(vectorsT)
    # vectorsR_old = list.copy(vectorsR)
    # tic = time.time()
    NDlists = []
    for i in range(num):
        NDlists.append(random_NDlist(materials_dict))
    structure = structure + NDlists
    # for NDlist in NDlists:
    #     T, R = generate_vector(NDlist, materials_dict)
    #     structure.append(NDlist)
    #     vectorsT.append(T)
    #     vectorsR.append(R)

    # toc = time.time()
    # t1 = toc-tic
    # tic = time.time()

    with Pool() as pool:
        res = pool.map(partial(generate_vector_par, materials_dict = materials_dict), NDlists)
    vectorsT_new = [i[0] for i in res]
    vectorsR_new = [i[1] for i in res]
    vectorsT = vectorsT + vectorsT_new
    vectorsR = vectorsR + vectorsR_new
    # assert len(vectorsT) == len(vectorsT_new)
    # assert all([all(vectorsT[i] == vectorsT_new[i]) for i in range(len(vectorsT))])
    # toc = time.time()
    # print(t1)
    # print(toc - tic)
    # print('a')
    structures2 = []
    for NDlist in structure:
        structures2 = structures2 + [NDlist]
    structures2 = np.array(structures2)
    vectorsT = np.array(vectorsT)
    vectorsR = np.array(vectorsR)
    print(structures2.shape, vectorsT.shape, vectorsR.shape)
    with open('data_5layers_100000.npy', 'wb') as f:
        np.save(f, structures2)
        np.save(f, vectorsT)
        np.save(f, vectorsR)


def coralation_of_data(materials_dict):
    with open('data.npy', 'rb') as f:
        vec_structure = np.load(f)
        vec_t = np.load(f)
        vec_r = np.load(f)
    mat_t = np.corrcoef(vec_t)
    mat_r = np.corrcoef(vec_r)
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.title("T: correlate")
    plt.imshow(mat_t, cmap='gray')
    plt.subplot(2, 2, 2)
    plt.title("T: histogram")
    a = mat_to_vec(mat_t)
    plt.hist(a)
    plt.subplot(2, 2, 3)
    plt.title("R: correlate")
    plt.imshow(mat_r, cmap='gray')
    plt.subplot(2, 2, 4)
    plt.title("R: histogram")
    b = mat_to_vec(mat_r)
    plt.hist(b)
    plt.show()


def mat_to_vec(mat):
    vec = []
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            vec.append(mat[i][j])
    return vec


#######################################################################
# END PART 8- correlate and histogram###################################
#######################################################################


###################################################################
# PART 1- Create the M matrix ######################################
###################################################################

# This function gets the name of the excel file of the materials, and returns a dictionary of the materials and thier data
def materials(file_name):
    try:
        df = pd.read_csv(file_name)
        material = df['material']
        B1 = list(df['B1'])
        B2 = list(df['B2'])
        B3 = list(df['B3'])
        C1 = list(df['C1'])
        C2 = list(df['C2'])
        C3 = list(df['C3'])
        refractive_index = list(df['refractive_index'])
        d = {}
        for i in range(len(material)):
            d[material[i]] = (float(B1[i]), float(B2[i]), float(B3[i]), float(C1[i]), float(C2[i]), float(C3[i]),
                              float(refractive_index[i]))
        return d
    except IOError:
        print("An IO error occurred")


# This function gets the data of a single material, and return it's n
def n(B1, B2, B3, C1, C2, C3, refractive_index, k=0, wave_length=1550 * nm):
    m1 = (B1 * pow(1.55, 2)) / (pow(1.55, 2) - C1)
    m2 = (B2 * pow(1.55, 2)) / (pow(1.55, 2) - C2)
    m3 = (B3 * pow(1.55, 2)) / (pow(1.55, 2) - C3)
    if refractive_index == 0:
        N = pow(1 + m1 + m2 + m3, 0.5)
    else:
        N = refractive_index
    return N + k


# This function asks the user which materials he wants and their whidths, and returns list of tupels of n and width of each material
def inputs(materials_dict):
    lis = []
    print("The list of valid materials is:")
    print(list(materials_dict.keys()))
    first = True
    d = None
    while d != 0:
        if first:
            material = input("Enter the first material, choose a name from the list: ").lower()
        else:
            material = input("Enter material name from the list: ").lower()
        if material not in materials_dict:
            raise ValueError("The material is not in the excel file")
        if first:
            d = -1
            first = False
        else:
            d = float(input("Enter width, for the last material enter 0: "))
        m = materials_dict[material]
        N = n(m[0], m[1], m[2], m[3], m[4], m[5], m[6])
        lis += [(N, d)]
    return lis


# This function gets the wave vector and the width d of a single mediume, and returns the P matrix of this mediume
def P_matrix(n, d, wave_length):
    phase = 2 * (math.pi / wave_length) * d * n
    P11 = cmath.exp(-1j * phase)
    P22 = cmath.exp(1j * phase)
    P = np.array([[P11, 0], [0, P22]])
    return P


# This function gets n1 and n2, and returns the T matrix
def T_matrix(n1, n2):
    n1 = n1.real
    n2 = n2.real
    T11 = (n1 + n2) / (2 * n1)
    T12 = (n1 - n2) / (2 * n1)
    T21 = (n1 - n2) / (2 * n1)
    T22 = (n1 + n2) / (2 * n1)
    T = np.array([[T11, T12], [T21, T22]])
    return T


# This function gets the list of (n,d) of each material and the k of the first medium, and returns a list of the P matrices for each medium
def P_list(NDlist, wave_length):
    lis = []
    for i in range(len(NDlist) - 2):
        n = NDlist[i + 1][0]
        d = NDlist[i + 1][1]
        lis.append(P_matrix(n, d, wave_length))
    return lis


# This function get the list of (n,d) of each material, and return a list of the T matrices between every 2 materials
def T_list(NDlist):
    lis = []
    for i in range(len(NDlist) - 1):
        lis.append(T_matrix(NDlist[i][0], NDlist[i + 1][0]))
    return lis


# This function gets the input wave data and the list of (n,d) of each material, and returns the M matrix
def M_matrix(NDlist, wave_length):
    Tlist = T_list(NDlist)
    Plist = P_list(NDlist, wave_length)
    M = np.array([[1, 0], [0, 1]])
    for i in range(len(Plist)):
        m = np.dot(Tlist[i], Plist[i])
        M = np.dot(M, m)
    M = np.dot(M, Tlist[-1])
    return M


###################################################################
# END PART 1- Create the M matrix ##################################
###################################################################


###################################################################
# PART 2- Calculate the Reflectance and Transmittance: T, R, t, r  #
###################################################################

# This function returns the Reflectance R
def Reflectance(NDlist, wave_length):
    M = M_matrix(NDlist, wave_length)
    r = M[1][0] / M[0][0]
    return pow(abs(r), 2)


# This function returns the Transmittance T
def Transmittance(NDlist, wave_length):
    M = M_matrix(NDlist, wave_length)
    t = 1 / M[0][0]
    n0 = NDlist[0][0].real
    nN = NDlist[-1][0].real
    return pow(abs(t), 2) * (nN / n0)


# This function returns the transmittance t
def reflectance(NDlist, wave_length):
    M = M_matrix(NDlist, wave_length)
    r = M[1][0] / M[0][0]
    return r


# This function returns the transmittance t
def transmittance(NDlist, wave_length):
    M = M_matrix(NDlist, wave_length)
    t = 1 / M[0][0]
    return t


#######################################################################
# End PART 2- Calculate the Reflectance and Transmittance: T, R, t, r  #
#######################################################################


#######################################################################
# PART 3- Random input NDlist ##########################################
#######################################################################

# This function create a random NDlist
def random_NDlist(materials_dict):
    lis = [[1, 0, 'air']]
    keys = list(materials_dict.keys())[:-1]
    last_material = ''
    for i in range(5):
        material = keys[random.randint(0, len(keys) - 1)]
        while material == last_material:
            material = keys[random.randint(0, len(keys) - 1)]
        last_material = material
        B1 = materials_dict[material][0]
        B2 = materials_dict[material][1]
        B3 = materials_dict[material][2]
        C1 = materials_dict[material][3]
        C2 = materials_dict[material][4]
        C3 = materials_dict[material][5]
        refractive_index = materials_dict[material][6]
        N = n(B1, B2, B3, C1, C2, C3, refractive_index)
        D = random.randint(50, 10000) * nm
        lis.append([N, D, material])
    lis.append([1, 0, 'air'])
    return lis


#######################################################################
# END PART 3- Random input NDlist ######################################
#######################################################################

#######################################################################
# PART 4- Plots of the input wave, Transmitance and Reflectance ########
#######################################################################

# This function returns array (plot) of the point of Transmittance for array of wave lengths
def T_plot(NDlist, min=wavelength_vector[0], max=wavelength_vector[-1]):
    t = []
    wave_length_vector = np.linspace(min, max, 1000)
    for i in range(wave_length_vector.shape[0]):
        T = Transmittance(NDlist, wave_length_vector[i])
        t.append(T)
    return t


# This function returns array (plot) of the point of Reflectance for array of wave lengths
def R_plot(NDlist, min=wavelength_vector[0], max=wavelength_vector[-1]):
    r = []
    wave_length_vector = np.linspace(min, max, 1000)
    for i in range(wave_length_vector.shape[0]):
        R = Reflectance(NDlist, wave_length_vector[i])
        r.append(R)
    return r


# This function returns array (plot) of the point of transmittance t for array of wave lengths
def t_spectrum(NDlist):
    lis = []
    for i in range(wavelength_vector.shape[0]):
        t = transmittance(NDlist, wavelength_vector[i])
        lis.append(t)
    return lis


# This function returns array (plot) of the point of reflectance r for array of wave lengths
def r_spectrum(NDlist):
    lis = []
    for i in range(wavelength_vector.shape[0]):
        r = reflectance(NDlist, wavelength_vector[i])
        lis.append(r)
    return lis


# This function returns a vector of gaussian of the input wave length around mu
def wave_spectrum(mu, sigma):
    # vector = np.exp(-np.square(wavelength_vector - mu) / (2 * np.square(sigma)))
    vector = np.exp(-(wavelength_vector - mu) ** 2 / (2 * sigma ** 2))
    return vector


#######################################################################
# END PART 4- Plots of the input wave, Transmitance and Reflectance ####
#######################################################################

#######################################################################
# PART 5- Structure plot and color bar #################################
#######################################################################

# This function create dict of colors: the keys are the names of the material and the value is a number between 0-255
def colors_dict(materials_dict):
    keys = list(materials_dict.keys())[:-1]
    dict_colors = {}
    for i in range(len(keys)):
        dict_colors[keys[i]] = int((255 / (len(keys) - 1)) * i)
    return dict_colors


def colors_plot(materials_dict):
    lis = []
    d = colors_dict(materials_dict)
    for key in d:
        lis.append(np.ones((1, 5)) * d[key])
    return lis


def pow9(x):
    return int(x / nm)


def structure_plot(NDlist, materials_dict):
    d_list = []
    for i in NDlist:
        d_list.append(i[1])
    d_list = list(map(pow9, d_list))[1:-1]
    NDlist = NDlist[1:-1]
    plot = np.zeros((int(sum(d_list) / 3), sum(d_list)))
    dict_colors = colors_dict(materials_dict)
    place = 0
    for i in range(len(d_list)):
        d = d_list[i]
        for j in range(place, place + d):
            plot[:, j] = dict_colors[NDlist[i][2]]
        place += d
    return plot


def structure_plot_100nm(NDlist, materials_dict):
    d_list = []
    for i in NDlist:
        d_list.append(i[1])
    d_list = list(map(pow9, d_list))[1:-1]
    NDlist = NDlist[1:-1]
    length = 0
    for i in range(len(d_list)):
        length += int((d_list[i] / 50))
    plot = np.zeros((int(length / 3), length))
    dict_colors = colors_dict(materials_dict)
    place = 0
    for i in range(len(d_list)):
        d = int((d_list[i] / 50))
        for j in range(place, place + d):
            plot[:, j] = dict_colors[NDlist[i][2]]
        place += d
    return plot


#######################################################################
# END PART 5- Structure plot and color bar #############################
#######################################################################

#######################################################################
# PART 6- Plots: input wave, T and R, output waves, structure ##########
#######################################################################

def pulse_plots(materials_dict, NDlist):
    wave_in = wave_spectrum(mu=mu, sigma=sigma)
    transmit_vector = t_spectrum(NDlist)
    reflect_vector = r_spectrum(NDlist)
    transmitted_wave = wave_in * transmit_vector
    reflected_wave = wave_in * reflect_vector

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(wavelength_vector, np.abs(wave_in))
    axs[0, 0].set_title("Input spectrum")
    axs[0, 0].set(ylabel="Amplitude")

    axs[1, 0].plot(wavelength_vector, T_plot(NDlist))
    axs[1, 0].plot(wavelength_vector, R_plot(NDlist))
    axs[1, 0].set(xlabel='Wavelength', ylabel='Blue is T, Orange is R')
    axs[1, 0].set_title("T and R as function of wavelength")

    axs[0, 1].plot(wavelength_vector, np.abs(transmitted_wave))
    axs[0, 1].plot(wavelength_vector, np.abs(reflected_wave))
    axs[0, 1].set_title("Output spectrum- Blue is transmitted, Orange is reflected")
    axs[0, 1].set(ylabel="Amplitude")

    structure = structure_plot_100nm(NDlist, materials_dict)
    pcm = axs[1, 1].pcolormesh(structure, cmap=cm.get_cmap('viridis', len(materials_dict) - 1))
    fig.colorbar(pcm, ax=axs[1, 1])
    axs[1, 1].set_title("The structure of layers, number of layers is: " + str(len(NDlist) - 2))
    axs[1, 1].set(xlabel="The width of each layer in 100nm, max width is: 10000nm")

    plt.show()


#######################################################################
# END PART 6- Plots: input wave, T and R, output waves, structure ######
#######################################################################

if __name__ == '__main__':
    materials_dict = materials("materials.csv")
    NDlist = random_NDlist(materials_dict)
    print(NDlist)

    #pulse_plots(materials_dict, NDlist)

    #generate_pulse(NDlist, materials_dict)

    #plots1000(materials_dict)

    examples(materials_dict)

    #coralation_of_data(materials_dict)
