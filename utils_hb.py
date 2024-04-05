# imports
import numpy as np
import scipy
import scipy.linalg

from scipy.integrate import ode


# Driven harmonic oscillator
def ho_complex(time_, state_, f0_, k0_, kext_, F1_, F2_, fd1_, fd2_):
    # Angular frequencies
    w0 = 2 * np.pi * f0_
    wd1_ = 2 * np.pi * fd1_
    wd2_ = 2 * np.pi * fd2_

    # drives and drive derivatives
    drives_ = F1_ * np.exp(-1j * wd1_ * time_) + F2_ * np.exp(-1j * wd2_ * time_)

    # ho term
    ho_term = -1.0j * w0 * state_
    # linear damping term
    damp_term = -k0_ / 2 * state_

    return ho_term + damp_term - np.sqrt(kext_) * drives_


def ho_real(time_, state_real, f0_, k0_, kext_, F1_, F2_, fd1_, fd2_):
    state_complex = state_real[::2] + 1j * state_real[1::2]
    ret_complex = ho_complex(time_, state_complex, f0_, k0_, kext_, F1_, F2_, fd1_, fd2_)
    tmp = np.zeros(len(ret_complex) * 2)
    tmp[::2] = np.real(ret_complex)
    tmp[1::2] = np.imag(ret_complex)

    return tmp


def ho_simulation(time_, f0_, k0_, kext_, F1_, F2_, fd1_, fd2_, y0_, N_, dt_):
    # Integrator
    o = ode(ho_real).set_integrator('lsoda', atol=1e-11, rtol=1e-11)
    o.set_f_params(f0_, k0_, kext_, F1_, F2_, fd1_, fd2_)
    o.set_initial_value(y0_)

    # Time-domain output field
    y_all = np.zeros((len(time_), len(y0_)))
    for i, t_val in enumerate(time_):
        o.integrate(t_val)
        y_all[i] = o.y

    # Merge the results onto the complex plane
    field_all = y_all[:, 0] + 1.0j * y_all[:, 1]

    # We save one oscillation once we reached the steady state
    field = field_all[-N_ - 1:-1]
    t_ = time_[-N_ - 1:-1]

    # Fourier domain solution
    Field = np.fft.fft(field) / len(field)
    frequencies = np.fft.fftfreq(len(t_), d=dt_)

    return Field, field, frequencies, t_


# Kerr (Duffing) oscillator
def kerr_complex(time_, state_, f0_, k0_, k_, F1_, F2_, fd1_, fd2_, kext_, psi_,
                 phase_, gain_, att_, in_out):
    # Angular frequencies
    w0_ = 2 * np.pi * f0_
    wd1_ = 2 * np.pi * fd1_
    wd2_ = 2 * np.pi * fd2_

    # phase factor
    phase_factor = np.sqrt(gain_ / att_) * np.exp(1.0j * phase_)

    # normalized drives and drive derivatives
    drives_ = phase_factor * (F1_ * np.exp(-1j * wd1_ * time_) +
                              F2_ * np.exp(-1j * (wd2_ * time_ + psi_)))

    if not in_out:
        # ho term
        ho_term = -1.0j * w0_ * state_
        # linear damping term
        damp_term = -k0_ / 2 * state_
        # kerr term
        nonlin_damp_term = 1.0j * k_ / 2 * np.abs(state_) ** 2 * state_

        return ho_term + damp_term + nonlin_damp_term - np.sqrt(kext_) * drives_

    else:
        # normalised drive deerivatives
        ddrives_ = -1.0j * phase_factor * (F1_ * wd1_ * np.exp(-1j * wd1_ * time_) +
                                           F2_ * wd2_ * np.exp(-1j * (wd2_ * time_ + psi_)))

        # ho term
        ho_term = -1.0j * w0_ * (state_ - drives_)
        # linear damping term
        damp_term = -k0_ / 2 * (state_ - drives_)
        # duffing damping term
        nonlin_damp_term = 1.0j * k_ / (2 * kext_ * gain_) * np.abs(state_ - drives_) ** 2 * (state_ - drives_)

        return ho_term + damp_term + nonlin_damp_term - kext_ * drives_ + ddrives_


def kerr_real(time_, state_real, f0_, k0_, k_, F1_, F2_, fd1_, fd2_, kext_, psi_,
              phase_, gain_, att_, in_out):
    state_complex = state_real[::2] + 1j * state_real[1::2]
    ret_complex = kerr_complex(time_, state_complex, f0_, k0_, k_, F1_, F2_, fd1_, fd2_, kext_, psi_,
                               phase_, gain_, att_, in_out)
    tmp = np.zeros(len(ret_complex) * 2)
    tmp[::2] = np.real(ret_complex)
    tmp[1::2] = np.imag(ret_complex)

    return tmp


def kerr_simulation(time_, y0_, N_, dt_,
                    f0_, k0_, k_, F1_, F2_, fd1_, fd2_, kext_, psi_,
                    phase_=0, gain_=1, att_=1, in_out=False):
    # Integrator
    o = ode(kerr_real).set_integrator('lsoda', atol=1e-11, rtol=1e-11)
    o.set_f_params(f0_, k0_, k_, F1_, F2_, fd1_, fd2_, kext_, psi_,
                   phase_, gain_, att_, in_out)
    o.set_initial_value(y0_)

    # Time-domain output field
    y_all = np.zeros((len(time_), len(y0_)))
    for i, t_val in enumerate(time_):
        o.integrate(t_val)
        y_all[i] = o.y

    # Merge the results onto the complex plane
    field_all = y_all[:, 0] + 1.0j * y_all[:, 1]

    # We save one oscillation once we reached the steady state
    field = field_all[-N_ - 1:-1]
    t_ = time_[-N_ - 1:-1]

    # Fourier domain solution
    Field = np.fft.fft(field) / len(field)
    frequencies = np.fft.fftfreq(len(t_), d=dt_)

    return Field, field, frequencies, t_


def kerr_reconstruction(Field, Field_in, field, field_in, freqs, indices, in_out=False):
    # Angular frequency
    w = 2 * np.pi * freqs

    # Create H-matrix
    if not in_out:
        col1 = (1.0j * w * Field)[indices]
        col2 = (1.0j * Field)[indices]
        col3 = Field[indices]
        col4 = -1.0j * np.fft.fft(np.abs(field) ** 2 * field)[indices] / len(field)

    else:
        col1 = (1.0j * w * (Field - Field_in))[indices]
        col2 = (1.0j * (Field - Field_in))[indices]
        col3 = (Field - Field_in)[indices]
        col4 = -1.0j * np.fft.fft(np.abs(field - field_in) ** 2 * (field - field_in))[indices] / len(field)

    # Merge columns
    H = np.vstack((col1, col2, col3, col4))

    # Making the matrix real instead of complex
    Hcos = np.real(H)
    Hsin = np.imag(H)
    H = np.hstack((Hcos, Hsin))

    # Normalize H for a more stable inversion
    Nm = np.diag(1. / np.max(np.abs(H), axis=1))
    H_norm = np.dot(Nm, H)  # normalized H-matrix

    # The drive vector, Q (from the Yasuda paper)
    Qcos = np.real(-Field_in)[indices]
    Qsin = np.imag(-Field_in)[indices]
    Qmat = np.hstack((Qcos, Qsin))

    # Solve system Q = H*p
    H_norm_inv = scipy.linalg.pinv(H_norm)
    p_norm = np.dot(Qmat, H_norm_inv)

    # Re-normalize p-values
    # Note: we have actually solved Q = H * Nm * Ninv * p
    # Thus we obtained Ninv*p and multiply by Nm to obtain p
    p = np.dot(Nm, p_norm)  # re-normalize parameter values

    # Forward calculation to check result, should be almost everything zero vector
    Q_fit = np.dot(p, H)

    # Scale parameters by drive force assuming known resonant frequency
    param_rec = p

    # Parameters reconstructed
    if not in_out:
        kext_rec = 1 / param_rec[0]**2
        f0_rec = param_rec[1] / (2 * np.pi) / param_rec[0]
        k0_rec = 2 * param_rec[2] / param_rec[0]
        k_rec = 2 * param_rec[3] / param_rec[0]

    else:
        kext_rec = 1 / param_rec[0]
        f0_rec = param_rec[1] / (2 * np.pi) / param_rec[0]
        k0_rec = 2 * param_rec[2] / param_rec[0]
        k_rec = 2 * param_rec[3] / param_rec[0]**2

    return kext_rec, f0_rec, k0_rec, k_rec, Q_fit


# TLS damping driven oscillator
def tls_complex(time_, state_, f0_, k0_, ktls_, ac_, beta_, F1_, F2_, fd1_, fd2_, kext_, psi_,
                phase_, gain_, att_, in_out):
    # Angular frequencies
    w0_ = 2 * np.pi * f0_
    wd1_ = 2 * np.pi * fd1_
    wd2_ = 2 * np.pi * fd2_

    # phase factor
    phase_factor = np.sqrt(gain_ / att_) * np.exp(1.0j * phase_)

    # normalized drives and drive derivatives
    drives_ = phase_factor * (F1_ * np.exp(-1j * wd1_ * time_) +
                              F2_ * np.exp(-1j * (wd2_ * time_ + psi_)))

    if not in_out:
        # ho term
        ho_term = -1.0j * w0_ * state_
        # linear damping term
        damp_term = -k0_ / 2 * state_
        # duffing damping term
        nonlin_damp_term = -ktls_ / 2 / (1 + (np.abs(state_) / ac_) ** 2) ** beta_ * state_

        return ho_term + damp_term + nonlin_damp_term - np.sqrt(kext_) * drives_

    else:
        ddrives_ = -1.0j * phase_factor * (F1_ * wd1_ * np.exp(-1j * wd1_ * time_) +
                                           F2_ * wd2_ * np.exp(-1j * (wd2_ * time_ + psi_)))

        # ho term
        ho_term = -1.0j * w0_ * (state_ - drives_)
        # linear damping term
        damp_term = -k0_ / 2 * (state_ - drives_)
        # non-linear damping term
        nonlin_damp_term = -ktls_ / 2 / (
                    1 + (np.abs(state_ - drives_) / (np.sqrt(gain_) * ac_ * np.sqrt(kext_))) ** 2) ** beta_ * (
                                       state_ - drives_)

        return ho_term + damp_term + nonlin_damp_term - np.exp(-1.0j * phase_) * kext_ * drives_ + ddrives_


def tls_real(time_, state_real, f0_, k0_, ktls_, ac_, beta_, F1_, F2_, fd1_, fd2_, kext_, psi_,
             phase_, gain_, att_, in_out):
    state_complex = state_real[::2] + 1j * state_real[1::2]
    ret_complex = tls_complex(time_, state_complex, f0_, k0_, ktls_, ac_, beta_, F1_, F2_, fd1_, fd2_, kext_, psi_,
                              phase_, gain_, att_, in_out)
    tmp = np.zeros(len(ret_complex) * 2)
    tmp[::2] = np.real(ret_complex)
    tmp[1::2] = np.imag(ret_complex)

    return tmp


def tls_simulation(time_, y0_, N_, dt_,
                   f0_, k0_, ktls_, ac_, beta_, F1_, F2_, fd1_, fd2_, kext_, psi_,
                   phase_=0, gain_=1, att_=1, in_out=False):
    # Integrator
    o = ode(tls_real).set_integrator('lsoda', atol=1e-11, rtol=1e-11)
    o.set_f_params(f0_, k0_, ktls_, ac_, beta_, F1_, F2_, fd1_, fd2_, kext_, psi_,
                   phase_, gain_, att_, in_out)
    o.set_initial_value(y0_)

    # Time-domain output field
    y_all = np.zeros((len(time_), len(y0_)))
    for i, t_val in enumerate(time_):
        o.integrate(t_val)
        y_all[i] = o.y

    # Merge the results onto the complex plane
    field_all = y_all[:, 0] + 1.0j * y_all[:, 1]

    # We save one oscillation once we reached the steady state
    field = field_all[-N_ - 1:-1]
    t_ = time_[-N_ - 1:-1]

    # Fourier domain solution
    Field = np.fft.fft(field) / len(field)
    frequencies = np.fft.fftfreq(len(t_), d=dt_)

    return Field, field, frequencies, t_


def tls_reconstruction(Field, Field_in, field, field_in, freqs, indices, order, phase=0, in_out=False, daniel=False):
    # Angular frequency
    w = 2 * np.pi * freqs

    # Create H-matrix
    if not in_out:
        if daniel:
            col1 = Field_in[indices]
        else:
            col1 = (1.0j * w * Field)[indices]

        col2 = (1.0j * Field)[indices]

        # Merge columns
        H = np.vstack((col1, col2))

        # Non-linear damping terms
        for exp_ind, exp_val in enumerate(order):
            col = np.fft.fft(np.abs(field) ** exp_val * field)[indices] / len(field)
            H = np.vstack((H, col))

    else:
        if daniel:
            col1 = np.exp(-1j * phase) * Field_in[indices]
        else:
            col1 = (1.0j * w * (Field - Field_in))[indices]

        col2 = (1.0j * (Field - Field_in))[indices]

        # Merge columns
        H = np.vstack((col1, col2))

        # Non-linear damping terms
        for exp_ind, exp_val in enumerate(order):
            col = np.fft.fft(np.abs(field - field_in) ** exp_val * (field - field_in))[indices] / len(field)
            H = np.vstack((H, col))

    # Making the matrix real instead of complex
    Hcos = np.real(H)
    Hsin = np.imag(H)
    H = np.hstack((Hcos, Hsin))

    # Normalize H for a more stable inversion
    Nm = np.diag(1. / np.max(np.abs(H), axis=1))
    H_norm = np.dot(Nm, H)  # normalized H-matrix

    # The drive vector, Q (from the Yasuda paper)
    if not in_out:
        if daniel:
            Qcos = np.real(-1.0j * w * Field)[indices]
            Qsin = np.imag(-1.0j * w * Field)[indices]
        else:
            Qcos = np.real(-Field_in)[indices]
            Qsin = np.imag(-Field_in)[indices]

    else:
        if daniel:
            Qcos = np.real(-1.0j * w * (Field - Field_in))[indices]
            Qsin = np.imag(-1.0j * w * (Field - Field_in))[indices]
        else:
            Qcos = np.real(-np.exp(-1j * phase) * Field_in)[indices]
            Qsin = np.imag(-np.exp(-1j * phase) * Field_in)[indices]
            # Qcos = np.real(-Field_in)[indices]
            # Qsin = np.imag(-Field_in)[indices]

    Qmat = np.hstack((Qcos, Qsin))

    # Solve system Q = H*p
    H_norm_inv = scipy.linalg.pinv(H_norm)
    p_norm = np.dot(Qmat, H_norm_inv)

    # Re-normalize p-values
    # Note: we have actually solved Q = H * Nm * Ninv * p
    # Thus we obtained Ninv*p and multiply by Nm to obtain p
    p = np.dot(Nm, p_norm)  # re-normalize parameter values

    # Forward calculation to check result, should be almost everything zero vector
    Q_fit = np.dot(p, H)

    # Scale parameters by drive force assuming known resonant frequency
    param_rec = p

    # Parameters reconstructed
    if not in_out:
        if daniel:
            kext_rec = param_rec[0] ** 2
            f0_rec = param_rec[1] / (2 * np.pi)
            c_rec = param_rec[2:]
        else:
            kext_rec = 1 / param_rec[0] ** 2
            f0_rec = param_rec[1] / (2 * np.pi) / param_rec[0]
            c_rec = param_rec[2:] / param_rec[0]

        return kext_rec, f0_rec, c_rec, Q_fit

    else:
        if daniel:
            kext_rec = param_rec[0]
            f0_rec = param_rec[1] / (2 * np.pi)
            c_rec = param_rec[2:]
        else:
            kext_rec = 1 / param_rec[0]
            f0_rec = param_rec[1] / (2 * np.pi) / param_rec[0]
            c_rec = param_rec[2:] / param_rec[0]

        return kext_rec, f0_rec, c_rec, Q_fit

