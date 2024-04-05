import numpy as np
import time
import os
import h5py
import inspect
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import signal

import presto
from presto import lockin, utils
from presto.hardware import AdcFSample, AdcMode, DacFSample, DacMode

plt.rcParams['axes.formatter.useoffset'] = True
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['legend.frameon'] = False


############################################################################
# Saving methods

# Save script function
def save_script(folder, file, myrun, myrun_attrs):
    # Create folders if they do not exist
    if not os.path.isdir(folder):
        os.makedirs(folder)

    # String handles
    run_str = "{}".format(myrun)
    source_str = "{}/Source code".format(myrun)

    # Read lines of the script
    filename = inspect.getframeinfo(inspect.currentframe()).filename
    with open(filename, "r") as codefile:
        code_lines = codefile.readlines()

    # Write lines of script and run attributes
    with h5py.File(os.path.join(folder, file), "a") as savefile:

        dt = h5py.special_dtype(vlen=str)
        code_set = savefile.create_dataset(source_str.format(myrun), (len(code_lines),), dtype=dt)
        for ii in range(len(code_lines)):
            code_set[ii] = code_lines[ii]

        # Save the attributes of the run
        for key in myrun_attrs:
            savefile[run_str].attrs[key] = myrun_attrs[key]

    # Debug
    print("Saved script and run attributes.")


# Save data function
def save_data(folder, file, myrun, center_freq, detuning, amp_drive, usb_arr):
    if not os.path.isdir(folder):
        os.makedirs(folder)

    # Open the save file (.hdf5) in append mode
    with h5py.File(os.path.join(folder, file), "a") as savefile:
        # String as handles
        center_freq_data_str = "{}/center freq".format(myrun)
        detuning_data_str = "{}/comb detuning".format(myrun)
        drive_amp_data_str = "{}/drive amp sweep".format(myrun)
        usb_data_str = "{}/USB".format(myrun)

        # Write data to datasets
        savefile.create_dataset(center_freq_data_str, (np.shape(center_freq)),
                                dtype=float, data=center_freq)
        savefile.create_dataset(detuning_data_str, (np.shape(detuning)),
                                dtype=float, data=detuning)
        savefile.create_dataset(drive_amp_data_str, (np.shape(amp_drive)),
                                dtype=float, data=amp_drive)
        savefile.create_dataset(usb_data_str, (np.shape(usb_arr)),
                                dtype=complex, data=usb_arr)

        # Write dataset attributes
        savefile[center_freq_data_str].attrs["Unit"] = "Hz"
        savefile[detuning_data_str].attrs["Unit"] = "Hz"
        savefile[drive_amp_data_str].attrs["Unit"] = "fsu"
        savefile[usb_data_str].attrs["Unit"] = "fsu complex"


def dB(Sjk):
    return 20 * np.log10(np.abs(Sjk))


def plot2D(data, x, y, figure, axis, zmin_, zmax_):
    dx = (x[1] - x[0]) / 2
    dy = (y[1] - y[0]) / 2

    xmin, xmax = (x[0] - dx), (x[-1] + dx)
    ymin, ymax = (y[0] - dy), (y[-1] + dy)

    a = axis.imshow(dB(data),
                    origin="lower",
                    aspect="auto",
                    extent=[xmin, xmax, ymin, ymax],
                    vmin=zmin_,
                    vmax=zmax_,
                    cmap='viridis',
                    interpolation=None,
                    )

    figure.colorbar(a, ax=axis, aspect=50, label='signal amplitude [dBFS]')

    axis.set_xlabel('detuning [Hz]')
    axis.set_ylabel('center frequency [Hz]')


def main():
    # Saving folder location, saving file and run name
    save_folder = r'/media/jc/2022-11/HB-Data/'
    save_file = r'2023-07.hdf5'
    myrun: str = time.strftime("%Y-%m-%d_%H_%M_%S")
    t_start = time.strftime("%Y-%m-%d_%H_%M_%S")

    # Sample name and total attenuation along measurement chain
    sample = 'MPAF13_7_13'
    meas_type = 'freq sweep'
    atten = 0
    temperature = 300

    # Lab Network
    ADDRESS = '130.237.35.90'  # from Office
    # PORT = 42870              # Vivace ALFA
    # PORT = 42871              # Vivace BRAVO
    PORT = 42873  # Presto DELTA

    if PORT == 42870:
        Box = 'Vivace ALFA'
    elif PORT == 42871:
        Box = 'Vivace BETA'
    else:
        Box = 'Presto DELTA'

    # Physical Ports
    input_port = 2
    output_port = 2

    # DAC current
    DAC_CURRENT = 32_000  # uA

    # DAC/ADC Converters
    CONVERTER_CONFIGURATION = {
        "adc_mode": AdcMode.Mixed,
        "adc_fsample": AdcFSample.G2,
        "dac_mode": DacMode.Mixed02,
        "dac_fsample": DacFSample.G6,
    }

    # Pseudorandom noise (only when working with small amplitudes)
    dither = False

    # MEASUREMENT PARAMETERS
    # Resonant frequency
    fres = 4.345e9
    # NCO frequencies
    fNCO = 4.2e9
    # Bandwidth in Hz
    _df = 15e3
    # Number of pixels
    Npix = 5000
    Navg = 1
    # Number of pixels we discard
    Nskip = 0

    # DRIVES PARAMETERS
    # Number of drives
    nr_drive_freqs = 2
    # Signal output amplitude from Vivace/Presto
    amp_drive_min = -2
    amp_drive_max = np.log10(1 / nr_drive_freqs)
    nr_amp_drive = 25
    amp_drive_arr = np.logspace(amp_drive_min, amp_drive_max, nr_amp_drive)

    # CENTER FREQUENCY SWEEP PARAMETERS
    # Center frequency sweep
    _fcenter_start = fres - 20e6 - fNCO
    _fcenter_end = fres + 20e6 - fNCO
    # Number of frequencies of the sweep
    nr_center_freqs = 801
    # Center frequencies array
    _fcenter_arr = np.linspace(_fcenter_start, _fcenter_end, nr_center_freqs)
    fcenter_arr = np.zeros_like(_fcenter_arr)

    # LISTENING COMB PARAMETERS
    # Number of frequencies of the listening frequency comb
    nr_sig_freqs = 65

    # Instantiate lockin device
    with lockin.Lockin(address=ADDRESS,
                       port=PORT,
                       **CONVERTER_CONFIGURATION,
                       ) as lck:
        # Start timer
        t_start = time.strftime("%Y-%m-%d_%H_%M_%S")

        # Print Presto version
        print("Presto version: " + presto.__version__)

        # DAC current
        lck.hardware.set_dac_current(output_port, DAC_CURRENT)

        # Configure mixer
        lck.hardware.configure_mixer(freq=fNCO,
                                     in_ports=input_port,
                                     out_ports=output_port,
                                     )

        # Create output group for the drives
        og = lck.add_output_group(ports=output_port, nr_freq=nr_drive_freqs)
        og.set_phases(phases=np.zeros(nr_drive_freqs),
                      phases_q=np.full(nr_drive_freqs, -np.pi / 2))

        # Create input group for the listening comb
        ig = lck.add_input_group(port=input_port, nr_freq=nr_sig_freqs)

        # Add pseudorandom noise if needed
        lck.set_dither(dither, output_port)

        # Apply settings and wait
        lck.apply_settings()

        # Data array
        usb_arr = np.zeros((nr_amp_drive, nr_center_freqs, nr_sig_freqs), dtype=np.complex128)

        # Display nice progress bar
        with tqdm(total=(nr_amp_drive * nr_center_freqs), ncols=80) as pbar:

            # Drive amplitudes sweep
            for amp_ind, amp_val in enumerate(amp_drive_arr):

                og.set_amplitudes(amp_val * np.ones(nr_drive_freqs))

                # Center frequency sweep
                for fcenter_idx, _fcenter_val in enumerate(_fcenter_arr):
                    # Tune center frequency
                    fs_center, df = lck.tune(_fcenter_val, _df)
                    # Listening comb
                    fs_comb = fs_center + df * np.linspace(int(-nr_sig_freqs / 2), int(nr_sig_freqs / 2), nr_sig_freqs)
                    # Drive comb
                    fd_comb = fs_center + df * np.linspace(int(-nr_drive_freqs / 2), int(nr_drive_freqs / 2),
                                                           nr_drive_freqs)

                    # Save tuned center frequency
                    fcenter_arr[fcenter_idx] = fs_center

                    # Set df
                    lck.set_df(df)

                    # Set listening comb and drive comb frequencies
                    og.set_frequencies(fd_comb)
                    ig.set_frequencies(fs_comb)

                    # Apply settings
                    lck.apply_settings()
                    lck.hardware.sleep(1e-4, False)

                    # Get lock-in packets (pixels) from the local buffer
                    data = lck.get_pixels(Npix + Nskip, summed=False, nsum=Navg)
                    freqs, pixels_i, pixels_q = data[input_port]

                    # Convert a measured IQ pair into a low/high sideband pair
                    LSB, HSB = utils.untwist_downconversion(pixels_i, pixels_q)

                    usb_arr[amp_ind, fcenter_idx] = np.mean(HSB[-Npix:], axis=0)

                    # Update progress bar
                    pbar.update(1)

        # Mute outputs at the end of the sweep
        og.set_amplitudes(0.0)
        lck.apply_settings()

    # Stop timer
    t_end = time.strftime("%Y-%m-%d_%H_%M_%S")

    fdrive_det_arr = df * np.linspace(-nr_drive_freqs // 2, nr_drive_freqs // 2, nr_drive_freqs)
    fs_det_arr = df * np.linspace(-nr_sig_freqs // 2, nr_sig_freqs // 2, nr_sig_freqs)

    # Create dictionary with attributes
    myrun_attrs = {"Meas": meas_type,
                   "Instr": Box,
                   "T": temperature,
                   "Sample": sample,
                   "att": atten,
                   "4K-amp_out": 0,
                   "RT-amp_out": 0,
                   "RT-amp_in": 0,
                   "fNCO": fNCO,
                   "fcenter_start": _fcenter_start + fNCO,
                   "fcenter_stop": _fcenter_end + fNCO,
                   "nr_center_freq": nr_center_freqs,
                   "df": df,
                   "fd_detuning": fdrive_det_arr,
                   "fs_detuning": fs_det_arr,
                   "nr_sig_freqs": nr_sig_freqs,
                   "amp_drives": amp_drive_arr,
                   "Npixels": Npix,
                   "Naverages": Navg,
                   "Nskip": Nskip,
                   "Dither": dither,
                   "t_start": t_start,
                   "t_meas": t_end,
                   "Script name": os.path.basename(__file__),
                   }

    # Save script and attributes
    save_script(save_folder, save_file, myrun, myrun_attrs)

    # Save data
    save_data(save_folder, save_file, myrun, fcenter_arr + fNCO, fs_det_arr, amp_drive_arr, usb_arr)

    # Sanity check
    fig, ax = plt.subplots(1)
    zmin = -160
    zmax = -30
    for i in range(len(usb_arr)):
        fig2, ax2 = plt.subplots(1)
        plot2D(usb_arr[i], fs_det_arr, (fcenter_arr + fNCO) / 1e3, fig2, ax2, zmin, zmax)
        ax2.set_ylabel('frequency [kHz]')
        ax2.set_title(r'Pump Amplitude' + f' = {amp_drive_arr[i]:.3f} f.s.u.,   ')

        ax.plot((fNCO + fcenter_arr) / 1e3, signal.detrend(np.angle(usb_arr[i, :, 15])), '.-',
                label=f'{amp_drive_arr[i]:.3f}')

    ax.set_xlabel('frequency [kHz]')
    ax.set_ylabel('phase [rad]')
    ax.legend(title='$A_p$ [FS]')
    plt.show()


main()
