import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import scipy.signal as signal

# %% import_data (adjusted from lab3functions)
def import_data(separator):
    def time_norm(data):
        a = list(data.iloc[:][:]['t'])
        b = list(data.iloc[:][:]['t'])

        for u in range(len(a) - 1):
            if a[u] > a[u + 1]:
                if b[u] > b[u + 1]:
                    offset = a[u] - a[u + 1] + 1
                    a[u + 1] = offset + a[u + 1]
                    u += 1
                else:
                    a[u + 1] = offset + a[u + 1]
                    u += 1

        output = pd.DataFrame({'emg_chest': data.emg_chest, 'emg_shoulder': data.emg_shoulder, 't': a})
        output.reset_index(inplace=True, drop=True)
        return output

    column_names = [
        'emg_chest', 'emg_shoulder', 't'
    ]
    # Creating an empty Dataframe with column names only
    flys_raw = pd.DataFrame(columns=column_names)
    mvc_raw = pd.DataFrame(columns=column_names)

    for i in range(1,3):
        mvc_string = f"./data/mvc_test/mvc_test_{i}.txt"

        mcv_data = pd.read_csv(
            mvc_string,
            sep=separator, names=column_names, skiprows=50,
            skipfooter=50
        )
        mvc_raw = pd.concat([mvc_raw, mcv_data], ignore_index=True)

    for i in range(1,4):
        flys_string = f"data/dumbbell_flys_angle_test/dumbbell_flys_test_{i}.txt"
        flys_data = pd.read_csv(flys_string,
                                sep=separator, names=column_names, skiprows=50,
                                skipfooter=50)
        flys_raw = pd.concat([flys_raw, flys_data], ignore_index=True)

    flys = time_norm(flys_raw)
    mvc = time_norm(mvc_raw)
    return flys, mvc


# %%
# MVC

def raw(emgC, emgS, mvc_time):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(mvc_time / 1000, emgC)
    axs[0].set_title("MVC Raw EMG Chest")
    axs[1].plot(mvc_time / 1000, emgS)
    axs[1].set_title("MVC Raw EMG Shoulder")


def bandpass_filter(emgC, emgS, time):
    fs = 1000
    lowcut = 20
    highcut = 450
    nyq = 0.5 * fs
    high = highcut / nyq
    low = lowcut / nyq
    order = 4

    b, a = signal.butter(order, [low, high], "bandpass", analog=False)
    filtered_chest_signal = signal.filtfilt(b, a, emgC, axis=0)

    b, a = signal.butter(order, [low, high], "bandpass", analog=False)
    filtered_shoudler_signal = signal.filtfilt(b, a, emgS, axis=0)

    return filtered_chest_signal, filtered_shoudler_signal


def rectifier(emgC, emgS, time):
    AbsolutwerteC = np.abs(emgC)
    AbsolutwerteS = np.abs(emgS)

    return AbsolutwerteC, AbsolutwerteS


def envelope(emgc, emgs, time):
    fs = 1000
    lowcut = 5
    nyq = 0.5 * fs
    low = lowcut / nyq
    order = 4

    b, a = signal.butter(order, [low], "lowpass", analog=False)
    yC = signal.filtfilt(b, a, emgc, axis=0)

    b, a = signal.butter(order, [low], "lowpass", analog=False)
    yS = signal.filtfilt(b, a, emgs, axis=0)

    return yC, yS


def plot(SR, SF, SE, yS, time, time1):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs[0][0].plot(time / 1000, SR, label="Raw Signal")
    axs[0][1].plot(time / 1000, SF, label="Filtered Signal")
    axs[1][0].plot(time / 1000, SE, label="Rectified & Enveloped Signal")
    axs[1][1].plot(time1 / 1000, yS, label="3rd Burst Signal")
    axs[1][1].axvline(x=91, ymin=0, ymax=1800, color="green", label="0° Markers")
    axs[1][1].axvline(x=95, ymin=0, ymax=1800, color="green")
    axs[1][1].axvline(x=101, ymin=0, ymax=1800, color="yellow", label="45° Markers")
    axs[1][1].axvline(x=105, ymin=0, ymax=1800, color="yellow")
    axs[1][1].axvline(x=81, ymin=0, ymax=1800, color="red", label="90° Markers", alpha=0.6)
    axs[1][1].axvline(x=85, ymin=0, ymax=1800, color="red", alpha=0.6)

    plt.savefig("Verarbeitung.png")

    for ax in axs.flat:
        ax.set_ylabel('EMG (mV)', labelpad=2)
        ax.set_xlabel('Time (sec)', labelpad=2)
        ax.legend(loc="upper left")

    return yS


def plot_all(CR, SR, CF, SF, CA, SA, CE, SE, time):
    fig, axs = plt.subplots(4, 2, figsize=(15, 20))
    axs[0][0].plot(time / 1000, CR, label="Raw Signal")
    axs[0][0].set_title("Chest")
    axs[0][1].plot(time / 1000, SR, label="Raw Signal")
    axs[0][1].set_title("Shoulder")
    axs[1][0].plot(time / 1000, CF, label="Filtered Signal")
    axs[1][1].plot(time / 1000, SF, label="Filtered Signal")
    axs[2][0].plot(time / 1000, CA, label="Rectified Signal")
    axs[2][1].plot(time / 1000, SA, label="Rectified Signal")
    axs[3][0].plot(time / 1000, CE, label="Enveloped Signal")
    axs[3][1].plot(time / 1000, SE, label="Enveloped Signal")

    for ax in axs.flat:
        ax.set_ylabel('EMG (mV)', labelpad=1)
        ax.set_xlabel('Time (sec)', labelpad=2)
        ax.legend(loc="upper right")




def seperate_1(emgc, emgs, time):
    fs = 1000
    lowcut = 15 
    nyq = 0.5 * fs
    low = lowcut / nyq
    order = 4

    b, a = signal.butter(order, [low], "lowpass", analog=False)
    yC = signal.filtfilt(b, a, emgc, axis=0)

    b, a = signal.butter(order, [low], "lowpass", analog=False)
    yS = signal.filtfilt(b, a, emgs, axis=0)

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(time / 1000, yC)
    axs[0].set_title("EMG Chest")
    axs[1].plot(time / 1000, yS)
    axs[1].set_title("EMG Shoulder")

    axs[0].axvline(x=22.5, ymin=0, ymax=1800, color="red", label="90°")
    axs[0].axvline(x=26.5, ymin=0, ymax=1800, color="red")
    axs[0].axvline(x=33.5, ymin=0, ymax=1800, color="green", label="0°")
    axs[0].axvline(x=37.5, ymin=0, ymax=1800, color="green")
    axs[0].axvline(x=42, ymin=0, ymax=1800, color="yellow", label="45°")
    axs[0].axvline(x=46, ymin=0, ymax=1800, color="yellow")

    axs[1].axvline(x=22.5, ymin=0, ymax=1800, color="red", label="90°")
    axs[1].axvline(x=26.5, ymin=0, ymax=1800, color="red")
    axs[1].axvline(x=33.5, ymin=0, ymax=1800, color="green", label="0°")
    axs[1].axvline(x=37.5, ymin=0, ymax=1800, color="green")
    axs[1].axvline(x=42, ymin=0, ymax=1800, color="yellow", label="45°")
    axs[1].axvline(x=46, ymin=0, ymax=1800, color="yellow")

    for ax in axs.flat:
        ax.set_ylabel('EMG (mV)', labelpad=1)
        ax.set_xlabel('Time (sec)', labelpad=2)
        ax.legend(loc="upper right")

    return yC, yS


def seperate_2(emgc, emgs, time):
    fs = 1000
    lowcut = 15 
    nyq = 0.5 * fs
    low = lowcut / nyq
    order = 4

    b, a = signal.butter(order, [low], "lowpass", analog=False)
    yC = signal.filtfilt(b, a, emgc, axis=0)

    b, a = signal.butter(order, [low], "lowpass", analog=False)
    yS = signal.filtfilt(b, a, emgs, axis=0)

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(time / 1000, yC)
    axs[0].set_title("EMG Chest")
    axs[1].plot(time / 1000, yS)
    axs[1].set_title("EMG Shoulder")

    axs[0].axvline(x=51, ymin=0, ymax=1800, color="red", label="90°")
    axs[0].axvline(x=55, ymin=0, ymax=1800, color="red")
    axs[0].axvline(x=59, ymin=0, ymax=1800, color="green", label="0°")
    axs[0].axvline(x=63, ymin=0, ymax=1800, color="green")
    axs[0].axvline(x=67.5, ymin=0, ymax=1800, color="yellow", label="45°")
    axs[0].axvline(x=71.5, ymin=0, ymax=1800, color="yellow")

    axs[1].axvline(x=51, ymin=0, ymax=1800, color="red", label="90°")
    axs[1].axvline(x=55, ymin=0, ymax=1800, color="red")
    axs[1].axvline(x=59, ymin=0, ymax=1800, color="green", label="0°")
    axs[1].axvline(x=63, ymin=0, ymax=1800, color="green")
    axs[1].axvline(x=67.5, ymin=0, ymax=1800, color="yellow", label="45°")
    axs[1].axvline(x=71.5, ymin=0, ymax=1800, color="yellow")

    for ax in axs.flat:
        ax.set_ylabel('EMG (mV)', labelpad=1)
        ax.set_xlabel('Time (sec)', labelpad=2)
        ax.legend(loc="upper right")

    return yC, yS


def seperate_3(emgc, emgs, time):
    fs = 1000
    lowcut = 15 
    nyq = 0.5 * fs
    low = lowcut / nyq
    order = 4

    b, a = signal.butter(order, [low], "lowpass", analog=False)
    yC = signal.filtfilt(b, a, emgc, axis=0)

    b, a = signal.butter(order, [low], "lowpass", analog=False)
    yS = signal.filtfilt(b, a, emgs, axis=0)

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(time / 1000, yC)
    axs[0].set_title("EMG Chest")
    axs[1].plot(time / 1000, yS)
    axs[1].set_title("EMG Shoulder")

    axs[0].axvline(x=81, ymin=0, ymax=1800, color="red", label="90°")
    axs[0].axvline(x=85, ymin=0, ymax=1800, color="red")
    axs[0].axvline(x=91, ymin=0, ymax=1800, color="green", label="0°")
    axs[0].axvline(x=95, ymin=0, ymax=1800, color="green")
    axs[0].axvline(x=101, ymin=0, ymax=1800, color="yellow", label="45°")
    axs[0].axvline(x=105, ymin=0, ymax=1800, color="yellow")

    axs[1].axvline(x=81, ymin=0, ymax=1800, color="red", label="90°")
    axs[1].axvline(x=85, ymin=0, ymax=1800, color="red")
    axs[1].axvline(x=91, ymin=0, ymax=1800, color="green", label="0°")
    axs[1].axvline(x=95, ymin=0, ymax=1800, color="green")
    axs[1].axvline(x=101, ymin=0, ymax=1800, color="yellow", label="45°")
    axs[1].axvline(x=105, ymin=0, ymax=1800, color="yellow")

    for ax in axs.flat:
        ax.set_ylabel('EMG (mV)', labelpad=1)
        ax.set_xlabel('Time (sec)', labelpad=2)
        ax.legend(loc="upper right")

    return yC, yS


def bandpass_filter_mvcvalue(emg, time):
    fs = 1000
    lowcut = 20
    highcut = 450
    nyq = 0.5 * fs
    high = highcut / nyq
    low = lowcut / nyq
    order = 4
    b, a = signal.butter(order, [low, high], "bandpass", analog=False)
    Filtered = signal.filtfilt(b, a, emg, axis=0)
    return Filtered


def rectifier_mvcvalue(emg, time):
    Absolutwerte = np.abs(emg)

    return Absolutwerte


def envelope_mvcvalue(emg, time):
    fs = 1000
    lowcut = 5
    nyq = 0.5 * fs
    low = lowcut / nyq
    order = 4
    b, a = signal.butter(order, [low], "lowpass", analog=False)
    y = signal.filtfilt(b, a, emg, axis=0)
    return y


def mvc_value(emg, time):
    a = bandpass_filter_mvcvalue(emg, time)
    b = rectifier_mvcvalue(a, time)
    c = envelope_mvcvalue(b, time)
    d = np.mean(c)
    return d


def create_bar_chart(Percent1C, Percent2C, Percent3C, Percent1S, Percent2S, Percent3S):
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    position = ['0°', '45°', '90°']
    index = np.arange(3)
    bar_width = 0.35
    opacity = 0.8
    PercentC = [Percent1C, Percent2C, Percent3C]
    PercentS = [Percent1S, Percent2S, Percent3S]
    plt.xticks(index + (bar_width / 2), position)
    plt.xlabel("Position")
    plt.ylabel("% of MVC Value")
    ax.set_title("Relative Muscle Activity")
    ax.bar(index, PercentC, bar_width, alpha=opacity, label='Chest')
    ax.bar(index + bar_width, PercentS, bar_width, alpha=opacity, label='Shoulder')
    ax.legend(loc="upper right")
    plt.savefig("Relative Muscle Activity.png")


# %%
def trajectory_position(Data1, Data2, Data3):
    Einhüllende1 = pd.DataFrame(Data1)
    Einhüllende2 = pd.DataFrame(Data2)
    Einhüllende3 = pd.DataFrame(Data3)

    length = len(Einhüllende1)
    array = np.ones(length)

    for i in range(0, length):
        array[i] = np.average([Einhüllende1.iloc[i], Einhüllende2.iloc[i], Einhüllende3.iloc[i]])
    time = np.linspace(0, length, length)
    return array
