import numpy as np
import pandas as pd
import wfdb
import matplotlib.pyplot as plt
from scipy.stats import f_oneway, ttest_ind
import glob
import os
import neurokit2 as nk
from scipy.signal import firwin, lfilter

""" This piece of code downloads the ECG data, calculates QT intervals and performs the 
anova tests for different genotype groups (AA, AB, BB). The ECG data analysis, i.e., calculating the
QT intervals is ready-made for you. Finding the associations between different groups is done by you. 
Hint for these are in the code! Read all the comments carefully!!"""

# MODIFY YOUR PATH
path = 'D:/binf_python/QTproject/new'


# This function loads the ECG data files: .dat, and .hea -files. .dat files consists of the ECG voltage data
# and .hea files consist of the header information (e.g., sampling frequency, leads etc.)
def loadEcg(path):
    dat_files = glob.glob(os.path.join(path, "*.dat"))
    hea_files = glob.glob(os.path.join(path, "*.hea"))
    base_names = set([os.path.splitext(os.path.basename(f))[0] for f in dat_files + hea_files])

    ecg_dict, field_dict, fs_dict = {}, {}, {}

    # Read the signal and metadata for each file. The read file consist of field names (contain, e.g.,
    # the sampling frequency and the lead names), and the actual ecg signal data.
    for i, base_name in enumerate(sorted(base_names), start=1):
        ecg, fields = wfdb.rdsamp(os.path.join(path, base_name))
        patient_key = f'Patient{i}'
        ecg_dict[patient_key] = ecg
        field_dict[patient_key] = fields
        fs_dict[patient_key] = fields['fs']
    return ecg_dict, field_dict, fs_dict


# Function to filter ECG signal. You can modify the filter, i.e. create your own IIR
# or FIR filter with order and cut-off of your choosing. Justify your selection!
def filterEcg(signal, fs, filter_order=20, lowcut=0.5, highcut=150):
    nyquist = 0.5 * fs  # Nyquist frequency is half of the sampling frequency
    low = lowcut / nyquist  # Normalize the low cut frequency
    high = highcut / nyquist  # Normalize the high cut frequency

    # Design the FIR bandpass filter
    filter_coeffs = firwin(filter_order + 1, [low, high], pass_zero=False)    
    filtered_signal = lfilter(filter_coeffs, [1.0], signal[:, 1])

    return filtered_signal

# Function to calculate QT intervals.
    # Process the ECG signal using neurokit2 (detects R-peaks, Q, and T points)
    # ecg_process has many steps:
    # 1) cleaning the ecg with ecg_clean()
    # 2) peak detection with ecg_peaks()
    # 3) HR calculus with signal_rate()
    # 4) signal quality assessment with ecg_quality()
    # 5) QRS elimination with ecg_delineate() and
    # 6) cardiac phase determination with ecg_phase().
def calculateQtIntervals(key, filtered_signal, fs):
    ecg_analysis, _ = nk.ecg_process(filtered_signal, sampling_rate=fs)
    q_points = ecg_analysis['ECG_Q_Peaks'] # This is default output of the ecg_process.
    t_points = ecg_analysis['ECG_T_Offsets'] # This is default output of the ecg_process.
    q_indices = q_points[q_points == 1].index.to_list()
    t_indices = t_points[t_points == 1].index.to_list()

    time = np.arange(filtered_signal.size) / fs

    # Calculate QT intervals and plot them as red lines
    qt_intervals = []
    for idx, (q, t) in enumerate(zip(q_indices, t_indices), start=1):
        if t > q:  # Ensure T point is after Q point for a valid QT interval
            qt_interval = (t - q) / fs  # The indexes are in samples, thus we need to convert them to seconds.
            qt_intervals.append(qt_interval)

            # Plot the QT interval as a red line segment. YOU NEED TO RUN ALSO THESE FOR TASK 2.
            #plt.plot([q / fs, t / fs], [filtered_signal[q], filtered_signal[t]], color='red', lw=2,
               #     label='QT Interval' if len(qt_intervals) == 1 else "")  # Label only the first for legend clarity
            #plt.title(f'{key} - QT Interval {idx}')
            #plt.show()
            
    return qt_intervals


# Function to calculate and store average QT interval
def calculateAverageQt(ecg_dict, fs_dict):
    # The average QT intervals for all patients will be stored in average_qt_dict
    average_qt_dict = {}
    for key, ecg_signal in ecg_dict.items():
        fs = fs_dict[key] # corresponding sampling freq. for each signal

        leadII_signal = ecg_signal[:, 1]
        filtered_signal = filterEcg(ecg_signal, fs) # This calls the filtering function
        time = np.arange(leadII_signal.size) / fs

        """
        plt.figure(figsize=(12, 6))
        # Plot original signal
        plt.subplot(2, 1, 1)
        plt.plot(time, leadII_signal, label='Original ECG signal', color='blue')
        plt.title(f'{key} - Original ECG Lead II')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)

        # Plot filtered signal
        plt.subplot(2, 1, 2)
        plt.plot(time, filtered_signal, label='Filtered ECG signal', color='green')
        plt.title(f'{key} - Filtered ECG Lead II')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)

        # Show the plots
        plt.tight_layout()
        plt.show()
        """

        qt_intervals = calculateQtIntervals(key, filtered_signal, fs) # Calculates the intervals based on the filtered data
        average_qt_interval = np.mean(qt_intervals) if qt_intervals else None # Calculates the average QT for each patient.
        average_qt_dict[key] = average_qt_interval
    return average_qt_dict


# TASK 3: 
# Function to load genotype data and reshape (reshaping to keep the structure for 7 rows). New genotype starts
# every 7th row (if you look at the result txt file, you can see this).
def loadAndReshapeGenotype(filepath='extracted_genotypes.txt'):
    results = pd.read_csv(filepath, delimiter="\t", header=None)
    # Select every 7th column starting from index 1 --> each new genotype AA, AB or BB.
    selected = results.iloc[:, 1::7]  
    s_array = selected.values

    # Automatically calculate columns based on data size, when we want to have
    # the original 7 rows.
    reshaped = s_array.reshape(7, -1)
    
    # the return array has shape (7,130) in which 7 represents 7 SNPs
    return reshaped

# Function to extract QT intervals based on genotype AA, AB or BB. Thus, this function goes through the reshaped
# data, consisting of 7 rows and 129 columns. 129 is the number of different genotypes i.e. different patients.
# One group to study = same genotype from one row. E.g. all BB genotypes from row 1. YOU NEED THIS INFORMATION ABOUT
# ALL 7 ROWS.
def QtByGenotype(reshaped, average_qt_dict):
    patients = list(average_qt_dict.keys()) # patients will be a list of integers from 1 to 129.

    # Change these to calculate for different rows (1-7). 
    AB1 = np.where(reshaped[7, :] == "AB")[0]  
    BB1 = np.where(reshaped[7, :] == "BB")[0]
    AA1 = np.where(reshaped[7, :] == "AA")[0]

    qt_AB1 = [average_qt_dict[patients[idx]] for idx in AB1 if average_qt_dict[patients[idx]] is not None]
    qt_BB1 = [average_qt_dict[patients[idx]] for idx in BB1 if average_qt_dict[patients[idx]] is not None]
    qt_AA1 = [average_qt_dict[patients[idx]] for idx in AA1 if average_qt_dict[patients[idx]] is not None]

    return qt_AB1, qt_BB1, qt_AA1


# Function to perform ANOVA and print results.
def anovaTest(qt_AB3, qt_BB3, qt_AA3):
    # PERFORM HERE ANOVA TEST (OR T-TEST IF YOU WISH) WITH THE THREE DIFFERENT VARIABLES.
    # STATISTICAL TESTS FOR ALL 7 GROUPS / ROWS.
    if qt_AB3 and qt_BB3 and qt_AA3:
        # Perform one-way ANOVA test for three groups (AB, BB, AA)
        f_stat, p_value = f_oneway(qt_AB3, qt_BB3, qt_AA3)
        print(f"ANOVA results: F-statistic = {f_stat:.2f}, p-value = {p_value:.2f}")
    else:
        # If one of the groups is missing, perform t-test between the two remaining groups
        if not qt_AB3:
            print("No AB group found. Performing t-test between AA and BB.")
            t_stat, p_value = ttest_ind(qt_AA3, qt_BB3)
        elif not qt_BB3:
            print("No BB group found. Performing t-test between AA and AB.")
            t_stat, p_value = ttest_ind(qt_AA3, qt_AB3)
        elif not qt_AA3:
            print("No AA group found. Performing t-test between AB and BB.")
            t_stat, p_value = ttest_ind(qt_AB3, qt_BB3)

        print(f"T-test results: T-statistic = {t_stat:.2f}, p-value = {p_value:.2f}")

# Main processing function
def main():
    ecg_dict, field_dict, fs_dict = loadEcg(path)
    average_qt_dict = calculateAverageQt(ecg_dict, fs_dict)

    """"
    for j, (key, value) in enumerate(fs_dict.items()):
        if j < 10:
            print(f"{key}: {value} Hz")
        else:
            print("")
            break

    for i, (key_qt, value_qt) in enumerate(average_qt_dict.items()):
        if i < 10:
            print(f"{key_qt}: {value_qt:.2f}")
        else:
            break

    """
    # Load genotype data (from the file you prepared in Task 1)
    reshaped_genotype = loadAndReshapeGenotype()

    # CALCULATE THE DIFFERENT GROUPS HERE, E.G. FOR GROUP 3 (ROW 3) QT_AB3, QT_BB3, QT_AA3. USE THE
    qt_AB3, qt_BB3, qt_AA3 = QtByGenotype(reshaped_genotype, average_qt_dict)

    # Perform ANOVA test (or t-test if one group is missing)
    anovaTest(qt_AB3, qt_BB3, qt_AA3)
    

# Run the main function
if __name__ == "__main__":
    main()
