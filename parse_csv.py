import csv
import numpy as np
from utils import *

# -------------- Import digital control signals from csv file -------------- #
Ts = 54e-6 # Sampling time
file_name = 'ctrl_sig/digital_control_1v0in_single'
csv_file = file_name + '.csv'
npy_file = file_name + '.npy'

# ---------------------------------------------------------------------------- #
print('# Get total row count')
# ---------------------------------------------------------------------------- #

with open(csv_file) as cf0:
    csv_reader = csv.reader(cf0)
    next(csv_reader)  # Skip first line
    row_count = sum(1 for row in csv_reader)

time_stamps = np.zeros(row_count)  # To hold all time stamps
i=0

# ---------------------------------------------------------------------------- #
print('# Read total simulation time')
# ---------------------------------------------------------------------------- #

with open(csv_file) as cf1:
    csv_reader = csv.reader(cf1)
    next(csv_reader)  # Skip first line

    for row in csv_reader:
        #print(row)
        time_stamps[i] = row[0]
        i += 1

time_len = time_stamps[-1]
n_ctrl_signals = int(np.floor(time_len/Ts))

s = np.zeros((5, n_ctrl_signals))  # Declare control signal array


# ---------------------------------------------------------------------------- #
print('# Start Reading CSV file')
# ---------------------------------------------------------------------------- #

with open(csv_file) as cf:
    csv_reader = csv.reader(cf)

    # Extract data
    next(csv_reader)  # Skip first line
    i = 0
    for row in csv_reader:
        time_stamp = float(row[0]) - i*54e-6
        if time_stamp >= 54e-6:
            s[0,i] = 1 if float(row[1]) > 1.25 else -1
            s[1,i] = 1 if float(row[3]) > 1.25 else -1
            s[2,i] = 1 if float(row[5]) > 1.25 else -1
            s[3,i] = 1 if float(row[7]) > 1.25 else -1
            s[4,i] = 1 if float(row[9]) > 1.25 else -1
            i += 1

# -------------- Save numpy array to text file -------------- #
with open(npy_file, 'wb') as f:
    np.save(f, s)
