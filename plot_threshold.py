''' 
Plot the AUC and ACC against fraction of stragglers
## User need to input file_paths and x values
## This script assumes user log are named in given format: using 0.1 straggling fraction as example:
##      results/0_1/BIBD_0_1_1.txt
'''

import re
import matplotlib.pyplot as plt
import numpy as np


AUC_REF = 0.883
ACC_REF = 0.953

############################### User inputs ###############################
x = [0.1, 0.25, 0.4]        ## TODO: User input - Fractions of stragglers
path = f"/home/ubuntu/log_thtest" ## TODO: User input - path to logs
###########################################################################

## Get the y values required for plottings
def get_y(path, test_name, rounds, x):
    AUC_y = []
    AUC_y_max =[]
    AUC_y_min = []
    ACC_y = []
    ACC_y_max =[]
    ACC_y_min = []

    for i in range(len(x)):
        # Convert the decimal number to a string
        x_str = str(x[i])
        # Replace the decimal point with an underscore
        x_mod = x_str.replace('.', '_')

        file_prefix = f"{path}/{test_name}_{x_mod}"
        AUC_tmp, ACC_tmp = process_file(file_prefix, rounds)

        # store the avg, max, and min values
        AUC_y.append(np.average(AUC_tmp))
        AUC_y_max.append(np.max(AUC_tmp))
        AUC_y_min.append(np.min(AUC_tmp))
        ACC_y.append(np.average(ACC_tmp))
        ACC_y_max.append(np.max(ACC_tmp))
        ACC_y_min.append(np.min(ACC_tmp))

    return AUC_y,AUC_y_max,AUC_y_min,ACC_y,ACC_y_max,ACC_y_min


def process_file(file_prefix, rounds):
    # returned two array of all data
    AUC_list = np.zeros(rounds)
    ACC_list = np.zeros(rounds)

    # Iterate through the files BIBD_1.txt to BIBD_75.txt
    for i in range(1, rounds+1):
        file_name = f"{file_prefix}_{i}.txt"
        
        # Open and read the file
        try:
            with open(file_name, 'r') as file:
                lines = file.readlines()
                
                # extract 2nd last line
                AUC_ACC_line = lines[-2].strip()

                # Regex patterns to extract AUC, ACC, and iter_time
                auc_acc_pattern = r'AUC\s*=\s*(\d+\.\d+),\s*ACC\s*=\s*(\d+\.\d+)'

                # Search for the AUC and ACC values in the 3rd last line
                auc_acc_match = re.search(auc_acc_pattern, AUC_ACC_line)
                if auc_acc_match:
                    AUC_list[i-1] = float(auc_acc_match.group(1))
                    ACC_list[i-1] = float(auc_acc_match.group(2))
                else:   # not found
                    AUC_list[i-1] = 0
                    ACC_list[i-1] = 0
                    print(f"AUC or ACC not found in {file_name}")
        
        except FileNotFoundError:
            AUC_list[i-1] = 0
            ACC_list[i-1] = 0
            print(f"File {file_name} not found.")
        except IndexError:
            AUC_list[i-1] = 0
            ACC_list[i-1] = 0
            print(f"File {file_name} does not have enough lines.")
        except Exception as e:
            AUC_list[i-1] = 0
            ACC_list[i-1] = 0
            print(f"An error occurred while processing {file_name}: {str(e)}")

    return AUC_list, ACC_list


# Plot values x and y
##################################
BIBD_AUC_y, BIBD_AUC_y_max, BIBD_AUC_y_min, BIBD_ACC_y, BIBD_ACC_y_max, BIBD_ACC_y_min = get_y(path,"BIBD",10,x)
SPG_AUC_y, SPG_AUC_y_max, SPG_AUC_y_min, SPG_ACC_y, SPG_ACC_y_max, SPG_ACC_y_min = get_y(path,"SPG",10,x)



############################################# BIBD results #############################################
## Plotting AUC
# Calculate error bars
AUC_y_err = [[BIBD_AUC_y[i] - BIBD_AUC_y_min[i] for i in range(len(x))],
         [BIBD_AUC_y_max[i] - BIBD_AUC_y[i] for i in range(len(x))]]

# Plotting with error bars
plt.errorbar(x, BIBD_AUC_y, yerr=AUC_y_err, capsize=3, fmt='--o', label='BIBD AUC y', color="blue", ecolor="black")
plt.fill_between(x, BIBD_AUC_y_min, BIBD_AUC_y_max, alpha=0.2)

# Plot horizontal line of the average iteration time excluding zeros
plt.axhline(y=AUC_REF, color='g', linestyle='--', label=f'AUC reference = {AUC_REF:.2f}')

## Plotting ACC
ACC_y_err = [[BIBD_ACC_y[i] - BIBD_ACC_y_min[i] for i in range(len(x))],
             [BIBD_ACC_y_max[i] - BIBD_ACC_y[i] for i in range(len(x))]]

# Plotting with error bars
plt.errorbar(x, BIBD_ACC_y, yerr=ACC_y_err, capsize=3, fmt='--o', label='BIBD ACC y', color="orange", ecolor="black")
plt.fill_between(x, BIBD_ACC_y_min, BIBD_ACC_y_max, alpha=0.2)

# Plot horizontal line of the average iteration time excluding zeros
plt.axhline(y=ACC_REF, color='r', linestyle='--', label=f'ACC reference = {ACC_REF:.2f}')

# Adding labels and title
plt.xlabel('fraction of stragglers')
plt.ylabel('AUC/ACC')
plt.title('BIBD: AUC/ACC vs fraction of stragglers')
plt.legend()

# Show plot
plt.savefig("BIBD_threshold.png")
plt.show()



############################################# SPG results #############################################
## Plotting AUC
# Calculate error bars
AUC_y_err = [[SPG_AUC_y[i] - SPG_AUC_y_min[i] for i in range(len(x))],
         [SPG_AUC_y_max[i] - SPG_AUC_y[i] for i in range(len(x))]]

# Plotting with error bars
plt.errorbar(x, SPG_AUC_y, yerr=AUC_y_err, capsize=3, fmt='--o', label='SPG AUC y', color="blue", ecolor="black")
plt.fill_between(x, SPG_AUC_y_min, SPG_AUC_y_max, alpha=0.2)

# Plot horizontal line of the average iteration time excluding zeros
plt.axhline(y=AUC_REF, color='g', linestyle='--', label=f'AUC reference = {AUC_REF:.2f}')

## Plotting ACC
ACC_y_err = [[SPG_ACC_y[i] - SPG_ACC_y_min[i] for i in range(len(x))],
             [SPG_ACC_y_max[i] - SPG_ACC_y[i] for i in range(len(x))]]

# Plotting with error bars
plt.errorbar(x, SPG_ACC_y, yerr=ACC_y_err, capsize=3, fmt='--o', label='SPG ACC y', color="orange", ecolor="black")
plt.fill_between(x, SPG_ACC_y_min, SPG_ACC_y_max, alpha=0.2)

# Plot horizontal line of the average iteration time excluding zeros
plt.axhline(y=ACC_REF, color='r', linestyle='--', label=f'ACC reference = {ACC_REF:.2f}')

# Adding labels and title
plt.xlabel('fraction of stragglers')
plt.ylabel('AUC/ACC')
plt.title('SPG: AUC/ACC vs fraction of stragglers')
plt.legend()

# Show plot
plt.savefig("SPG_threshold.png")
plt.show()

