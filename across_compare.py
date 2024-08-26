## Plot the AUC, ACC and avg_time and compare horizontally across Naive, BIBD, and SPG

import re
import matplotlib.pyplot as plt
import numpy as np

# Initialize lists to store the extracted values
bibd_AUC = []
bibd_ACC = []
bibd_iter_time = []

spg_AUC = []
spg_ACC = []
spg_iter_time = []

naive_AUC = []
naive_ACC = []
naive_iter_time = []

error_run = []


# Counter list to keep track of the file indices
counter = []

# Input directory
input_dir = "log0901"

# Iterate through the files BIBD_1.txt to BIBD_75.txt
for i in range(1, 101):
    # file_name = f"bibd_multiple/BIBD_{i}.txt"
    file_name = input_dir + "/" + f"BIBD_{i}.txt"
    counter.append(i)
    # Open and read the file
    try:
        with open(file_name, 'r') as file:
            lines = file.readlines()
            
            # Extract the 3rd last and 2nd last lines
            third_last_line = lines[-2].strip()
            second_last_line = lines[-1].strip()

            # Regex patterns to extract AUC, ACC, and iter_time
            auc_acc_pattern = r'AUC\s*=\s*(\d+\.\d+),\s*ACC\s*=\s*(\d+\.\d+)'
            iter_time_pattern = r'avg iter_time:\s*(\d+\.\d+)'

            # Search for the AUC and ACC values in the 3rd last line
            auc_acc_match = re.search(auc_acc_pattern, third_last_line)
            if auc_acc_match:
                bibd_AUC.append(float(auc_acc_match.group(1)))
                bibd_ACC.append(float(auc_acc_match.group(2)))
            else:   # not found
                error_run.append(i)
                bibd_AUC.append(0)
                bibd_ACC.append(0)
                print(f"AUC and ACC not found in {file_name}")

            # Search for the iter_time in the 2nd last line
            iter_time_match = re.search(iter_time_pattern, second_last_line)
            if iter_time_match:
                bibd_iter_time.append(float(iter_time_match.group(1)))
            else:
                bibd_iter_time.append(0.0)
                print(f"avg_iter_time not found {file_name}")
    
    except FileNotFoundError:
        bibd_AUC.append(0)
        bibd_ACC.append(0)
        bibd_iter_time.append(0.0)
        print(f"File {file_name} not found.")
    except IndexError:
        bibd_AUC.append(0)
        bibd_ACC.append(0)
        bibd_iter_time.append(0.0)
        print(f"File {file_name} does not have enough lines.")
    except Exception as e:
        bibd_AUC.append(0)
        bibd_ACC.append(0)
        bibd_iter_time.append(0.0)
        print(f"An error occurred while processing {file_name}: {str(e)}")

    # file_name = f"spg_multiple/SPG_{i}.txt"
    file_name = input_dir + "/" + f"SPG_{i}.txt"
    try:
        with open(file_name, 'r') as file:
            lines = file.readlines()
            
            # Extract the 3rd last and 2nd last lines
            third_last_line = lines[-2].strip()
            second_last_line = lines[-1].strip()

            # Regex patterns to extract AUC, ACC, and iter_time
            auc_acc_pattern = r'AUC\s*=\s*(\d+\.\d+),\s*ACC\s*=\s*(\d+\.\d+)'
            iter_time_pattern = r'avg iter_time:\s*(\d+\.\d+)'

            # Search for the AUC and ACC values in the 3rd last line
            auc_acc_match = re.search(auc_acc_pattern, third_last_line)
            if auc_acc_match:
                spg_AUC.append(float(auc_acc_match.group(1)))
                spg_ACC.append(float(auc_acc_match.group(2)))
            else:   # not found
                error_run.append(i)
                spg_AUC.append(0)
                spg_ACC.append(0)
                print(f"AUC and ACC not found in {file_name}")

            # Search for the iter_time in the 2nd last line
            iter_time_match = re.search(iter_time_pattern, second_last_line)
            if iter_time_match:
                spg_iter_time.append(float(iter_time_match.group(1)))
            else:
                spg_iter_time.append(0.0)
                print(f"avg_iter_time not found {file_name}")
    
    except FileNotFoundError:
        spg_AUC.append(0)
        spg_ACC.append(0)
        spg_iter_time.append(0.0)
        print(f"File {file_name} not found.")
    except IndexError:
        spg_AUC.append(0)
        spg_ACC.append(0)
        spg_iter_time.append(0.0)
        print(f"File {file_name} does not have enough lines.")
    except Exception as e:
        spg_AUC.append(0)
        spg_ACC.append(0)
        spg_iter_time.append(0.0)
        print(f"An error occurred while processing {file_name}: {str(e)}")


    # file_name = f"naive_multiple/NAIVE_{i}.txt"
    file_name = input_dir + "/" + f"NAIVE_{i}.txt"
    try:
        with open(file_name, 'r') as file:
            lines = file.readlines()
            
            # Extract the 3rd last and 2nd last lines
            third_last_line = lines[-2].strip()
            second_last_line = lines[-1].strip()

            # Regex patterns to extract AUC, ACC, and iter_time
            auc_acc_pattern = r'AUC\s*=\s*(\d+\.\d+),\s*ACC\s*=\s*(\d+\.\d+)'
            iter_time_pattern = r'avg iter_time:\s*(\d+\.\d+)'

            # Search for the AUC and ACC values in the 3rd last line
            auc_acc_match = re.search(auc_acc_pattern, third_last_line)
            if auc_acc_match:
                naive_AUC.append(float(auc_acc_match.group(1)))
                naive_ACC.append(float(auc_acc_match.group(2)))
            else:   # not found
                error_run.append(i)
                naive_AUC.append(0)
                naive_ACC.append(0)
                print(f"AUC and ACC not found in {file_name}")

            # Search for the iter_time in the 2nd last line
            iter_time_match = re.search(iter_time_pattern, second_last_line)
            if iter_time_match:
                naive_iter_time.append(float(iter_time_match.group(1)))
            else:
                naive_iter_time.append(0.0)
                print(f"avg_iter_time not found {file_name}")
    
    except FileNotFoundError:
        naive_AUC.append(0)
        naive_ACC.append(0)
        naive_iter_time.append(0.0)
        print(f"File {file_name} not found.")
    except IndexError:
        naive_AUC.append(0)
        naive_ACC.append(0)
        naive_iter_time.append(0.0)
        print(f"File {file_name} does not have enough lines.")
    except Exception as e:
        naive_AUC.append(0)
        naive_ACC.append(0)
        naive_iter_time.append(0.0)
        print(f"An error occurred while processing {file_name}: {str(e)}")


# Plotting the values
plt.figure(figsize=(12, 6))

# Plot AUC values
plt.plot(counter, naive_AUC, label='naive', marker='o')

# Plot AUC values
plt.plot(counter, bibd_AUC, label='BIBD', marker='^')

# Plot ACC values
plt.plot(counter, spg_AUC, label='SPG', marker='s')


# Adding titles and labels
plt.title('AUC results per round')
plt.xlabel('Round')
plt.ylabel('Values')

# Adding a legend
plt.legend()

# Show the plot
plt.grid(True)
plt.show()

##################################################################################

# Plotting the values
plt.figure(figsize=(12, 6))

# Plot AUC values
plt.plot(counter, naive_ACC, label='naive', marker='o')

# Plot AUC values
plt.plot(counter, bibd_ACC, label='BIBD', marker='^')

# Plot ACC values
plt.plot(counter, spg_ACC, label='SPG', marker='s')


# Adding titles and labels
plt.title('ACC results per round')
plt.xlabel('Round')
plt.ylabel('Values')

# Adding a legend
plt.legend()

# Show the plot
plt.grid(True)
plt.show()


#################################################################################

# Plotting the values
plt.figure(figsize=(12, 6))

# Plot AUC values
plt.plot(counter, naive_iter_time, label='naive', marker='o')

# Plot AUC values
plt.plot(counter, bibd_iter_time, label='BIBD', marker='^')

# Plot ACC values
plt.plot(counter, spg_iter_time, label='SPG', marker='s')


# Adding titles and labels
plt.title('avg_itertime results per round')
plt.xlabel('Round')
plt.ylabel('Values')

# Adding a legend
plt.legend()

# Show the plot
plt.grid(True)
plt.show()