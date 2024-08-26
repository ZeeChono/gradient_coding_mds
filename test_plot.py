## Plot the metrics value of one type of test
import re
import matplotlib.pyplot as plt
import numpy as np

TEST_NAME = "NAIVE"

# Initialize lists to store the extracted values
AUC_values = []
ACC_values = []
iter_time_values = []

error_run = []


# Counter list to keep track of the file indices
counter = []

# Iterate through the files BIBD_1.txt to BIBD_75.txt
for i in range(1, 101):
    file_name = f"log0901/{TEST_NAME}_{i}.txt"
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
                AUC_values.append(float(auc_acc_match.group(1)))
                ACC_values.append(float(auc_acc_match.group(2)))
            else:   # not found
                error_run.append(i)
                AUC_values.append(0)
                ACC_values.append(0)
                print(f"AUC and ACC not found in {file_name}")

            # Search for the iter_time in the 2nd last line
            iter_time_match = re.search(iter_time_pattern, second_last_line)
            if iter_time_match:
                iter_time_values.append(float(iter_time_match.group(1)))
            else:
                iter_time_values.append(0.0)
                print(f"avg_iter_time not found {file_name}")
    
    except FileNotFoundError:
        AUC_values.append(0)
        ACC_values.append(0)
        iter_time_values.append(0.0)
        print(f"File {file_name} not found.")
    except IndexError:
        AUC_values.append(0)
        ACC_values.append(0)
        iter_time_values.append(0.0)
        print(f"File {file_name} does not have enough lines.")
    except Exception as e:
        AUC_values.append(0)
        ACC_values.append(0)
        iter_time_values.append(0.0)
        print(f"An error occurred while processing {file_name}: {str(e)}")


# Filter out the zero values from the iteration time list
filtered_iter_time_values = [time for time in iter_time_values if time > 0]

# Calculate the average iteration time excluding zeros
avg_iter_time = np.mean(filtered_iter_time_values)

# Plotting the values
plt.figure(figsize=(14, 6))

# Plot AUC values
plt.plot(counter, AUC_values, label='AUC', marker='o')

# Plot ACC values
plt.plot(counter, ACC_values, label='ACC', marker='s')

# Plot Iteration Time values
plt.plot(counter, iter_time_values, label='Avg Iteration Time', marker='^')

# Plot horizontal line of the average iteration time excluding zeros
plt.axhline(y=avg_iter_time, color='g', linestyle='--', label=f'Avg Iter Time = {avg_iter_time:.2f}')

# Adding titles and labels
plt.title(f'{TEST_NAME}: AUC, ACC, and Iteration Time vs. File Index')
plt.xlabel('Round')
plt.ylabel('Values')

# Annotate points where iteration time > threshold
threshold = np.max(iter_time_values) * 0.8
print(threshold)
for i, iter_time in enumerate(iter_time_values):
    if iter_time > threshold:
        plt.text(counter[i], iter_time, f'{iter_time:.2f}', fontsize=9, ha='center', va='bottom')

# Adding a legend
plt.legend()

# Show the plot
plt.grid(True)
plt.show()
