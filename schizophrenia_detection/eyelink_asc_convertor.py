# _*_ coding: utf-8 _*_
# Author: GC Zhu
# Email: zhugc2016@gmail.com

# _*_ coding: utf-8 _*_
# Author: GC Zhu
# Email: zhugc2016@gmail.com
import argparse
import glob
import os
import subprocess

# Create argument parser
parser = argparse.ArgumentParser(description="Script to process dataset directory.")

# Add arguments
parser.add_argument(
    "--dataset_dir",
    type=str,
    default='dataset',
    help="Path to the dataset directory."
)

parser.add_argument(
    '--edf2asc_path',
    type=str,
    default='C:\\Program Files (x86)\\SR Research\\EyeLink\\bin\\64',
    help="Path to the EyeLink edf2asc binary."
)

# # Parse arguments
# args = parser.parse_args()
#
# # Get argument values
# dataset_dir = args.dataset_dir

# Get the current script directory
current_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Current script directory: {current_dir}")

# Construct file matching pattern
glob_file_pattern = os.path.join(current_dir, "data_eyelink_edf", "*", "*.edf")
print(f"File matching pattern: {glob_file_pattern}")

# Find all matching EDF files
files = glob.glob(glob_file_pattern)
if not files:
    print("No matching EDF files found.")
    exit(1)

# Directory where the edf2asc tool is located
target_dir = "C:\\Program Files (x86)\\SR Research\\EyeLink\\bin\\64"
if not os.path.exists(target_dir):
    print(f"Target directory does not exist: {target_dir}")
    exit(1)

# Path to the external edf2asc tool
edf2asc_exe = os.path.join(target_dir, "edf2asc64.exe")

# Verify if the edf2asc tool exists
if not os.path.exists(edf2asc_exe):
    print(f"edf2asc tool not found: {edf2asc_exe}")
    exit(1)

os.makedirs("data_eyelink_asc", exist_ok=True)

# Iterate through EDF files and convert them
for edf_file in files:
    # Generate the corresponding ASC file directory
    asc_dir = os.path.split(edf_file.replace("edf", "asc"))[0]
    os.makedirs(asc_dir, exist_ok=True)  # Ensure the directory exists

    # Construct the full command
    command = [
        "edf2asc",  # edf2asc executable
        "-p",  # Specify output directory parameter
        asc_dir,  # ASC file directory
        "-y",
        # "-utf8",
        "-res",
        "-vel",
        edf_file  # Input EDF file
    ]

    print(f"Executing command: {' '.join(command)}")
    # Execute the command
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print(f"Successfully converted: {edf_file}")
        # print(f"Tool output: {result.stdout}")
    except Exception as e:
        print(f"Exception occurred: {e}")

print("Conversion completed.")

# Add `MSG	21932607 movement_onset` in ps_209_1.asc at line 4162
# Add `MSG	21585592 movement_onset` in ps_201_1.asc at line 3960
# Add `MSG	3313283 movement_onset` in ps_204_1.asc at line 2387
# Add `MSG	6074074 movement_onset` in ps_206_1.asc at line 12578
# Add `MSG	969890 movement_onset` in ps_222_1.asc at line 521
# Add `MSG	19302923 movement_onset` in ps_224_1.asc at line 579
# Add `MSG	20544902 movement_onset` in ps_225_1.asc at line 1570
# Add `MSG	1092052 movement_onset` in ps_228_1.asc at line 547

