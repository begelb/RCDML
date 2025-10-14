import os
import ast
import pprint

# Read the base configuration from the provided file
with open('base_config/Leslie.txt', 'r') as f:
    base_config_str = f.read()
    # Using ast.literal_eval is safer than eval() for converting string dict to dict
    base_config = ast.literal_eval(base_config_str)

# Create the directory to store the generated config files
output_dir_name = 'data_and_hsize_configs'
if not os.path.exists(output_dir_name):
    os.makedirs(output_dir_name)

# Keep track of all generated file names and their parameters
summary = []
config_index = 0

# Define the ranges for the parameters
k_range = range(1, 11)  # 1 to 10 for num_pts
j_range = range(1, 7)   # 1 to 6 for hidden_shape
ex_index_range = range(0, 11) # 0 to 10 for ex_index

# Iterate through all combinations of the parameters
for k in k_range:
    num_pts = (2**k) * 10
    for j in j_range:
        hidden_shape = 2**j
        for ex_index in ex_index_range:
            # Create a copy of the base config to modify
            new_config = base_config.copy()

            # Update the parameters
            new_config['num_pts'] = num_pts
            new_config['hidden_shape'] = hidden_shape
            new_config['ex_index'] = ex_index

            # Construct the output directory path
            output_dir = f"output/Leslie/23.5_23.5/{num_pts}_pts_{hidden_shape}_hshape/{ex_index}"
            # In the base config the key is 'base_output_dir', let's create a new key 'output_dir'
            # or you can overwrite 'base_output_dir' if you prefer.
            # For this example, I'll add a new key 'output_dir' and keep 'base_output_dir'.
            new_config['output_dir'] = output_dir

            # Define the filename for the new config file
            config_filename = f'config_{config_index}.txt'
            config_filepath = os.path.join(output_dir_name, config_filename)

            # Write the new configuration to the file
            with open(config_filepath, 'w') as f:
                # Use pprint to format the dictionary for better readability
                f.write(pprint.pformat(new_config))

            # Add the details to our summary list
            summary.append(f'{config_filename}: num_pts={num_pts}, hidden_shape={hidden_shape}, ex_index={ex_index}')

            config_index += 1

# Write the summary to a text file
summary_filename = 'config_summary.txt'
with open(summary_filename, 'w') as f:
    f.write('\n'.join(summary))

print(f"Successfully generated {config_index} configuration files in '{output_dir_name}'.")
print(f"A summary has been saved to '{summary_filename}'.")
