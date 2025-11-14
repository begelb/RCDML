import os
import ast
import pprint

# Create the directory to store the generated config files
output_dir_name = 'true_dynamics_configs'
if not os.path.exists(output_dir_name):
    os.makedirs(output_dir_name)

summary = []
config_index = 0

subdiv_range = range(10, 40)

base_config_path = 'true_dynamics_configs/base.txt'
with open(base_config_path, 'r') as f:
    base_config_str = f.read()
    # Using ast.literal_eval is safer than eval() for converting string dict to dict
    base_config = ast.literal_eval(base_config_str)

for k in subdiv_range:
    subdiv_init = k
    subdiv_min = k
    subdiv_max = k + 2
    new_config = base_config.copy()

    new_config['subdiv_init'] = subdiv_init
    new_config['subdiv_min'] = subdiv_min
    new_config['subdiv_max'] = subdiv_max

    output_dir = os.path.join(base_config["output_dir"], f'{config_index}/')
    new_config['output_dir'] = output_dir

    config_filename = f'config_{config_index}.txt'
    config_filepath = os.path.join(output_dir_name, config_filename)

    with open(config_filepath, 'w') as f:
        # Use pprint to format the dictionary for better readability
        f.write(pprint.pformat(new_config))

    # Add the details to our summary list
    summary.append(f'{config_filename}: subdiv_init={subdiv_init}')

    config_index += 1

summary_filename = output_dir_name + '/config_summary.txt'
with open(summary_filename, 'w') as f:
    f.write('\n'.join(summary))

print(f"Successfully generated {config_index} configuration files in '{output_dir_name}'.")
print(f"A summary has been saved to '{summary_filename}'.")
