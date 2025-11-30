import os
import ast
import pprint

base_config_nums = [2]

EX = 2

output_dir_name = f'config_uniform_sbdv_EX{EX}'
if not os.path.exists(output_dir_name):
    os.makedirs(output_dir_name)

summary = []
config_index = 0

subdiv_range = range(18, 23)

for k in subdiv_range:
    for ex_index in range(0, 10):
        for base in base_config_nums:
            base_config_path = f'new_MG_configs/config_{base}.txt'
            with open(base_config_path, 'r') as f:
                base_config_str = f.read()
                base_config = ast.literal_eval(base_config_str)

            subdiv_init = k
            subdiv_min = k
            subdiv_max = k + 2
            new_config = base_config.copy()

            new_config['subdiv_init'] = subdiv_init
            new_config['subdiv_min'] = subdiv_min
            new_config['subdiv_max'] = subdiv_max

            b_base_output_dir = base_config['base_output_dir']
            num_pts = base_config['num_pts']
            new_config['scaler_dir'] = os.path.join(b_base_output_dir, f'{num_pts}/scalers')

            hidden_shape = base_config['hidden_shape']

            b_output_dir = f"output/Leslie/23.5_23.5/{num_pts}_pts_{hidden_shape}_hshape/{ex_index}"
            new_config['model_dir'] = os.path.join(b_output_dir, 'models')
            
            new_output_dir = b_output_dir = f"output/Leslie/23.5_23.5/EX{EX}/{num_pts}_pts_{hidden_shape}_width_{subdiv_init}_subdiv/{ex_index}"
            new_config['output_dir'] = new_output_dir
            base_output_dir = f"output/Leslie/23.5_23.5/EX{EX}/"
            new_config['base_output_dir'] = base_output_dir

            config_filename = f'config_{config_index}.txt'
            config_filepath = os.path.join(output_dir_name, config_filename)

            new_config['ex_index'] = ex_index

            with open(config_filepath, 'w') as f:
                f.write(pprint.pformat(new_config))

            # Add the details to our summary list
            summary.append(f'{config_filename}: num_pts={num_pts}, hidden_shape={hidden_shape}, subdiv_init={subdiv_init}, ex_index={ex_index}')

            config_index += 1

# base_config_path = f'base_config/Leslie.txt'
# with open(base_config_path, 'r') as f:
#     base_config_str = f.read()
#     # Using ast.literal_eval is safer than eval() for converting string dict to dict
#     base_config = ast.literal_eval(base_config_str)

# for k in k_range:
#     num_pts = (2**k) * 20
#     for j in j_range:
#         hidden_shape = 2**j
#         for s in subdiv_range:
#             for ex_index in ex_index_range:
#                 # Create a copy of the base config to modify
#                 new_config = base_config.copy()

#                 # Update the parameters
#                 new_config['num_pts'] = num_pts
#                 new_config['hidden_shape'] = hidden_shape
#                 new_config['ex_index'] = ex_index

#                 '''
#                 subdiv_min = 24
#                 subdiv_max = 25
#                 subdiv_init = 23
#                 '''

#                 new_config['subdiv_min'] = s #23
#                 new_config['subdiv_max'] = s + 2#23
#                 new_config['subdiv_init'] = s #23

#                 # Construct the output directory path
#             # output_dir = f"output/Leslie/23.5_23.5/20_iterations/{num_pts}_pts_{hidden_shape}_hshape_less_subdivision/{ex_index}"
#                 # In the base config the key is 'base_output_dir', let's create a new key 'output_dir'
#                 # or you can overwrite 'base_output_dir' if you prefer.
#                 # For this example, I'll add a new key 'output_dir' and keep 'base_output_dir'.
#             #   new_config['output_dir'] = output_dir

#                 b_output_dir = base_config['output_dir']
#                 b_base_output_dir = base_config['base_output_dir']

#                 output_dir = os.path.join(base_config["output_dir"], f'{num_pts}', f'h{hidden_shape}', f's{s}', f'{ex_index}/')
#                 new_config['output_dir'] = output_dir

#                 model_dir = os.path.join(output_dir, 'models')
#                 new_config['model_dir'] = model_dir

#             #  num_pts = base_config['num_pts']
#                 new_config['scaler_dir'] = os.path.join(b_base_output_dir, f'{num_pts}/scalers')

#                 new_config['base_output_dir'] = base_config['base_output_dir']

#                 # Define the filename for the new config file
#                 config_filename = f'config_{config_index}.txt'
#                 config_filepath = os.path.join(output_dir_name, config_filename)

#                 # Write the new configuration to the file
#                 with open(config_filepath, 'w') as f:
#                     # Use pprint to format the dictionary for better readability
#                     f.write(pprint.pformat(new_config))

#                 # Add the details to our summary list
#                 summary.append(f'{config_filename}: num_pts={num_pts}, hidden_shape={hidden_shape}, ex_index={ex_index}, s={s}, output={output_dir}')

#                 config_index += 1

# Write the summary to a text file
summary_filename = output_dir_name + '/config_summary.txt'
with open(summary_filename, 'w') as f:
    f.write('\n'.join(summary))

print(f"Successfully generated {config_index} configuration files in '{output_dir_name}'.")
print(f"A summary has been saved to '{summary_filename}'.")
