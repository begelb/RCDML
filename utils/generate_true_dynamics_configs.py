import os
import ast
import pprint

output_dir_name = 'part_c_ten_trials'
if not os.path.exists(output_dir_name):
    os.makedirs(output_dir_name)

summary = []
config_index = 0

trial_range = range(0, 10)
#subdiv_range = range(10, 40)

base_config_path = 'new_MG_configs/config_new.txt'
with open(base_config_path, 'r') as f:
    base_config_str = f.read()
    base_config = ast.literal_eval(base_config_str)

#for k in subdiv_range:
for trial in trial_range:
    subdiv_init = base_config["subdiv_init"]
    subdiv_min = base_config["subdiv_min"]#k
    subdiv_max = base_config["subdiv_max"]#k + 2
    new_config = base_config.copy()

    new_config['subdiv_init'] = subdiv_init
    new_config['subdiv_min'] = subdiv_min
    new_config['subdiv_max'] = subdiv_max
    new_config['ex_index'] = trial

    output_dir = os.path.join(base_config["output_dir"], f'{config_index}/')
    new_config['output_dir'] = output_dir

    model_dir = os.path.join(output_dir, 'models')
    new_config['model_dir'] = model_dir

    config_filename = f'config_{config_index}.txt'
    config_filepath = os.path.join(output_dir_name, config_filename)

    with open(config_filepath, 'w') as f:
        f.write(pprint.pformat(new_config))

    summary.append(f'{config_filename}: subdiv_init={subdiv_init}')

    config_index += 1

summary_filename = output_dir_name + '/config_summary.txt'
with open(summary_filename, 'w') as f:
    f.write('\n'.join(summary))

print(f"Successfully generated {config_index} configuration files in '{output_dir_name}'.")
print(f"A summary has been saved to '{summary_filename}'.")
