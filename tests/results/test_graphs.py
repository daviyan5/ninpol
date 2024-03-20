import numpy as np


def graph_performance():
    import os
    import yaml
    import matplotlib.pyplot as plt
    import re

    # Get the file path
    file_path = os.path.join(os.path.dirname(__file__), "performance_test.yaml")

    # Load YAML file
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)

    # Group data by element_type
    grouped_data = {}
    for filename, values in data['files'].items():
        if "sphere" in filename:
            continue
        
        element_type = re.match(r'([a-zA-Z]+)', filename).group(1)

        if element_type not in grouped_data:
            grouped_data[element_type] = {'n_points': [], 'time_to_build': [], 'time_to_interpolate': []}

        grouped_data[element_type]['n_points'].append(values['n_points'])
        grouped_data[element_type]['time_to_build'].append(values['time_to_build'])
        grouped_data[element_type]['time_to_interpolate'].append(values['time_to_interpolate'])

    # Plotting
    num_element_types = len(grouped_data)
    num_rows = (num_element_types + 1) // 2  # Calculate number of rows for subplots
    fig, axes = plt.subplots(num_rows, 2, figsize=(15, 6 * num_rows), squeeze=False)

    for i, (element_type, values) in enumerate(grouped_data.items()):
        row = i // 2
        col = i % 2
        ax = axes[row, col]

        ax.set_title(f'Time to Build and Interpolate for "{element_type}" meshes')
        ax.set_xlabel('Number of Points')
        ax.set_ylabel('Time (seconds)')

        ax.plot(values['n_points'], values['time_to_build'], 'o-', label='Time to Build')
        ax.plot(values['n_points'], values['time_to_interpolate'], 'o-', label='Time to Interpolate')

        ax.legend()
        ax.grid(True)


    plt.tight_layout()

    save_path = os.path.join(os.path.dirname(__file__), "performance_test.png")
    plt.savefig(save_path)

    # Do only interpolation
    num_element_types = len(grouped_data)
    num_rows = (num_element_types + 1) // 2  # Calculate number of rows for subplots
    fig, axes = plt.subplots(num_rows, 2, figsize=(15, 6 * num_rows), squeeze=False)

    for i, (element_type, values) in enumerate(grouped_data.items()):
        row = i // 2
        col = i % 2
        ax = axes[row, col]

        ax.set_title(f'Time to Interpolate for "{element_type}" meshes')
        ax.set_xlabel('Number of Points')
        ax.set_ylabel('Time (seconds)')

        ax.plot(values['n_points'], values['time_to_interpolate'], 'o-', label='Time to Interpolate')

        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(__file__), "performance_test_interpolation.png")
    plt.savefig(save_path)

def graph_accuracy():
    import os
    import re
    import yaml
    import matplotlib.pyplot as plt

    # Get the file path
    file_path = os.path.join(os.path.dirname(__file__), "accuracy_test.yaml")

    # Load YAML file
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)

    # Group elements by type
    grouped_data = {}
    for filename, values in data['files'].items():
        element_type = re.match(r'([a-zA-Z]+)', filename).group(1)

        if "sphere" in filename:
            continue
        if element_type not in grouped_data:
            grouped_data[element_type] = {'errors': {}, 'n_points': []}
        
        grouped_data[element_type]['n_points'].append(values['n_points'])

        for method_name, method in values['methods'].items():
            if method_name not in grouped_data[element_type]['errors']:
                grouped_data[element_type]['errors'][method_name] = {}
            
            for variable in method:
                # Strip 'error_' from the beginning of the variable name
                new_variable = re.sub(r'^error_', '', variable)
                if new_variable not in grouped_data[element_type]['errors'][method_name]:
                    grouped_data[element_type]['errors'][method_name][new_variable] = []
                
                grouped_data[element_type]['errors'][method_name][new_variable].append(method[variable])

    # Determine number of rows
    num_element_types = len(grouped_data)
    num_rows = (num_element_types + 1) // 2  # Add 1 to ensure rounding up

    # Plotting
    fig, axes = plt.subplots(num_rows, 2, figsize=(15, 6 * num_rows), squeeze=False)

    for i, (element_type, values) in enumerate(grouped_data.items()):
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        ax.set_title(f'Error for "{element_type}" meshes')
        ax.set_xlabel('Number of Points')
        ax.set_ylabel('L2rel')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.axis('equal')

        ax.grid(True)
        
        minn = np.inf
        minerr = np.inf
        for method, errors in values['errors'].items():
            for variable, error_list in errors.items():
                ax.plot(values['n_points'], error_list, 'o-', label=f'{method}_{variable}')
                minn   = min(minn, np.min(values['n_points']))
                minerr = min(minerr, np.min(error_list))
                    
        ax.legend()

    # Remove any unused subplots
    for i in range(num_element_types, num_rows * 2):
        row = i // 2
        col = i % 2
        fig.delaxes(axes[row, col])

    plt.tight_layout()
    # Save the plot
    save_path = os.path.join(os.path.dirname(__file__), "accuracy_test.png")
    plt.savefig(save_path)


if __name__ == "__main__":
    graph_performance()
    graph_accuracy()