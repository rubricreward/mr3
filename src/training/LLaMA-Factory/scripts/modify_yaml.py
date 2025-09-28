import yaml
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference on R3 Evaluation Dataset(s)')
    parser.add_argument('--yaml_file_path', '-c', type=str, required=True,
                        help=f"YAML file path.")
    parser.add_argument('--learning_rate', '-l', type=float, required=True,
                        help="Learning rate.")
    parser.add_argument('--pref_beta', '-b', type=float, required=True,
                        help="Pref beta.")
    parser.add_argument('--output_dir', '-o', type=str, required=True,
                        help="Output dir.")
    args = parser.parse_args()

    # Define the file path
    file_path = args.yaml_file_path

    # Open and read the YAML file
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)

    data['learning_rate'] = args.learning_rate
    data['pref_beta'] = args.pref_beta
    data['output_dir'] = args.output_dir

    # Open the file in write mode and dump the modified data back
    with open(file_path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)
