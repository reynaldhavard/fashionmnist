import yaml


def get_config(config_file):
    """
    Load a YAML file to a dictionary
    """
    with open(config_file) as f:
        config = yaml.safe_load(f)

    return config
