"""Helpers for resolving paths in YAML config relative to the project root."""
import os


def normalize_output_dir(config, config_path):
    """
    Make output.dir absolute, anchored to the project root (parent of config/).

    Config values like ./outputs_medium are relative to the bmeaiproj1 folder,
    not the process current working directory.
    """
    if 'output' not in config or 'dir' not in config['output']:
        return
    out = config['output']['dir']
    if not out or os.path.isabs(out):
        return
    config_dir = os.path.dirname(os.path.abspath(config_path))
    project_root = os.path.dirname(config_dir)
    config['output']['dir'] = os.path.normpath(os.path.join(project_root, out))
