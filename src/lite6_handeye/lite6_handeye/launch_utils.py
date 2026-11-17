"""Helpers para launch files."""

import os

import yaml


def node_params(params_file, node_name, overrides=None):
    """Parámetros de un nodo desde el YAML (/** + sección del nodo) con overrides encima.

    No se pasa el YAML y un dict por separado: rcl aplica el dict como '/**' y la
    sección específica del nodo en el YAML tendría precedencia sobre el override.
    """
    with open(os.path.expanduser(params_file)) as f:
        data = yaml.safe_load(f) or {}
    out = {}
    for key in ('/**', node_name, '/' + node_name):
        out.update((data.get(key) or {}).get('ros__parameters', {}))
    out.update(overrides or {})
    return out
