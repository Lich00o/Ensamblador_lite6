#!/usr/bin/env python3
"""Instala con apt las dependencias declaradas en los package.xml de un workspace.

Reemplaza `rosdep install` cuando la base de datos de rosdep
(raw.githubusercontent.com) no es accesible. Mapeo:
  python3-*            -> mismo nombre en apt
  claves en MANUAL     -> paquete indicado
  resto (paquetes ROS) -> ros-<distro>-<clave con '_' -> '-'>
Uso: install_deps.py <src_dir> [--dry-run]
"""

import glob
import os
import subprocess
import sys
import xml.etree.ElementTree as ET

MANUAL = {
    'python-transforms3d-pip': 'python3-transforms3d',
}
DEP_TAGS = ('depend', 'build_depend', 'exec_depend', 'buildtool_depend',
            'build_export_depend', 'run_depend', 'test_depend')


def collect(src):
    names, deps = set(), set()
    for f in glob.glob(os.path.join(src, '**', 'package.xml'), recursive=True):
        if os.path.exists(os.path.join(os.path.dirname(f), 'COLCON_IGNORE')):
            continue
        root = ET.parse(f).getroot()
        names.add(root.findtext('name').strip())
        for tag in DEP_TAGS:
            deps.update(e.text.strip() for e in root.findall(tag))
    return sorted(deps - names)


def to_apt(key, distro):
    if key in MANUAL:
        return MANUAL[key]
    if key.startswith('python3-'):
        return key
    return f'ros-{distro}-{key.replace("_", "-")}'


def main():
    src = sys.argv[1]
    distro = os.environ.get('ROS_DISTRO', 'jazzy')
    pkgs = sorted({to_apt(k, distro) for k in collect(src)})
    available = subprocess.run(['apt-cache', 'policy', *pkgs], capture_output=True, text=True).stdout
    missing = [p for p in pkgs if f'{p}:\n  Installed:' not in available
               or f'{p}:\n  Installed: (none)\n  Candidate: (none)' in available]
    if missing:
        sys.exit(f'Sin candidato en apt: {missing}')
    print('apt install:', ' '.join(pkgs))
    if '--dry-run' not in sys.argv:
        subprocess.run(['apt-get', 'install', '-y', '--no-install-recommends', *pkgs], check=True)


if __name__ == '__main__':
    main()
