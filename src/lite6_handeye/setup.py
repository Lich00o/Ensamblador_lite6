import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'lite6_handeye'


def _clean_broken_symlinks():
    """Elimina symlinks huérfanos en build/ cuando se borran archivos en src/."""
    for f in glob('**/*', recursive=True):
        if os.path.islink(f) and not os.path.exists(f):
            os.unlink(f)


def sim_files():
    """Instala sim/ (mundos y modelos de Gazebo) conservando la estructura."""
    out = {}
    for f in glob('sim/**/*', recursive=True):
        if os.path.isfile(f) and not f.startswith('sim/models/person_target/meshes/'):
            out.setdefault(os.path.join('share', package_name, os.path.dirname(f)), []).append(f)
    return list(out.items())


_clean_broken_symlinks()

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', [f for f in glob('launch/*.py') if os.path.isfile(f)]),
        ('share/' + package_name + '/config', [f for f in glob('config/*') if os.path.isfile(f)]),
    ] + sim_files(),
    install_requires=['setuptools'],
    extras_require={'test': ['pytest']},
    zip_safe=True,
    maintainer='licho01',
    maintainer_email='galindbenjamin@gmail.com',
    description='Calibración Eye-in-Hand Lite 6 + ZED/Orbbec + ChArUco y seguimiento 6-DOF de personas.',
    license='BSD-3-Clause',
    entry_points={
        'console_scripts': [
            'charuco_detector = lite6_handeye.charuco_detector_node:main',
            'auto_calibrator = lite6_handeye.auto_calibrator_node:main',
            'person_tracker = lite6_handeye.person_tracker_node:main',
            'orbbec_uvc = lite6_handeye.orbbec_uvc_node:main',
            'import_person_usd = lite6_handeye.usd_person:main',
            'generate_board = lite6_handeye.cli:generate_board',
            'solve_offline = lite6_handeye.cli:solve_offline',
            'compare_gt = lite6_handeye.cli:compare_gt',
        ],
    },
)
