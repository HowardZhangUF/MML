from setuptools import find_packages
from setuptools import setup

setup(
    name='mml_guidance',
    version='0.0.0',
    packages=find_packages(
        include=('mml_guidance', 'mml_guidance.*')),
)
