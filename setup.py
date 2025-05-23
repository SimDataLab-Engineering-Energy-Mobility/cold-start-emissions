from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT = "-e ."
def get_requirements(file_path: str) -> List[str]:
    """
    This function reads the requirements file and returns a list of required packages.
    """
    requirements = []
    with open (file_path) as file:
        requirements = file.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements
    
setup(
name='COLD-START-EMISSIONS',
version='1.0',
author='Manoj_and_Jordan',
author_email='jordan.denev@kit.edu',
packages=find_packages(),
install_requires=get_requirements('requirements.txt'),
description='A package for cold start emissions prediction',
)