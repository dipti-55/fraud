# used for creating entire packages
from setuptools import setup, find_packages
from typing import List 

HYPEN_E_DOT = '-e .'

def get_requirements(file_path:str) -> List[str]:
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

        return requirements



setup(
    name='fraud-detection',
    version='0.0.1',
    author='Dipti Singh',
    author_email='dipti1010.singh@gmail.com',
    install_requires=get_requirements("requirements.txt"),
    packages=find_packages(),
)