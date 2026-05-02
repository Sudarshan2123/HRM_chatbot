from setuptools import find_packages,setup
from typing import List
HIPHEN_E_DOT='-e .'
def get_reuquirements(file_path:str)-> List[str]:
    requirements=[]
    with open(file_path, 'r') as file:
        requirements=file.readlines()
        requirements=[req.replace("\n","") for req in requirements ]
        if HIPHEN_E_DOT in requirements:
            requirements.remove(HIPHEN_E_DOT)
    return requirements
    
setup(
    name='Agentic-Macom',
    version='0.0.1',
    author='Sudarshan M Patil,Anika Shaji',
    author_email='sudarshan.patil@mactech.net.in,anika.shaji@mactech.net.in',
    description='Gen AI project',
    packages=find_packages(),
    install_requires=get_reuquirements('requirements.txt'),
    
)