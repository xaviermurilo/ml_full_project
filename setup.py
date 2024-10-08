from setuptools import find_packages, setup
from typing import List


CONST_TO_DEL = '-e .'
def get_requirements(path: str) -> List[str]:
    """
    function return the list of requirements
    """

    requirements = []
    with open(path) as req:
        requirements = req.readlines()

    requirements = [re.replace("\n", "") for re in requirements]

    if CONST_TO_DEL in requirements:
        requirements.remove(CONST_TO_DEL)

    return requirements


setup(
    name='Mlproject',
    version='0.0.1',
    author='MuriloXavier',
    author_email='xavierdesouzamurilo@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)