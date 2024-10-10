from setuptools import setup, find_packages
from typing import List

CONST_TO_IGNORE = "-e ."

def get_packages(path: str) -> List[str]:
    """
    responsible to get and return all libraries in the requirements.txt
    """

    all_libraries = []
    with open(path) as docs:
        all_libraries = docs.readlines()
        all_libraries.remove(CONST_TO_IGNORE)

    all_libraries = [lib.replace("\n", "") for lib in all_libraries]

    return all_libraries


setup(
    name='Mlproject',
    version='0.0.1',
    author='MuriloXavier',
    author_email='xavierdesouzamurilo@gmail.com',
    packages=find_packages(),
    install_requires=get_packages('requirements.txt')
)