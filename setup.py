from setuptools import setup, find_packages

with open("requirements.txt") as requirement_file:
    requirements = requirement_file.read().split()

setup(
    name='bird_challenge',
    version='0.1.0',
    description='Code for a bird image classification challenge',
    author='Amric Trudel',
    url='https://github.com/atrudel/bird_challenge',
    python_requires='>=3.7',
    install_requires=requirements,
    packages=find_packages()
)