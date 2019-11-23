# -*- coding: utf-8 -*-

# Learn more: https://github.com/kennethreitz/setup.py

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='jargon_distance',
    version='0.1.0',
    description='Calculate jargon distance metric between texts',
    long_description=readme,
    author='Jason Portenoy',
    author_email='jason.portenoy@gmail.com',
    url='https://github.com/h1-the-swan/jargon_distance',
    license=license,
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=['numpy', 'six'],
    include_package_data=True
)

