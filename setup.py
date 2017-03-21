from setuptools import setup
from setuptools import find_packages


setup(name='NTM-Lasagne',
    version='0.3.0',
    description='Neural Turing Machines in Theano with Lasagne',
    author='Tristan Deleu',
    author_email='tristan.deleu@snips.ai',
    url='',
    download_url='',
    license='MIT',
    install_requires=[
        'numpy>=1.12.1',
        'theano==0.9.0'
    ],
    packages=['ntm'],
    include_package_data=False,
    zip_safe=False)