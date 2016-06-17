from setuptools import setup
from setuptools import find_packages


setup(name='NTM-Lasagne',
    version='0.2.0',
    description='Neural Turing Machines in Theano with Lasagne',
    author='Tristan Deleu',
    author_email='tristan.deleu@snips.ai',
    url='',
    download_url='',
    license='MIT',
    install_requires=[
        'numpy',
        'theano',
        'Lasagne'],
    packages=['ntm'],
    include_package_data=False,
    zip_safe=False)