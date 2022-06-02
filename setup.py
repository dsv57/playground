from os.path import exists
from setuptools import setup

setup(name='python-kanren',
      version='0.2.3',
      description='Python live programming environment',
      url='http://github.com/nekomimi/playground',
      author='Dmitry Ivanov',
      author_email='dmitry@ivanov.com',
      license='BSD',
      packages=['python-playground'],
      install_requires=open('requirements.txt').read().split('\n'),
      long_description=open('README.md').read() if exists("README.md") else "",
      classifiers=["Development Status :: 3 - Alpha",
                   "License :: OSI Approved :: BSD License",
                   "Programming Language :: Python",
                   "Programming Language :: Python :: 3.7",
                   "Programming Language :: Python :: Implementation :: CPython",
      ],
)
