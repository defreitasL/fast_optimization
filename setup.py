from setuptools import setup, find_packages

setup(
    name='fast_optimization',
    version='1.1.19',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy',
        'numba',
    ],
    author='Lucas de Freitas Pereira',
    author_email='lucas.defreitas@unican.es',
    description='IH-SET Calibration package for shoreline evolution models.',
    url='https://github.com/defreitasL/fast_optimization',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)