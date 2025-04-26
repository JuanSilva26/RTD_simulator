from setuptools import setup, find_packages

setup(
    name="rtd_simulator",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "torch>=1.9.0",
        "matplotlib>=3.4.0",
        "scipy>=1.7.0",
    ],
    python_requires=">=3.8",
) 