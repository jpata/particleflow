from setuptools import setup, find_packages
import os

# Read requirements.txt
with open("requirements.txt") as f:
    install_requires = [line.strip() for line in f if line.strip() and not line.startswith("#")]

# Read README.md for long description
if os.path.exists("README.md"):
    with open("README.md", encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "Machine Learning for Particle Flow Reconstruction"

setup(
    name="particleflow",
    version="3.0.0",
    author="Joosep Pata, Farouk Mokhtar, Eric Wulff, Javier Duarte",
    author_email="joosep.pata@cern.ch",
    description="Machine Learning for Particle Flow Reconstruction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jpata/particleflow",
    packages=find_packages(exclude=["tests", "tests.*"]),
    include_package_data=True,
    install_requires=install_requires,
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "mlpf = mlpf.pipeline:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
