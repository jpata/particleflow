import os
import setuptools


this_dir = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_dir, "README.md"), "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as req_file:
    REQUIRED = req_file.readlines()

DEV_REQUIRE = ["pytest"]

setuptools.setup(
    name="heptfds",
    version="0.0.1",
    author="Eric Wulff",
    author_email="eric.wulff@cern.ch",
    description="Repository for storing custom HEP datasets",
    install_requires=REQUIRED,
    extras_require={"dev": DEV_REQUIRE},
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
