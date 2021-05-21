from setuptools import find_packages, setup

# with open("README.md", "r") as fh:
#     long_description = fh.read()

setup(
    name="smg-rotorsim",
    version="0.0.1",
    author="Stuart Golodetz",
    author_email="stuart.golodetz@cs.ox.ac.uk",
    description="A simple drone simulator",
    long_description="",  #long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sgolodetz/smg-rotorsim",
    packages=find_packages(include=["smg.rotorsim", "smg.rotorsim.*"]),
    include_package_data=True,
    install_requires=[
        "numpy",
        "open3d",
        "pygame",
        "PyOpenGL",
        "smg-joysticks",
        "smg-navigation",
        "smg-open3d",
        "smg-opengl",
        "smg-pyoctomap",
        "smg-rigging",
        "smg-rotory",
        "smg-utility"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
