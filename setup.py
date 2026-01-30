from setuptools import setup, find_packages

setup(
    name="world-models-reproducibility",
    version="0.1.0",
    # find_packages() automatically finds the 'src' folder 
    # and treats it as a package
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "matplotlib",
        "tqdm",
    ],
)