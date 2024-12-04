from setuptools import setup, find_packages

setup(
    name="extra-attributions",
    version="0.2.0",
    description="Extra attribution methods based on Captum",
    author="Oscar Pellicer",
    author_email="",  # Add if desired
    url="https://github.com/OscarPellicer/extra-attributions",
    packages=find_packages(),
    install_requires=[
        "torch",
        "captum",
        "numpy",
        "tqdm"
    ],
    python_requires=">=3.6",
) 