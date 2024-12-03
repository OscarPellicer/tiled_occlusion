from setuptools import setup, find_packages

setup(
    name="tiled_occlusion",
    version="0.1.0",
    description="Tiled Occlusion attribution method based on Captum",
    author="Oscar Pellicer",
    author_email="",  # Add if desired
    url="https://github.com/OscarPellicer/tiled_occlusion",
    packages=find_packages(),
    install_requires=[
        "torch",
        "captum",
        "numpy",
        "tqdm"
    ],
    python_requires=">=3.6",
) 