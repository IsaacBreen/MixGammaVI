import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mix_gamma_vi", # Replace with your own username
    version="1.0.12",
    author="Isaac Breen",
    author_email="isaac.breen@icloud.com",
    description="A small example package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/IsaacBreen/MixGammaVI",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'tensorflow>=2',
        'tensorflow_probability>=0.8',
        'numpy'
    ],
)
