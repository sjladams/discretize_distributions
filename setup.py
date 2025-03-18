import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="discretize_distributions",
    version="1.0.1",
    author="Steven Adams",
    author_email="stevenjladams@gmail.com",
    description="Signatures of Probability Distributions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sjladams/discretize_distributions",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.9.10",
    install_requires=[
        'torch>=1.13.1',
        'torch-kmeans>=0.2.0',
        'xitorch>=0.3.0',
        'numpy',
        'stable-trunc-gaussian>=1.3.9',
        'tqdm'
    ],
    package_data = {
        "discretize_distributions": ["data/*.pickle"],  # Include the pickle file
    },
    include_package_data = True,
)