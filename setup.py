import setuptools

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()


REQUIREMENTS = [
    "matplotlib", 
    "numpy"
    "matplotlib", 
    "numpy"
    ]


CLASSIFIERS = [
    # see https://pypi.org/classifiers/
    "Intended Audience :: Science/Research",
    "Programming Language :: Python :: 3",
    "Operating System :: OS Independent",
    ]

setuptools.setup(
    name="htb",
    version="1.0.0",
    author=["Davide", "Alex", "Alex", "Maro", "Cezary"],
    description="An image upscaling program with land classification.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="module_name"),
    classifiers=CLASSIFIERS,
    python_requires="==3.10",
    install_requires=REQUIREMENTS
)