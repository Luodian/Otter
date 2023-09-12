import json
from setuptools import setup, find_packages


with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="otter-ai",
    version="0.0.0-alpha-3",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=requirements,
    # package_data={
    #     "otter": ["resources/**/*"],
    # },
    # include_package_data=True,
    author="Otter Team",
    author_email="drluodian@gmail.com",
    description="Otter: A Multi-Modal Model with In-Context Instruction Tuning",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Luodian/Otter",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    # entry_points={
    #     "console_scripts": [
    #         # "syphus = syphus.cli.syphus_cli:main",
    #     ],
    # },
)
