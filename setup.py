import json
from setuptools import setup, find_packages


with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="otter-ai",
    version="0.0.0-alpha",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=requirements,
    # package_data={
    #     "otter": ["resources/**/*"],
    # },
    # include_package_data=True,
    author="Bo Li, Yuanhan Zhang, Liangyu Chen, Jinghao Wang, Fanyi Pu, Jingkang Yang, Chunyuan Li, Ziwei Liu",
    author_email="LIBO0013@e.ntu.edu.sg, YUANHAN002@e.ntu.edu.sg, LCHEN025@e.ntu.edu.sg, JINGHAO003@e.ntu.edu.sg, FPU001@e.ntu.edu.sg, chunyl@microsoft.com, ziwei.liu@ntu.edu.sg",
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
