#!/usr/bin/env python

"""The setup script."""

import os

from setuptools import find_packages, setup


def _read_version():
    init_py = os.path.join(os.path.dirname(__file__), "biolm", "__init__.py")
    with open(init_py, encoding="utf-8") as f:
        for line in f:
            if line.startswith("__version__"):
                return line.split("=", 1)[1].strip().strip("'\"")
    raise RuntimeError("Could not find __version__ in biolm/__init__.py")

with open("README.md", encoding="utf-8") as readme_file:
    readme = readme_file.read()

requirements = [
    "Click>=6.0",
    "requests",
    "httpx>=0.23.0",
    "httpcore",
    "h2",
    "synchronicity>=0.5.0; python_version >= '3.9'",
    "synchronicity<0.5.0; python_version < '3.9'",
    "typing_extensions; python_version < '3.9'",
    "aiodns",
    "aiohttp<=3.8.6; python_version < '3.12'",
    "aiohttp>=3.9.0; python_version >= '3.12'",
    "async-lru",
    "aiofiles",
    "cryptography",
    "nest_asyncio",
    "rich>=13.0.0",
    "pyyaml>=5.0",
    "jsonschema<=4.26.0",
]

test_requirements = [
    "pytest>=3",
]

setup(
    author="BioLM",
    author_email="support@biolm.ai",
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    description="BioLM Python client for the hosted platform and biolm-hub",
    entry_points={
        "console_scripts": [
            "biolm=biolm.cli.entry:cli",
            "biolmai=biolm.cli.entry:cli",
        ],
        "mlflow.request_header_provider": [
            "biolm=biolm.core.seqflow_auth:BiolmRequestHeaderProvider",
            "unused=biolm.core.seqflow_auth:BiolmaiRequestHeaderProvider",
        ],
    },
    install_requires=requirements,
    extras_require={
        "pipeline": [
            "pandas>=1.3.0,<3",
            "numpy>=1.20,<3",
            "tqdm>=4.60.0",
            "matplotlib>=3.3.0",
            "seaborn>=0.11.0",
            "scikit-learn>=1.0",
            "umap-learn>=0.5.0",
            "biotite>=0.34.0",
            "biopython>=1.78",
            "scipy>=1.7.0",
            "duckdb>=0.9.0,<2",
            "pyarrow>=10.0.0",
            "jmespath>=1.0.0",
        ],
    },
    license="Apache Software License 2.0",
    long_description=readme,
    include_package_data=True,
    keywords=["biolm", "bioai", "bio-ai", "bio-lm", "bio-llm"],
    name="biolm-sdk",
    packages=find_packages(include=["biolm", "biolm.*", "biolmai", "biolmai.*"]),
    package_data={"biolm.hub": ["data/*.json"]},
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/BioLM/biolm-sdk",
    version=_read_version(),
    zip_safe=False,
)
