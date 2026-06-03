from setuptools import setup, find_packages

setup(
    name="personal-brain",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "anthropic>=0.30.0",
        "click>=8.1.0",
        "pyvis>=0.3.2",
        "httpx>=0.27.0",
        "beautifulsoup4>=4.12.0",
    ],
    entry_points={
        "console_scripts": [
            "brain=cli:cli",
        ],
    },
)
