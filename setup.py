from setuptools import setup, find_packages

setup(
    name="personal-brain",
    version="0.1.0",
    packages=find_packages(),
    py_modules=["cli"],          # cli.py is a top-level module, not in a package
    python_requires=">=3.10",    # uses PEP 604 (list[str] | None) syntax
    install_requires=[
        # LLM calls go through brain/llm.py over stdlib urllib — no SDK needed.
        "click>=8.1.0",
        "pyvis>=0.3.2",
        "networkx>=3.0",
        "httpx>=0.27.0",
        "beautifulsoup4>=4.12.0",
    ],
    entry_points={
        "console_scripts": [
            "brain=cli:cli",
        ],
    },
)
