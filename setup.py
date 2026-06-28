from setuptools import setup, find_packages

setup(
    name="scandium-labs",
    version="1.0.0",
    description="AI-Driven Solid Electrolyte Discovery Platform",
    author="Scandium Labs",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.3.0",
        "torch-geometric>=2.5.0",
        "pymatgen>=2024.4.13",
        "fastapi>=0.111.0",
        "pydantic>=2.7.0",
        "pandas>=2.2.2",
        "numpy>=1.26.4",
        "scikit-learn>=1.4.2",
    ],
)
