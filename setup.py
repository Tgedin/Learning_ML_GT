from setuptools import find_packages, setup

setup(
    name="ml_learning",
    version="0.1.0",
    description="Personal ML learning journey",
    author="Theo Gedin",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
)
