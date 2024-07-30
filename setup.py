from setuptools import setup, find_packages

dependencies = [
    "torch",
    "google-cloud-storage",
    "absl-py",
    "numpy",
    "pytest",
    "lightning",
]
setup(
    name="dataflux-pytorch",
    packages=find_packages(exclude=["*tests*"]),
    install_requires=dependencies,
)
