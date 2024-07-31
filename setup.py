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
    packages=find_packages(exclude=["*tests*", "*dataflux_client_python"]),
    package_dir={"dataflux_core": "dataflux_client_python/dataflux_core"},
    install_requires=dependencies,
)
