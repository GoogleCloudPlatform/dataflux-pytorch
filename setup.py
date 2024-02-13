from setuptools import setup

dependencies = [
    "torch",
    "google-cloud-storage",
    "absl-py",
    "numpy",
    "pytest",
]
setup(
    name="dataflux-pytorch",
    packages=["dataflux_core", "dataflux_pytorch"],
    package_dir={"dataflux_core": "dataflux_client_python/dataflux_core"},
    install_requires=dependencies,
)
