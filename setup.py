from setuptools import setup

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
    packages=[
        "dataflux_core", "dataflux_pytorch", "dataflux_pytorch.lightning",
        "dataflux_pytorch.multipart_upload", "demo.lightning.text_based",
        "demo.lightning.checkpoint.multinode"
    ],
    package_dir={"dataflux_core": "dataflux_client_python/dataflux_core"},
    install_requires=dependencies,
)
