import os
import socket
import time
import lightning as pl
from model import Unet3DLightning
from data import Unet3DDataModule
from arguments import PARSER


def configure_master_addr():
    """Get coordinator IP Address with retries"""
    coordinator_address = ""
    coordinator_ip_address = ""
    if os.environ.get("COORDINATOR_ADDRESS") is not None:
        coordinator_address = os.environ.get("COORDINATOR_ADDRESS")
        coordinator_found = False
        lookup_attempt = 1
        max_coordinator_lookups = 50
        while not coordinator_found and lookup_attempt <= max_coordinator_lookups:
            try:
                coordinator_ip_address = socket.gethostbyname(
                    coordinator_address)
                coordinator_found = True
            except socket.gaierror:
                print(
                    f"Failed to recognize coordinator address {coordinator_address} on"
                    f" attempt {lookup_attempt}, retrying...")
                lookup_attempt += 1
                time.sleep(5)
    print(f"Coordinator IP address: {coordinator_ip_address}")
    os.environ["MASTER_ADDR"] = str(coordinator_ip_address)


def init_processes():
    """Initializes the distributed environment."""
    # Get the necessary environment variables from the GKE environment
    job_index = int(os.environ.get("JOB_INDEX"))
    job_completion_index = int(os.environ.get("JOB_COMPLETION_INDEX"))
    processes_in_job = int(os.environ.get("PROCESSES_IN_JOB"))
    rank = job_index * processes_in_job + job_completion_index
    os.environ["NODE_RANK"] = str(rank)

    configure_master_addr()


if __name__ == "__main__":
    flags = PARSER.parse_args()
    if not flags.local:
        init_processes()
    profiler = None
    if flags.benchmark:
        profiler = "simple"

    listing_start = time.time()
    train_data_loader = Unet3DDataModule(flags)
    listing_end = time.time()
    if flags.listing_only:
        print(
            f"Skipping training because you've set listing_only to True\nListing took {listing_end - listing_start} seconds.")
        exit(0)

    model = Unet3DLightning(flags)
    trainer = pl.Trainer(
        accelerator=flags.accelerator,
        max_epochs=flags.epochs,
        devices=flags.num_devices,
        num_nodes=flags.num_nodes,
        strategy="ddp",
        profiler=profiler,
    )
    trainer.fit(model=model, train_dataloaders=train_data_loader)
