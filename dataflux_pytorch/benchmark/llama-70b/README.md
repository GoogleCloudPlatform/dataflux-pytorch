# Load Benchmark for Llama-70b

This benchmark is designed to give an approximation of load times for the [Llama-70b](https://huggingface.co/meta-llama/Llama-2-70b-hf/tree/main) model. This benchmark does not account for boot time of the virtual machine, or time to load the model into VRAM. This is a pure load of the model into memory from a GCS bucket. For this benchmark, the model is saved in 15 distinct `.bin` files. This benchmark iterates sequentially through each file, calling `torch.load()` on the dataflux checkpoint reader.

## Benchmark Results

This benchmark was performed on an [n2-standard-64](https://cloud.google.com/compute/docs/general-purpose-machines#n2_machine_types) VM with 256GB of memory in a single-node configuration. The bucket under test was co-located with the VM in the us-central1 region.

| Run | Time to Load Model (seconds) |
| --- | --- |
|  1  | 651 |
|  2  | 646 |
|  3  | 652 |
|  4  | 712 |
|  5  | 705 |
|  6  | 682 |
|  7  | 721 |
|  8  | 703 |

Average across 8 benchmark runs: 684 seconds
