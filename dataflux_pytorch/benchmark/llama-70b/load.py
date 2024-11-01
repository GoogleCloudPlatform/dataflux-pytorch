import os
import time

import torch

from dataflux_pytorch import dataflux_checkpoint

start_time = time.time()
project_name = os.getenv("PROJECT")
bucket_name = os.getenv("BUCKET")
ckpt = dataflux_checkpoint.DatafluxCheckpoint(project_name=project_name,
                                              bucket_name=bucket_name)
CKPT_PATH = "llama2/llama2-70b-hf/"
LOAD_COUNT = 10
files = [
    'pytorch_model-00001-of-00015.bin', 'pytorch_model-00002-of-00015.bin',
    'pytorch_model-00003-of-00015.bin', 'pytorch_model-00004-of-00015.bin',
    'pytorch_model-00005-of-00015.bin', 'pytorch_model-00006-of-00015.bin',
    'pytorch_model-00007-of-00015.bin', 'pytorch_model-00008-of-00015.bin',
    'pytorch_model-00009-of-00015.bin', 'pytorch_model-00010-of-00015.bin',
    'pytorch_model-00011-of-00015.bin', 'pytorch_model-00012-of-00015.bin',
    'pytorch_model-00013-of-00015.bin', 'pytorch_model-00014-of-00015.bin',
    'pytorch_model-00015-of-00015.bin'
]

times = 0
for i in range(LOAD_COUNT):
    print(f"--------- RUN {i} ----------")
    print("Performing read...")
    for f in files:
        with ckpt.reader(CKPT_PATH + f) as reader:
            read_state_dict = torch.load(reader)

    end_time = time.time()
    delta = end_time - start_time
    times += delta
    print(f'Iteration {i} workflow took ' + str(delta) + ' seconds.')

print("----------------------------")
print("average time per run took " + str(times / LOAD_COUNT) + " seconds.")
