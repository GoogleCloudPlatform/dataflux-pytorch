from ray.job_submission import JobSubmissionClient

client = JobSubmissionClient("http://127.0.0.1:8265")
job_id = client.submit_job(
    # Entrypoint shell command to execute.
    entrypoint="bash run_and_time.sh",
    # Path to the local directory that contains the entrypoint file.
    runtime_env={
        "working_dir": "./",
        "pip": ["torch", "torchvision", "tqdm", "scipy", "numpy", "nibabel"],
    },
)
print(f"""
    -------------------------------------------------------
    Job '{job_id}' submitted successfully
    -------------------------------------------------------

    Next steps
      Tail the logs of the job:
        ray job logs {job_id} --follow
      Query the status of the job:
        ray job status {job_id}
      Request the job to be stopped:
        ray job stop {job_id}
    """)
