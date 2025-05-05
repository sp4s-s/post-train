FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

RUN pip install --no-cache-dir \
    megatron-lm==2.0.0 \
    vllm==0.6.0 \
    transformers==4.35.0 \
    mpi4py==3.1.4 \
    requests==2.31.0 \
    datasets==2.13.1

WORKDIR /workspace
COPY src /workspace/src

ENV OMP_NUM_THREADS=1
ENV NCCL_DEBUG=INFO
ENV CUDA_DEVICE_MAX_CONNECTIONS=1

# Entrypoint command
ENTRYPOINT ["bash", "/workspace/src/entrypoint.sh"]
