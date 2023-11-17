# OpenRQA: retrieval augmented QA systems that can be easily served, trained, and evaluated

```bash
docker run -it --gpus all --shm-size=256m \
-v /local/data:/local/data \
-v /local2/data:/local2/data \
-v /proj/interaction/interaction-filer/xy2437:/proj/interaction/interaction-filer/xy2437 \  # change this
-v /local/data/xy2437/.cache:/home/docker/.cache \  # change this
-v /home/xy2437/OpenRQA:/workspace/OpenRQA \  # change this
jasonyux/nvcr_torch:23.08 bash
```
