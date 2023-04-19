# Downloading pre-trained teacher models

To compare with our baselines, we start from their pre-trained teacher model checkpoints and distill our student model.

Follow the [guide](https://cloud.google.com/storage/docs/gsutil_install) to install gsutils that will be used to download checkpoints from Google Cloud Storage.

Since the original model is written in JAX, we need to convert the teacher's weights from JAX to torch.

Run

```bash
python teacher/download_and_convert_jax.py
```
