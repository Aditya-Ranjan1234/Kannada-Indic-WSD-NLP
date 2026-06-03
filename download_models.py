import os
from huggingface_hub import snapshot_download
from huggingface_hub import logging

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
logging.set_verbosity_info()

def download_models():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    models = [
        "xlm-roberta-base",
        "bert-base-multilingual-cased",
        "google/muril-base-cased",
        "ai4bharat/indic-bert"
    ]

    for model_id in models:
        print(f"\nDownloading {model_id}...")

        local_dir = os.path.join(
            models_dir,
            model_id.replace("/", "_")
        )

        token = os.getenv("HF_HUB_TOKEN")

        try:
            snapshot_download(
                repo_id=model_id,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                allow_patterns=[
                    "*.json",
                    "*.txt",
                    "*.model",
                    "*.safetensors",
                    "*.bin"
                ],
                use_auth_token=token,
                resume_download=True
            )

            print(f"Done: {model_id}")

        except Exception as e:
            print(f"Failed: {model_id}")
            print(e)
            # continue to next model without aborting
            continue

if __name__ == "__main__":
    download_models()