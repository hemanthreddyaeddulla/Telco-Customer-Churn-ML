#!/usr/bin/env python3
"""
Deploy the app to a free Hugging Face Space (Docker SDK).

One-time prerequisites:
    pip install -U huggingface_hub
    huggingface-cli login           # paste a WRITE token from huggingface.co/settings/tokens

Usage:
    python scripts/deploy_hf.py --user <your_hf_username>

Creates (or updates) a public Docker Space and uploads the files needed to build the
serving image. Hugging Face then builds the Dockerfile (which fetches the dataset,
trains the model, and starts FastAPI+Gradio) and hosts it for free. The live URL is
printed at the end. Your GitHub repo is untouched.
"""

import argparse
import os
import shutil
import tempfile

from huggingface_hub import HfApi

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Hugging Face Space config (frontmatter) + landing text. app_port must match the app.
HF_README = """---
title: Telco Customer Churn
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
---

# Telco Customer Churn - live demo

FastAPI + Gradio churn predictor: an XGBoost pipeline with calibrated probabilities,
a cost-based decision threshold, and SHAP explanations.

- `/ui`   - interactive Gradio app
- `/docs` - REST API documentation
- `/`     - health check

Source code & full write-up:
https://github.com/hemanthreddyaeddulla/Telco-Customer-Churn-ML
"""

# What the Docker build needs (the Dockerfile fetches the dataset itself at build time).
INCLUDE = ["src", "scripts", "requirements.txt"]


def main(args):
    api = HfApi()
    repo_id = f"{args.user}/{args.name}"
    api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
    print(f"Space ready: https://huggingface.co/spaces/{repo_id}")

    with tempfile.TemporaryDirectory() as tmp:
        if args.image:
            # Serve a pre-built public image (e.g. from Docker Hub): HF just pulls and runs
            # it, with no training during the Space build. The image already carries the app,
            # the trained model and the CMD/EXPOSE, so the Space Dockerfile is a single line.
            # Use this when the full --tune build is too heavy for HF's free build tier.
            with open(os.path.join(tmp, "Dockerfile"), "w", encoding="utf-8") as f:
                f.write(f"FROM {args.image}\n")
        else:
            for item in INCLUDE:
                src = os.path.join(ROOT, item)
                dst = os.path.join(tmp, item)
                if os.path.isdir(src):
                    shutil.copytree(
                        src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
                    )
                else:
                    shutil.copy2(src, dst)
            # HF requires a capital-D Dockerfile and a README.md carrying the Space metadata.
            shutil.copy2(os.path.join(ROOT, "dockerfile"), os.path.join(tmp, "Dockerfile"))
        with open(os.path.join(tmp, "README.md"), "w", encoding="utf-8") as f:
            f.write(HF_README)
        api.upload_folder(folder_path=tmp, repo_id=repo_id, repo_type="space")

    print("Uploaded. Hugging Face is now building the image (watch the Space's Logs tab).")
    print(f"Live in a few minutes at: https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Deploy to a free Hugging Face Space")
    p.add_argument("--user", required=True, help="your Hugging Face username")
    p.add_argument("--name", default="telco-churn", help="Space name")
    p.add_argument(
        "--image",
        default=None,
        help="serve a pre-built public image (e.g. youruser/telco-churn:latest) instead of "
        "training during the HF build",
    )
    main(p.parse_args())
