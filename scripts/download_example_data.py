#!/usr/bin/env python3
import argparse
import json
import os
import posixpath
import shutil
import sys
import urllib.error
import urllib.parse
import urllib.request


REPO_ID = "travisdriver/phomo-data"
REVISION = "main"

REMOTE_RECONSTRUCTION_DIR = (
    "reconstructions/cornelia/"
    "reconstruction_init_roma_lunar_lambert_schroder2013_gtsfm_if_corrected_half_res"
)
REMOTE_IMAGES_DIR = "images/cornelia"

LOCAL_RECONSTRUCTION_DIR = "data/reconstructions/cornelia/lunar_lambert_if_corrected_half_res/init"
LOCAL_IMAGES_DIR = "data/images/cornelia"


def _api_list_tree(repo_id, revision, path):
    quoted_path = urllib.parse.quote(path, safe="/")
    url = f"https://huggingface.co/api/datasets/{repo_id}/tree/{revision}/{quoted_path}"
    try:
        with urllib.request.urlopen(url) as response:
            payload = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Failed to list '{path}': HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to list '{path}': {exc.reason}") from exc

    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON from {url}") from exc


def _gather_files(repo_id, revision, base_path):
    files = []
    stack = [base_path]
    while stack:
        current = stack.pop()
        entries = _api_list_tree(repo_id, revision, current)
        for entry in entries:
            entry_type = entry.get("type")
            entry_path = entry.get("path")
            if not entry_type or not entry_path:
                continue
            if entry_type == "file":
                files.append(entry_path)
            elif entry_type == "directory":
                stack.append(entry_path)
    return files


def _download_file(repo_id, revision, file_path, local_path, force):
    if os.path.exists(local_path) and not force:
        return "skip"

    quoted_path = urllib.parse.quote(file_path, safe="/")
    url = f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{quoted_path}"
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    try:
        with urllib.request.urlopen(url) as response, open(local_path, "wb") as handle:
            shutil.copyfileobj(response, handle)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Failed to download '{file_path}': HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to download '{file_path}': {exc.reason}") from exc
    return "download"


def _download_directory(repo_id, revision, remote_dir, local_dir, force):
    files = _gather_files(repo_id, revision, remote_dir)
    if not files:
        raise RuntimeError(f"No files found under '{remote_dir}'.")

    downloaded = 0
    skipped = 0
    for file_path in sorted(files):
        relative_path = posixpath.relpath(file_path, remote_dir)
        local_path = os.path.join(local_dir, relative_path)
        action = _download_file(repo_id, revision, file_path, local_path, force)
        if action == "download":
            downloaded += 1
        else:
            skipped += 1
    return downloaded, skipped


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download PhoMo example data from Hugging Face datasets.",
    )
    parser.add_argument("--repo", default=REPO_ID, help="Hugging Face dataset repo ID.")
    parser.add_argument("--revision", default=REVISION, help="Repo revision or branch.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files instead of skipping.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    targets = [
        (REMOTE_RECONSTRUCTION_DIR, LOCAL_RECONSTRUCTION_DIR),
        (REMOTE_IMAGES_DIR, LOCAL_IMAGES_DIR),
    ]

    total_downloaded = 0
    total_skipped = 0
    for remote_dir, local_dir in targets:
        print(f"Syncing {remote_dir} -> {local_dir}")
        downloaded, skipped = _download_directory(
            args.repo, args.revision, remote_dir, local_dir, args.force
        )
        total_downloaded += downloaded
        total_skipped += skipped
        print(f"  downloaded: {downloaded}, skipped: {skipped}")

    print(f"Done. downloaded: {total_downloaded}, skipped: {total_skipped}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
