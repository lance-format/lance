# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import os
import shutil
import subprocess
import sys
import tarfile
import traceback
from io import BytesIO

from .lance import language_model_home

LANGUAGE_MODEL_HOME = language_model_home()


def _safe_tar_extractall(tar, dest_dir):
    """Extract tar safely, blocking path traversal attacks (CVE-2007-4559).

    On Python >= 3.12 uses the built-in ``filter="data"`` safeguard.
    On older versions, manually validates every member path.
    """
    if sys.version_info >= (3, 12):
        tar.extractall(path=dest_dir, filter="data")
    else:
        abs_dest = os.path.realpath(dest_dir)
        for member in tar.getmembers():
            member_path = os.path.join(dest_dir, member.name)
            abs_member = os.path.realpath(member_path)
            if not abs_member.startswith(abs_dest + os.sep) and abs_member != abs_dest:
                raise Exception(
                    f"Tar member '{member.name}' would extract outside target directory"
                )
        tar.extractall(path=dest_dir)


def check_lindera():
    if not shutil.which("lindera"):
        raise Exception(
            "lindera is not installed. Please install it by following https://github.com/lindera/lindera/tree/main/lindera-cli"
        )


def import_requests():
    try:
        import requests
    except Exception:
        raise Exception("requests is not installed, Please pip install requests")
    return requests


def download_jieba():
    dirname = os.path.join(LANGUAGE_MODEL_HOME, "jieba", "default")
    os.makedirs(dirname, exist_ok=True)
    try:
        requests = import_requests()
        resp = requests.get(
            "https://github.com/messense/jieba-rs/raw/refs/heads/main/jieba/src/data/dict.txt"
        )
        content = resp.content
        with open(os.path.join(dirname, "dict.txt"), "wb") as out:
            out.write(content)
    except Exception as _:
        traceback.print_exc()
        print(
            "Download jieba language model failed. Please download dict.txt from "
            "https://github.com/messense/jieba-rs/tree/main/jieba/src/data "
            f"and put it in {dirname}"
        )


def download_lindera(lm: str):
    requests = import_requests()
    dirname = os.path.join(LANGUAGE_MODEL_HOME, "lindera", lm)
    src_dirname = os.path.join(dirname, "src")
    if lm == "ipadic":
        url = "https://Lindera.dev/mecab-ipadic-2.7.0-20070801.tar.gz"
    elif lm == "ko-dic":
        url = "https://Lindera.dev/mecab-ko-dic-2.1.1-20180720.tar.gz"
    elif lm == "unidic":
        url = "https://Lindera.dev/unidic-mecab-2.1.2.tar.gz"
    else:
        raise Exception(f"language model {lm} is not supported")
    os.makedirs(src_dirname, exist_ok=True)
    print(f"downloading language model: {url}")
    data = requests.get(url).content
    print(f"unzip language model: {url}")

    cwd = os.getcwd()
    try:
        os.chdir(src_dirname)
        with tarfile.open(fileobj=BytesIO(data)) as tar:
            _safe_tar_extractall(tar, src_dirname)
            name = tar.getnames()[0]
        cmd = [
            "lindera",
            "build",
            f"--dictionary-kind={lm}",
            os.path.join(src_dirname, name),
            os.path.join(dirname, "main"),
        ]
        print(f"compiling language model: {' '.join(cmd)}")
        subprocess.run(cmd)
    finally:
        os.chdir(cwd)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Lance tokenizer language model downloader"
    )
    parser.add_argument("tokenizer", choices=["jieba", "lindera"])
    parser.add_argument("-l", "--languagemodel")
    args = parser.parse_args()
    print(f"LANCE_LANGUAGE_MODEL_HOME={LANGUAGE_MODEL_HOME}")
    if args.tokenizer == "jieba":
        download_jieba()
    elif args.tokenizer == "lindera":
        download_lindera(args.languagemodel)


if __name__ == "__main__":
    main()
