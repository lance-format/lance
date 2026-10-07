#!/usr/bin/env python3
"""Publish Lance's signed Central bundle to R2; serialize callers per repository.

The java-publish workflow holds one concurrency group through artifact and
metadata uploads. Version files additionally use conditional creation, so a
retry can never replace published bytes. Metadata and its checksums are separate
objects: retry the job after an interrupted update to repair the complete set.
"""

import argparse
import hashlib
import logging
import os
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from contextlib import closing, nullcontext
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

REPOSITORY_PATH = "org/lance/lance-core"
CHECKSUMS = ("md5", "sha1", "sha256", "sha512")
IMMUTABLE_CACHE = "public, max-age=31536000, immutable"
METADATA_CACHE = "no-cache"


def version_key(version):
    """Order Lance release versions numerically, with beta < rc < stable."""
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)(?:-(beta|rc)\.(\d+))?", version)
    if not match:
        raise ValueError(f"Unsupported Lance release version: {version!r}")
    major, minor, patch, stage, number = match.groups()
    return (
        int(major),
        int(minor),
        int(patch),
        {"beta": 0, "rc": 1, None: 2}[stage],
        int(number or 0),
    )


def digest(stream, algorithm="sha256"):
    result = hashlib.new(algorithm)
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        result.update(chunk)
    return result.hexdigest()


def unpack_bundle(bundle, directory):
    """Extract one release under its Maven path without trusting ZIP paths."""
    with zipfile.ZipFile(bundle) as archive:
        names = [info.filename for info in archive.infolist() if not info.is_dir()]
        if not names or len(names) != len(set(names)):
            raise ValueError("Bundle is empty or contains duplicate paths")
        versions = set()
        for name in names:
            parts = name.split("/")
            if (
                len(parts) != 5
                or "/".join(parts[:3]) != REPOSITORY_PATH
                or "\\" in name
                or any(part in ("", ".", "..") for part in parts)
            ):
                raise ValueError(f"Unexpected bundle path: {name}")
            version_key(parts[3])
            versions.add(parts[3])
        if len(versions) != 1:
            raise ValueError(f"Expected one version in bundle, found: {versions}")
        version = versions.pop()
        # Only validated file paths are extracted; ZIP directory entries are ignored.
        for name in names:
            archive.extract(name, directory)
    return version, sorted(names)


def read_object(client, bucket, key):
    try:
        return client.get_object(Bucket=bucket, Key=key)["Body"]
    except ClientError as error:
        if error.response["Error"]["Code"] == "NoSuchKey":
            return None
        raise


def put_immutable(client, bucket, key, path):
    with path.open("rb") as stream:
        expected = digest(stream)
    existing = read_object(client, bucket, key)
    if existing is not None:
        with closing(existing):
            if digest(existing) != expected:
                raise ValueError(f"Refusing to replace published object: {key}")
        logging.info("Unchanged: %s", key)
        return
    try:
        with path.open("rb") as stream:
            client.put_object(
                Bucket=bucket,
                Key=key,
                Body=stream,
                IfNoneMatch="*",
                CacheControl=IMMUTABLE_CACHE,
            )
    except ClientError as error:
        if error.response["Error"]["Code"] != "PreconditionFailed":
            raise
        # Another writer won creation. Compare its bytes before accepting the race.
        existing = read_object(client, bucket, key)
        if existing is None:
            raise
        with closing(existing):
            if digest(existing) != expected:
                raise ValueError(
                    f"Refusing to replace published object: {key}"
                ) from error
    logging.info("Published: %s", key)


def merge_metadata(previous, version):
    versions = {version}
    if previous is not None:
        root = ET.fromstring(previous)
        if (
            root.tag != "metadata"
            or root.findtext("groupId") != "org.lance"
            or root.findtext("artifactId") != "lance-core"
        ):
            raise ValueError("Unexpected Maven metadata coordinates")
        versions.update(
            node.text for node in root.findall("versioning/versions/version")
        )
    ordered = sorted(versions, key=version_key)
    root = ET.Element("metadata")
    ET.SubElement(root, "groupId").text = "org.lance"
    ET.SubElement(root, "artifactId").text = "lance-core"
    versioning = ET.SubElement(root, "versioning")
    ET.SubElement(versioning, "latest").text = ordered[-1]
    # Maven's release means non-SNAPSHOT; numbered beta and RC versions qualify.
    ET.SubElement(versioning, "release").text = ordered[-1]
    listing = ET.SubElement(versioning, "versions")
    for value in ordered:
        ET.SubElement(listing, "version").text = value
    ET.SubElement(versioning, "lastUpdated").text = datetime.now(timezone.utc).strftime(
        "%Y%m%d%H%M%S"
    )
    ET.indent(root)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True) + b"\n"


def publish(client, bucket, bundle, prefix=""):
    prefix = prefix.strip("/")
    if prefix and any(part in ("", ".", "..") for part in prefix.split("/")):
        raise ValueError(f"Invalid repository prefix: {prefix!r}")
    prefix = f"{prefix}/" if prefix else ""
    with tempfile.TemporaryDirectory() as directory:
        version, names = unpack_bundle(bundle, directory)
        metadata_key = f"{prefix}{REPOSITORY_PATH}/maven-metadata.xml"
        existing = read_object(client, bucket, metadata_key)
        with closing(existing) if existing is not None else nullcontext():
            metadata = merge_metadata(
                existing.read() if existing is not None else None, version
            )
        for name in names:
            put_immutable(client, bucket, prefix + name, Path(directory) / name)
        client.put_object(
            Bucket=bucket,
            Key=metadata_key,
            Body=metadata,
            ContentType="application/xml",
            CacheControl=METADATA_CACHE,
        )
        for algorithm in CHECKSUMS:
            checksum = hashlib.new(algorithm, metadata).hexdigest().encode()
            client.put_object(
                Bucket=bucket,
                Key=f"{metadata_key}.{algorithm}",
                Body=checksum,
                ContentType="text/plain",
                CacheControl=METADATA_CACHE,
            )
    logging.info("Published org.lance:lance-core:%s (%s files)", version, len(names))
    return version


def verify_public(version, public_url):
    """Check that the new release is accessible through the public domain."""
    pom = f"{REPOSITORY_PATH}/{version}/lance-core-{version}.pom"
    subprocess.run(
        [
            "curl",
            "--fail",
            "--silent",
            "--show-error",
            "--retry",
            "5",
            "--output",
            os.devnull,
            f"{public_url.rstrip('/')}/{quote(pom)}",
        ],
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument(
        "--prefix", default="", help="Isolated repository prefix for tests"
    )
    parser.add_argument("--public-url", help="Check public access to the uploaded POM")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    client = boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT"],
        region_name="auto",
        config=Config(retries={"mode": "standard", "max_attempts": 5}),
    )
    version = publish(client, os.environ["R2_BUCKET"], args.bundle, args.prefix)
    if args.public_url:
        verify_public(version, args.public_url)


if __name__ == "__main__":
    main()
