"""Tests for the Maven bundle contract and interrupted R2 publication."""

import hashlib
import io
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

import pytest
from botocore.exceptions import ClientError

from publish_java_r2 import (
    CHECKSUMS,
    IMMUTABLE_CACHE,
    METADATA_CACHE,
    REPOSITORY_PATH,
    merge_metadata,
    publish,
    version_key,
)


def bundle_files(version="13.1.0-beta.2"):
    base = f"{REPOSITORY_PATH}/{version}/lance-core-{version}"
    pom = (
        f'<project xmlns="http://maven.apache.org/POM/4.0.0">'
        f"<groupId>org.lance</groupId><artifactId>lance-core</artifactId>"
        f"<version>{version}</version></project>"
    ).encode()
    files = {}
    for suffix in (".pom", ".jar", "-sources.jar", "-javadoc.jar"):
        name = base + suffix
        content = pom if suffix == ".pom" else b"test jar"
        files[name] = content
        files[name + ".asc"] = b"test signature"
        for algorithm in CHECKSUMS:
            files[f"{name}.{algorithm}"] = (
                hashlib.new(algorithm, content).hexdigest().encode()
            )
    return files


def write_bundle(path, files):
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return path


class MemoryS3:
    def __init__(self):
        self.objects = {}
        self.cache = {}
        self.puts = []
        self.fail_key = None
        self.race_content = None

    def get_object(self, *, Bucket, Key):
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        return {"Body": io.BytesIO(self.objects[Key])}

    def put_object(
        self, *, Bucket, Key, Body, CacheControl, IfNoneMatch=None, ContentType=None
    ):
        if Key == self.fail_key:
            raise RuntimeError("injected upload failure")
        if IfNoneMatch == "*":
            if self.race_content is not None:
                self.objects[Key] = self.race_content
                self.race_content = None
            if Key in self.objects:
                raise ClientError(
                    {"Error": {"Code": "PreconditionFailed"}}, "PutObject"
                )
        self.objects[Key] = Body.read() if hasattr(Body, "read") else Body
        self.cache[Key] = CacheControl
        self.puts.append(Key)


def test_publish_and_retry_preserve_all_bundle_bytes(tmp_path):
    files = bundle_files()
    bundle = write_bundle(tmp_path / "bundle.zip", files)
    client = MemoryS3()
    publish(client, "bucket", bundle, "test")
    for name, content in files.items():
        assert client.objects["test/" + name] == content
        assert client.cache["test/" + name] == IMMUTABLE_CACHE
    client.puts.clear()
    publish(client, "bucket", bundle, "test")
    assert all("maven-metadata.xml" in key for key in client.puts)
    metadata_key = "test/" + REPOSITORY_PATH + "/maven-metadata.xml"
    for algorithm in CHECKSUMS:
        assert (
            client.objects[f"{metadata_key}.{algorithm}"]
            == hashlib.new(algorithm, client.objects[metadata_key]).hexdigest().encode()
        )
        assert client.cache[f"{metadata_key}.{algorithm}"] == METADATA_CACHE
    assert client.cache[metadata_key] == METADATA_CACHE


@pytest.mark.parametrize("failure", ["artifact", "metadata", "checksum"])
def test_retry_repairs_interrupted_publication(tmp_path, failure):
    files = bundle_files()
    bundle = write_bundle(tmp_path / "bundle.zip", files)
    client = MemoryS3()
    metadata_key = REPOSITORY_PATH + "/maven-metadata.xml"
    client.fail_key = {
        "artifact": sorted(files)[3],
        "metadata": metadata_key,
        "checksum": metadata_key + ".sha256",
    }[failure]
    with pytest.raises(RuntimeError, match="injected upload failure"):
        publish(client, "bucket", bundle)
    if failure == "artifact":
        assert metadata_key not in client.objects
    client.fail_key = None
    publish(client, "bucket", bundle)
    assert len(client.objects) == len(files) + 5
    assert (
        client.objects[metadata_key + ".sha256"]
        == hashlib.sha256(client.objects[metadata_key]).hexdigest().encode()
    )


@pytest.mark.parametrize("race", [False, True])
def test_never_overwrites_different_version_bytes(tmp_path, race):
    files = bundle_files()
    client = MemoryS3()
    key = sorted(files)[0]
    if race:
        client.race_content = b"different published bytes"
    else:
        client.objects[key] = b"different published bytes"
    with pytest.raises(ValueError, match="Refusing to replace published object"):
        publish(client, "bucket", write_bundle(tmp_path / "bundle.zip", files))
    assert client.objects[key] == b"different published bytes"
    assert not client.puts


def test_accepts_identical_concurrent_create(tmp_path):
    files = bundle_files()
    client = MemoryS3()
    client.race_content = files[sorted(files)[0]]
    publish(client, "bucket", write_bundle(tmp_path / "bundle.zip", files))
    assert all(client.objects[key] == content for key, content in files.items())


@pytest.mark.parametrize(
    "invalid",
    ["missing", "checksum", "path", "duplicate", "coordinates", "multiple_versions"],
)
def test_rejects_bad_bundle_before_any_write(tmp_path, invalid):
    files = bundle_files()
    pom = next(name for name in files if name.endswith(".pom"))
    if invalid == "missing":
        del files[pom + ".asc"]
    elif invalid == "checksum":
        files[pom + ".sha256"] = b"0" * 64
    elif invalid == "path":
        files["../escape"] = b"unexpected"
    elif invalid == "coordinates":
        files[pom] = files[pom].replace(b"org.lance", b"org.other")
        for algorithm in CHECKSUMS:
            files[pom + "." + algorithm] = (
                hashlib.new(algorithm, files[pom]).hexdigest().encode()
            )
    elif invalid == "multiple_versions":
        files.update(bundle_files("13.1.0-rc.1"))
    bundle = write_bundle(tmp_path / "bundle.zip", files)
    if invalid == "duplicate":
        with zipfile.ZipFile(bundle, "a") as archive, pytest.warns(UserWarning):
            archive.writestr(pom, files[pom])
    client = MemoryS3()
    with pytest.raises(ValueError):
        publish(client, "bucket", bundle)
    assert not client.puts


def test_metadata_orders_all_release_channels_and_backfills():
    versions = [
        "13.1.0-beta.10",
        "13.1.0",
        "13.0.9",
        "13.1.0-rc.2",
        "13.1.0-beta.2",
        "13.1.0-rc.10",
        "14.0.0-beta.1",
    ]
    metadata = None
    for version in versions + versions:
        metadata = merge_metadata(metadata, version)
    root = ET.fromstring(metadata)
    assert [node.text for node in root.findall("versioning/versions/version")] == [
        "13.0.9",
        "13.1.0-beta.2",
        "13.1.0-beta.10",
        "13.1.0-rc.2",
        "13.1.0-rc.10",
        "13.1.0",
        "14.0.0-beta.1",
    ]
    assert root.findtext("versioning/latest") == "14.0.0-beta.1"
    assert root.findtext("versioning/release") == "14.0.0-beta.1"


@pytest.mark.parametrize("version", ["1.0-SNAPSHOT", "1.0.0-alpha.1", "../bad"])
def test_rejects_versions_outside_lance_release_contract(version):
    with pytest.raises(ValueError, match="Unsupported Lance release version"):
        version_key(version)


def test_dry_run_validates_without_s3(tmp_path):
    from publish_java_r2 import main

    bundle = write_bundle(tmp_path / "bundle.zip", bundle_files())
    with (
        patch("sys.argv", ["publish_java_r2.py", str(bundle), "--validate-only"]),
        patch("publish_java_r2.boto3.client") as client,
    ):
        main()
    client.assert_not_called()


def test_workflow_serializes_r2_and_preserves_bundle():
    import yaml

    workflow = yaml.safe_load(
        (Path(__file__).parents[1] / ".github/workflows/java-publish.yml").read_text()
    )
    jobs = workflow["jobs"]
    r2 = jobs["publish-r2"]
    assert r2["needs"] == "publish"
    assert r2["if"] == "github.event_name == 'release' || inputs.mode == 'release'"
    assert r2["concurrency"] == {"group": "java-maven-r2", "queue": "max"}
    assert "publish-r2" in jobs["report-failure"]["needs"]
    uploads = [
        step
        for step in jobs["publish"]["steps"]
        if step.get("uses", "").startswith("actions/upload-artifact@")
    ]
    assert uploads[0]["with"]["name"] == "java-central-bundle"
    assert uploads[0]["with"]["retention-days"] == 30
    assert not any("deploy -P" in step.get("run", "") for step in r2["steps"])
