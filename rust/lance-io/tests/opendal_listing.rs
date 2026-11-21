// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

#![cfg(feature = "aws")]

use std::collections::HashMap;
use std::sync::Arc;

use lance_io::object_store::{
    ObjectStore, ObjectStoreParams, ReadDirOptions, StorageOptionsAccessor,
};
use object_store::ObjectStoreExt;
use object_store::path::Path;
use object_store::prefix::PrefixStore;
use rstest::rstest;
use wiremock::matchers::{method, path, query_param};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn test_delimited_location_reopens_listed_key() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(query_param("prefix", "run~1/"))
        .and(query_param("delimiter", "/"))
        .respond_with(ResponseTemplate::new(200).set_body_string(
            "<ListBucketResult><IsTruncated>false</IsTruncated>\
             <Contents><Key>run~1/July%20report.json</Key>\
             <LastModified>2026-07-01T00:00:00Z</LastModified><Size>1</Size>\
             </Contents></ListBucketResult>",
        ))
        .up_to_n_times(1)
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/bucket/run~1/July%2520report.json"))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(b"1".to_vec()))
        .expect(1)
        .mount(&server)
        .await;
    let (store, dir) = opendal_s3(&server).await;
    let listed = store.list_with_delimiter(Some(&dir)).await.unwrap();
    assert_eq!(listed.objects.len(), 1);
    assert_eq!(
        listed.objects[0].location,
        dir.join("July%20report.json"),
        "listed location must reopen the report"
    );
    let contents = store.inner.get(&listed.objects[0].location).await.unwrap();
    assert_eq!(contents.bytes().await.unwrap().as_ref(), b"1");
}

#[rstest]
#[case::single(&["mount~1/part%23one"], "mount~1/part#one/")]
#[case::nested(&["mount~1", "part%23one"], "part#one/")]
#[tokio::test]
async fn test_mounted_page_resumes_after_directory_marker(
    #[case] mounts: &[&str],
    #[case] marker_token: &str,
) {
    let server = MockServer::start().await;
    let mut xml = String::from("<ListBucketResult><IsTruncated>false</IsTruncated>");
    for name in ["", "a", "b"] {
        xml.push_str(&format!(
            "<Contents><Key>mount~1/part#one/{name}</Key>\
             <LastModified>2026-07-01T00:00:00Z</LastModified><Size>0</Size></Contents>"
        ));
    }
    xml.push_str("</ListBucketResult>");
    Mock::given(method("GET"))
        .and(query_param("prefix", "mount~1/part#one/"))
        .and(query_param("delimiter", "/"))
        .respond_with(ResponseTemplate::new(200).set_body_string(xml))
        .expect(2)
        .mount(&server)
        .await;
    let (store, _) = opendal_s3(&server).await;
    let mut mounted = ObjectStore::memory();
    mounted.inner = store.inner.clone();
    for mount in mounts {
        mounted.inner = Arc::new(PrefixStore::new(
            mounted.inner,
            Path::parse(*mount).unwrap(),
        ));
    }
    let mut token = Some(marker_token.into());
    for expected in ["a", "b"] {
        let page = mounted
            .read_dir_page(
                Path::default(),
                ReadDirOptions {
                    page_token: token,
                    limit: Some(1),
                },
            )
            .await
            .unwrap();
        assert_eq!(page.result.objects.len(), 1);
        assert_eq!(page.result.objects[0].location.filename(), Some(expected));
        assert!(page.result.common_prefixes.is_empty());
        token = page.page_token;
    }
    assert!(token.is_none());
}

async fn opendal_s3(server: &MockServer) -> (Arc<ObjectStore>, Path) {
    let options = HashMap::from([
        ("use_opendal".into(), "true".into()),
        ("aws_endpoint".into(), server.uri()),
        ("aws_region".into(), "us-east-1".into()),
        ("aws_access_key_id".into(), "test".into()),
        ("aws_secret_access_key".into(), "test".into()),
        ("allow_http".into(), "true".into()),
    ]);
    let params = ObjectStoreParams {
        storage_options_accessor: Some(Arc::new(StorageOptionsAccessor::with_static_options(
            options,
        ))),
        ..Default::default()
    };
    ObjectStore::from_uri_and_params(Default::default(), "s3://bucket/run~1", &params)
        .await
        .unwrap()
}
