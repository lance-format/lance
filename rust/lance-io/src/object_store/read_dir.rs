// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Paginated listing of a single directory level.
//!
//! [`ObjectStore::read_dir_page`] returns one page of the immediate children of a prefix, plus
//! a token that resumes after it. Where the backend implements `object_store`'s paginated list
//! API — S3, GCS and Azure do — the page size and the resume position are pushed into the list
//! request, so a caller that wants the first few children pays for the first few children
//! rather than for the whole prefix. Everything else lists the level in full and pages
//! locally, which is correct but costs what the whole directory costs.
//!
//! The token is opaque, and a caller only ever hands it back: it is the backend's own
//! continuation token where there is one, and the last key of the page where there is not.
//! That is what lets a store without key-ordered listings, such as S3 Express, be paged at
//! all — nothing outside this module compares one token to another.

use object_store::list::{PaginatedListOptions, PaginatedListResult, PaginatedListStore};
use object_store::{ListResult, ObjectMeta, ObjectStore as OSObjectStore, path::Path};
use tracing::instrument;

use lance_core::{Error, Result};

use super::ObjectStore;

#[cfg(feature = "metrics")]
use crate::object_store::metrics::{InFlightGuard, record_outcome};
#[cfg(feature = "metrics")]
use std::time::Instant;

/// The path delimiter that separates directory levels.
const DELIMITER: &str = "/";

/// Operation label for the metrics and IO statistics a paginated listing records.
const LIST_OP: &str = "list_paginated";

/// Options for [`ObjectStore::read_dir_page`].
#[derive(Debug, Clone, Default)]
pub struct ReadDirOptions {
    /// Resume after the page a previous call returned, using the token it handed back.
    ///
    /// A token means something only to the store that minted it, and only for the directory
    /// it was minted over. Handing one to a different store resumes from the wrong place
    /// rather than failing.
    pub page_token: Option<String>,
    /// The page size to ask the backend for. Must be at least one. `None` lets the backend
    /// return as much as it will.
    pub limit: Option<usize>,
}

impl ObjectStore {
    /// One page of the immediate children of `dir`, one directory level deep.
    ///
    /// On backends with a paginated list API — S3, GCS and Azure — the resume position and the
    /// page size are pushed into the list request, so the page costs what the page holds.
    /// Elsewhere the directory is listed in full and paged locally, which is correct but no
    /// cheaper than [`Self::read_dir`].
    ///
    /// Child directories come back as [`ListResult::common_prefixes`] and child objects as
    /// [`ListResult::objects`], the same split [`Self::list_with_delimiter`] returns.
    ///
    /// One page is one request, so a page can hold fewer children than `limit` asked for and
    /// still be followed by more: with a delimiter a backend spends its page budget on keys it
    /// collapses away, and it has a cap of its own besides. Walk until
    /// [`PaginatedListResult::page_token`] is `None` rather than until a page comes back short.
    ///
    /// ```
    /// # use lance_io::object_store::{ObjectStore, ReadDirOptions};
    /// # async fn example(store: &ObjectStore) -> lance_core::Result<Vec<String>> {
    /// let mut tables = Vec::new();
    /// let mut page_token = None;
    /// loop {
    ///     let page = store
    ///         .read_dir_page("my_db", ReadDirOptions { page_token, limit: Some(10) })
    ///         .await?;
    ///     // A table is a directory, so a loose object that happens to be named like one is not
    ///     // a table.
    ///     tables.extend(page.result.common_prefixes.iter().filter_map(|table| {
    ///         Some(table.filename()?.strip_suffix(".lance")?.to_string())
    ///     }));
    ///     page_token = page.page_token;
    ///     if page_token.is_none() || tables.len() >= 10 {
    ///         break;
    ///     }
    /// }
    /// # Ok(tables)
    /// # }
    /// ```
    pub async fn read_dir_page(
        &self,
        dir: impl Into<Path>,
        options: ReadDirOptions,
    ) -> Result<PaginatedListResult> {
        let dir = dir.into();
        // A page of nothing cannot advance a listing, and the two paths below would disagree
        // about what it means: the pushdown path would report an empty directory while the
        // full listing ignored the limit and returned everything.
        if options.limit == Some(0) {
            return Err(Error::invalid_input(
                "read_dir_page limit must be at least 1, got 0",
            ));
        }
        match &self.paginated_lister {
            Some(lister) => self.pushdown_page(lister.as_ref(), &dir, options).await,
            // Goes through `inner`, so the wrappers around it instrument the request. The
            // pushdown path talks to the backend directly and instruments itself.
            None => full_listing_page(self.inner.as_ref(), &dir, options).await,
        }
    }

    /// One page from the backend's own paginated list API.
    ///
    /// The pushdown path holds the backend directly, so its request never passes through the
    /// wrappers around [`Self::inner`] that would otherwise record it, and it records itself.
    #[instrument(level = "debug", skip_all, fields(dir = %dir))]
    async fn pushdown_page(
        &self,
        lister: &dyn PaginatedListStore,
        dir: &Path,
        options: ReadDirOptions,
    ) -> Result<PaginatedListResult> {
        let prefix = list_prefix(dir);
        self.io_tracker.record_read(LIST_OP, dir.clone(), 0, None);
        #[cfg(feature = "metrics")]
        let _in_flight = InFlightGuard::new(&self.store_prefix, LIST_OP);
        #[cfg(feature = "metrics")]
        let start = Instant::now();

        let page = lister
            .list_paginated(
                prefix.as_deref(),
                PaginatedListOptions {
                    delimiter: Some(DELIMITER.into()),
                    max_keys: options.limit,
                    page_token: options.page_token,
                    // `offset` is left unset: a continuation token is a position of its own,
                    // and a caller-supplied key means something different on every store —
                    // S3 excludes it, Azure includes it.
                    ..Default::default()
                },
            )
            .await;

        #[cfg(feature = "metrics")]
        record_outcome(&self.store_prefix, LIST_OP, start, 0, page.is_err());
        let mut page = page?;

        retain_children(&mut page.result, prefix.as_deref());
        Ok(page)
    }
}

/// The prefix to list under, carrying the trailing delimiter that the paginated API expects.
/// `None` for the root of the store, which has no prefix at all.
fn list_prefix(dir: &Path) -> Option<String> {
    let dir = dir.as_ref();
    (!dir.is_empty()).then(|| format!("{dir}{DELIMITER}"))
}

/// One page of a directory on a store with no paginated list API: list the level in full and
/// page it locally.
///
/// The page has to be the smallest `limit` children past the token rather than any `limit` of
/// them, since the next call lists the same directory again and keeps only what sorts after
/// the key this page hands back. That means putting the listing in key order, which
/// `list_with_delimiter` does not promise — a sort over children already in memory, costing no
/// extra request.
async fn full_listing_page(
    store: &dyn OSObjectStore,
    dir: &Path,
    options: ReadDirOptions,
) -> Result<PaginatedListResult> {
    let listed = store.list_with_delimiter(Some(dir)).await?;
    let mut children = keyed_children(listed, list_prefix(dir).as_deref());
    if let Some(resume) = &options.page_token {
        children.retain(|child| child.key > *resume);
    }
    let total = children.len();
    children.truncate(options.limit.unwrap_or(total).min(total));
    // The last key this page took, so a page that took nothing ends the listing rather than
    // resuming from a position no page ever reached.
    let page_token = match children.last() {
        Some(last) if children.len() < total => Some(last.key.clone()),
        _ => None,
    };

    let mut result = ListResult {
        common_prefixes: Vec::new(),
        objects: Vec::new(),
    };
    for child in children {
        match child.child {
            Child::Directory(location) => result.common_prefixes.push(location),
            Child::File(meta) => result.objects.push(meta),
        }
    }
    Ok(PaginatedListResult { result, page_token })
}

/// Drop everything in `listed` that is not a child of the level being listed.
///
/// This covers the marker object some stores keep for a directory: it lists as an object whose
/// location is the directory's own prefix.
fn retain_children(listed: &mut ListResult, prefix: Option<&str>) {
    listed
        .common_prefixes
        .retain(|location| relative_key(prefix, location).is_some());
    listed
        .objects
        .retain(|object| relative_key(prefix, &object.location).is_some());
}

/// A child of the directory being listed, with the key the backend listed it under.
struct KeyedChild {
    /// The key relative to the directory, which is what a full-listing token names. A child
    /// directory keeps its trailing delimiter, since that is the prefix its keys share and so
    /// where it sits in the listing; a child file is its name.
    key: String,
    child: Child,
}

enum Child {
    Directory(Path),
    File(ObjectMeta),
}

/// The children of `prefix` in `listed`, in key order, each with the key it was listed under.
///
/// Stores report common prefixes and objects as two separate lists, so the two are put back
/// into one order here — the order a full-listing token pages through. Anything that is not a
/// child of this level is dropped, as in [`retain_children`].
fn keyed_children(listed: ListResult, prefix: Option<&str>) -> Vec<KeyedChild> {
    let ListResult {
        common_prefixes,
        objects,
    } = listed;
    let directories = common_prefixes.into_iter().filter_map(|location| {
        let key = format!("{}{DELIMITER}", relative_key(prefix, &location)?);
        Some(KeyedChild {
            key,
            child: Child::Directory(location),
        })
    });
    let files = objects.into_iter().filter_map(|meta| {
        let key = relative_key(prefix, &meta.location)?.to_string();
        Some(KeyedChild {
            key,
            child: Child::File(meta),
        })
    });
    let mut children: Vec<KeyedChild> = directories.chain(files).collect();
    children.sort_unstable_by(|left, right| left.key.cmp(&right.key));
    children
}

/// Where a listed location sits inside the directory being listed, which is the space
/// full-listing tokens live in, or `None` if it is not a child of that directory at all.
fn relative_key<'a>(prefix: Option<&str>, location: &'a Path) -> Option<&'a str> {
    let location = location.as_ref();
    let relative = match prefix {
        // Both halves of the prefix, so a location that merely starts with the directory's
        // name — `dbx/y` against `db/` — is reported as not being under it, and so is the
        // directory's own marker, whose location is `db`.
        Some(prefix) => location.strip_prefix(prefix)?,
        None => location,
    };
    (!relative.is_empty()).then_some(relative)
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::object_store::{ObjectStoreParams, ObjectStoreRegistry};
    use chrono::Utc;
    use object_store::memory::InMemory;
    use object_store::{ObjectStoreExt, PutPayload};
    use rstest::rstest;

    /// How the store under test resolves a listing.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Backend {
        /// No paginated API: list the whole directory and page it locally.
        FullListing,
        /// A paginated API, as the native S3, GCS and Azure stores have.
        Pushdown,
    }
    use Backend::{FullListing, Pushdown};

    /// One list request, as the backend saw it.
    #[derive(Debug, Clone)]
    struct ListRequest {
        prefix: Option<String>,
        opts: PaginatedListOptions,
    }

    /// A stand-in for a store with a paginated list API.
    ///
    /// Keys are listed in the order they were given, a delimiter collapses each level, and the
    /// continuation token is a position in that listing order — which is what a real token is:
    /// exact, and never compared against a key. `page_bound` is the store's own cap, which is
    /// why a page can come back holding less than it was asked for.
    #[derive(Debug)]
    struct FakeListStore {
        keys: Vec<String>,
        page_bound: usize,
        requests: Arc<Mutex<Vec<ListRequest>>>,
    }

    #[async_trait::async_trait]
    impl PaginatedListStore for FakeListStore {
        async fn list_paginated(
            &self,
            prefix: Option<&str>,
            opts: PaginatedListOptions,
        ) -> object_store::Result<PaginatedListResult> {
            self.requests.lock().unwrap().push(ListRequest {
                prefix: prefix.map(String::from),
                opts: opts.clone(),
            });
            let prefix = prefix.unwrap_or("");
            let budget = opts
                .max_keys
                .unwrap_or(self.page_bound)
                .min(self.page_bound);
            let mut result = ListResult {
                common_prefixes: Vec::new(),
                objects: Vec::new(),
            };
            let mut idx: usize = match &opts.page_token {
                Some(token) => token.parse().expect("a token this store minted"),
                None => 0,
            };

            while idx < self.keys.len() {
                if result.common_prefixes.len() + result.objects.len() >= budget {
                    return Ok(PaginatedListResult {
                        result,
                        page_token: Some(idx.to_string()),
                    });
                }
                // `Path::parse`, so that a key holding a character `Path::from` would encode
                // is reported under the name it was stored with. This is what the S3, GCS and
                // Azure clients do.
                let key = self.keys[idx].clone();
                idx += 1;
                let Some(rest) = key.strip_prefix(prefix) else {
                    continue;
                };
                match rest.find(DELIMITER) {
                    // A collapsed prefix, and everything behind it: a store reports the child
                    // directory once and skips the keys it stands for.
                    Some(end) => {
                        let child = format!("{prefix}{}", &rest[..=end]);
                        result.common_prefixes.push(Path::parse(&child).unwrap());
                        while idx < self.keys.len() && self.keys[idx].starts_with(&child) {
                            idx += 1;
                        }
                    }
                    None => result.objects.push(ObjectMeta {
                        location: Path::parse(&key).unwrap(),
                        last_modified: Utc::now(),
                        size: 1,
                        e_tag: None,
                        version: None,
                    }),
                }
            }

            Ok(PaginatedListResult {
                result,
                page_token: None,
            })
        }
    }

    struct TestStore {
        store: ObjectStore,
        requests: Arc<Mutex<Vec<ListRequest>>>,
    }

    impl TestStore {
        /// Every child of `dir`, taken a page at a time, which is how a caller walks a
        /// directory: the token ends the walk, never a short page.
        async fn walk(&self, dir: &str, limit: Option<usize>) -> Result<Vec<String>> {
            let mut names = Vec::new();
            let mut page_token = None;
            for _ in 0..100 {
                let page = self
                    .store
                    .read_dir_page(Path::from(dir), ReadDirOptions { page_token, limit })
                    .await?;
                names.extend(page_names(&page));
                page_token = page.page_token;
                if page_token.is_none() {
                    return Ok(names);
                }
            }
            panic!("the walk is not making progress: {names:?}")
        }

        async fn names(&self, dir: &str, limit: Option<usize>) -> Vec<String> {
            self.walk(dir, limit).await.unwrap()
        }

        /// The first page only, as a caller wanting a bounded number of children would take it.
        async fn first_page(&self, dir: &str, limit: Option<usize>) -> PaginatedListResult {
            self.store
                .read_dir_page(
                    Path::from(dir),
                    ReadDirOptions {
                        page_token: None,
                        limit,
                    },
                )
                .await
                .unwrap()
        }
    }

    /// The names of every child in a page, directories and files alike.
    fn page_names(page: &PaginatedListResult) -> Vec<String> {
        page.result
            .common_prefixes
            .iter()
            .chain(page.result.objects.iter().map(|object| &object.location))
            .map(|location| location.filename().unwrap().to_string())
            .collect()
    }

    async fn test_store(backend: Backend, keys: &[&str]) -> TestStore {
        paged_test_store(backend, keys, usize::MAX).await
    }

    /// A store over `keys`, listing them in the order given. `page_bound` is the store's own
    /// cap on a page, which only the pushdown backend has.
    async fn paged_test_store(backend: Backend, keys: &[&str], page_bound: usize) -> TestStore {
        let inner = Arc::new(InMemory::new());
        for key in keys {
            // `Path::parse`, so that a key holding a character `Path::from` would encode is
            // stored under the name it was given.
            inner
                .put(&Path::parse(key).unwrap(), PutPayload::from_static(b"x"))
                .await
                .unwrap();
        }
        #[allow(deprecated)]
        let params = ObjectStoreParams {
            object_store: Some((inner, url::Url::parse("memory:///").unwrap())),
            // Set because the deprecated hand-built path assumes nothing about the store it
            // was given, and set conservatively: nothing on the `read_dir_page` path reads it,
            // since the fallback sorts what it listed and the pushdown never compares keys.
            list_is_lexically_ordered: Some(false),
            ..Default::default()
        };
        let (store, _) = ObjectStore::from_uri_and_params(
            Arc::new(ObjectStoreRegistry::default()),
            "memory:///",
            &params,
        )
        .await
        .unwrap();
        let mut store = Arc::try_unwrap(store).unwrap();

        let requests = Arc::new(Mutex::new(Vec::new()));
        if backend == Pushdown {
            store.paginated_lister = Some(Arc::new(FakeListStore {
                keys: keys.iter().map(|key| key.to_string()).collect(),
                page_bound,
                requests: requests.clone(),
            }));
        }
        TestStore { store, requests }
    }

    const TABLES: &[&str] = &[
        "db/a.lance/_versions/1.manifest",
        "db/a.lance/data/1.lance",
        "db/b.lance/data/1.lance",
        "db/c.lance/data/1.lance",
        "db/loose.txt",
        "other/d.lance/data/1.lance",
    ];

    /// Walking a directory hands back every child exactly once, however the store resolves the
    /// listing and however small the pages are. That it holds whatever order the store lists
    /// in is [`test_an_unordered_store_is_still_paged`].
    #[rstest]
    #[case::whole_directory(TABLES, "db", vec!["a.lance", "b.lance", "c.lance", "loose.txt"])]
    #[case::empty_directory(TABLES, "nonexistent", vec![])]
    // A directory and the sibling that follows it: `foo/` and `foo0` are adjacent in key order
    // with nothing between them, so resuming past `foo`'s contents must not swallow `foo0`.
    #[case::the_sibling_after_a_directory(&["db/foo/inside", "db/foo0"], "db", vec!["foo", "foo0"])]
    // Siblings where one name is a prefix of another, which is where a page boundary is easiest
    // to get wrong: `foo/` and `foo-bar/` differ at `/` against `-`.
    #[case::a_prefix_shaped_sibling(&["db/foo/inside", "db/foo-bar/inside", "db/zzz.txt"], "db", vec!["foo", "foo-bar", "zzz.txt"])]
    // A store that keeps a marker object for a directory reports the directory itself when
    // that directory is listed. Dropping the marker must not also drop the progress the page
    // made, or a page holding nothing but the marker reads as the end of the listing.
    #[case::a_directory_marker(&["db/marked/", "db/marked/a.txt", "db/marked/b.txt"], "db/marked", vec!["a.txt", "b.txt"])]
    // A name holding a character `Path::from` would percent-encode is still reported, and
    // sorted, under the name it was stored with.
    #[case::an_encodable_name(&["db/az", "db/a~"], "db", vec!["az", "a~"])]
    #[tokio::test]
    async fn test_walking_a_directory_is_complete(
        #[values(FullListing, Pushdown)] backend: Backend,
        #[values(None, Some(1), Some(2), Some(3))] limit: Option<usize>,
        #[case] keys: &[&str],
        #[case] dir: &str,
        #[case] expected: Vec<&str>,
    ) {
        let store = test_store(backend, keys).await;

        // The order children come back in is the store's, so the walk is checked for holding
        // every child once rather than for holding them in one particular order.
        let mut listed = store.names(dir, limit).await;
        let seen = listed.clone();
        listed.sort();
        assert_eq!(listed, expected, "from {seen:?}");
    }

    /// A store that lists in no particular order — S3 Express — is still paged, because a
    /// continuation token is never compared to anything. This is the case a token spelled as a
    /// key could not serve.
    #[tokio::test]
    async fn test_an_unordered_store_is_still_paged() {
        let reversed: Vec<&str> = TABLES.iter().rev().copied().collect();
        let store = test_store(Pushdown, &reversed).await;

        let mut names = store.names("db", Some(1)).await;
        names.sort();

        assert_eq!(names, vec!["a.lance", "b.lance", "c.lance", "loose.txt"]);
        assert!(
            !store.requests.lock().unwrap().is_empty(),
            "the paginated lister should have been used"
        );
    }

    /// The point of the pushdown: a caller that wants one child of a directory makes one
    /// request, for one child, one level deep. A listing that quietly fell back to reading the
    /// whole directory would answer the same thing, so what tells the two apart is the request.
    #[tokio::test]
    async fn test_a_bounded_page_is_one_request_for_that_page() {
        let store = test_store(Pushdown, TABLES).await;

        let page = store.first_page("db", Some(1)).await;

        assert_eq!(page_names(&page).len(), 1);
        assert!(page.page_token.is_some());
        let requests = store.requests.lock().unwrap();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].prefix.as_deref(), Some("db/"));
        assert_eq!(requests[0].opts.max_keys, Some(1));
        // Without a delimiter the listing would be recursive rather than one level deep.
        assert_eq!(requests[0].opts.delimiter.as_deref(), Some(DELIMITER));
        // A continuation token is a position of its own, so no offset goes with it.
        assert_eq!(requests[0].opts.offset, None);
    }

    /// A backend that caps its pages below what was asked for hands back a short page with
    /// more to come. Only the token ends a walk, so a caller that stopped at a short page
    /// would report a directory as smaller than it is.
    #[tokio::test]
    async fn test_a_short_page_is_not_the_end_of_the_listing() {
        let store = paged_test_store(Pushdown, TABLES, 1).await;

        let page = store.first_page("db", Some(3)).await;

        assert_eq!(page_names(&page).len(), 1);
        assert!(
            page.page_token.is_some(),
            "the directory holds four children"
        );
        assert_eq!(
            store.names("db", Some(3)).await.len(),
            4,
            "the walk should still reach every child"
        );
    }

    /// A page of nothing is rejected rather than left to mean whatever the backend makes of it:
    /// pushing it down reports an empty directory, and a full listing ignores it. The rejection
    /// comes before the store is consulted, so one backend covers it.
    #[tokio::test]
    async fn test_zero_limit_is_rejected() {
        let store = test_store(Pushdown, TABLES).await;

        let err = store.walk("db", Some(0)).await.unwrap_err();

        assert!(matches!(err, Error::InvalidInput { .. }), "{err:?}");
        assert!(err.to_string().contains("limit must be at least 1"));
        assert!(store.requests.lock().unwrap().is_empty());
    }

    /// Child directories and child objects stay in the two lists a listing reports them in, so
    /// a caller that wants only one of the two — tables are directories — can take it.
    #[rstest]
    #[tokio::test]
    async fn test_directories_and_files_stay_apart(
        #[values(FullListing, Pushdown)] backend: Backend,
    ) {
        let store = test_store(backend, TABLES).await;

        let page = store.first_page("db", None).await;

        let mut directories: Vec<&str> = page
            .result
            .common_prefixes
            .iter()
            .map(|location| location.filename().unwrap())
            .collect();
        directories.sort();
        assert_eq!(directories, vec!["a.lance", "b.lance", "c.lance"]);
        let files: Vec<&str> = page
            .result
            .objects
            .iter()
            .map(|object| object.location.filename().unwrap())
            .collect();
        assert_eq!(files, vec!["loose.txt"]);
        // The metadata a listing reports for a child object survives the page.
        assert_eq!(page.result.objects[0].size, 1);
    }

    /// The pushdown path holds the backend directly, so it has to record its own IO. A listing
    /// invisible to `io_tracker` would also be invisible to the metrics and tracing layers that
    /// sit in the same chain.
    #[rstest]
    #[tokio::test]
    async fn test_listing_is_recorded_in_io_stats(
        #[values(FullListing, Pushdown)] backend: Backend,
    ) {
        let store = test_store(backend, TABLES).await;
        assert_eq!(store.store.io_tracker().stats().read_iops, 0);

        let _ = store.first_page("db", Some(2)).await;

        // The full listing reaches the store through its wrappers, which record it there.
        assert_eq!(store.store.io_tracker().stats().read_iops, 1);
    }
}
