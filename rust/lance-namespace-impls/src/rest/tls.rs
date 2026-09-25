// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! TLS configuration for the REST namespace client.
//!
//! Client certificates are usually short-lived - SPIFFE/Istio, Vault PKI and cert-manager all
//! re-mint them onto disk long before the process using them restarts - and a
//! `reqwest::Identity` cannot be swapped inside a live `reqwest::Client`. [`ReloadableClient`]
//! therefore re-reads the configured PEM files on the request path and publishes a rebuilt
//! client when their content changes.

use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use arc_swap::ArcSwap;
use lance_core::{Error, Result};
use lance_namespace::error::NamespaceError;

/// TLS options of the REST client, including the paths its material is loaded from.
#[derive(Clone, Debug)]
pub(super) struct TlsConfig {
    pub(super) cert_file: Option<String>,
    pub(super) key_file: Option<String>,
    pub(super) ssl_ca_cert: Option<String>,
    /// When false, the hostname in the server certificate is not verified.
    pub(super) assert_hostname: bool,
}

impl TlsConfig {
    /// Whether any PEM file is configured, i.e. whether there is anything to reload.
    fn watches_files(&self) -> bool {
        (self.cert_file.is_some() && self.key_file.is_some()) || self.ssl_ca_cert.is_some()
    }

    fn load_identity(&self) -> Result<Option<(reqwest::Identity, u64)>> {
        // mTLS needs both halves of the identity; a certificate without a key is ignored.
        let (Some(cert_file), Some(key_file)) = (&self.cert_file, &self.key_file) else {
            return Ok(None);
        };

        let cert = read_pem_file("certificate", cert_file)?;
        let key = read_pem_file("private key", key_file)?;
        let pem = [cert, key].concat();

        let identity = reqwest::Identity::from_pem(&pem).map_err(|e| {
            tls_config_error(format!(
                "Failed to load the client identity from certificate '{cert_file}' \
                 and key '{key_file}': {e}"
            ))
        })?;

        Ok(Some((identity, digest(&pem))))
    }

    fn load_root_cert(&self) -> Result<Option<(reqwest::Certificate, u64)>> {
        let Some(ca_cert_file) = &self.ssl_ca_cert else {
            return Ok(None);
        };

        let pem = read_pem_file("CA certificate", ca_cert_file)?;
        let root_cert = reqwest::Certificate::from_pem(&pem).map_err(|e| {
            tls_config_error(format!(
                "Failed to load the CA certificate '{ca_cert_file}': {e}"
            ))
        })?;

        Ok(Some((root_cert, digest(&pem))))
    }

    fn build_client(&self, material: TlsMaterial) -> Result<reqwest::Client> {
        // Build the client WITHOUT default headers - they are applied per-request.
        let mut client_builder =
            reqwest::Client::builder().danger_accept_invalid_hostnames(!self.assert_hostname);

        if let Some(identity) = material.identity {
            client_builder = client_builder.identity(identity);
        }

        if let Some(root_cert) = material.root_cert {
            client_builder = client_builder.add_root_certificate(root_cert);
        }

        client_builder
            .build()
            .map_err(|e| tls_config_error(format!("Failed to build the HTTP client: {e}")))
    }

    /// Build the initial client, dropping only the material it cannot use.
    ///
    /// A CA certificate is parsed when the client is built rather than when it is read, so it
    /// can still fail here, and construction has no working client to fall back to - drop it on
    /// its own rather than with the client identity. The digest survives only a build that used
    /// everything that loaded, so whatever was dropped is retried on the next check.
    fn build_best_effort(&self, material: TlsMaterial) -> (reqwest::Client, Option<u64>) {
        let digest = material.digest;
        let identity = material.identity.clone();

        let error = match self.build_client(material) {
            Ok(client) => return (client, digest),
            Err(e) => e,
        };
        log::warn!(
            "Failed to build the HTTP client of the REST namespace with the configured TLS \
             material, continuing without the CA certificate: {error}"
        );

        let client = self
            .build_client(TlsMaterial {
                identity,
                root_cert: None,
                digest: None,
            })
            .unwrap_or_else(|e| {
                log::warn!(
                    "Failed to build the HTTP client of the REST namespace, continuing without \
                     any TLS configuration: {e}"
                );
                reqwest::Client::new()
            });

        (client, None)
    }
}

/// One generation of the TLS material on disk, parsed and ready to apply to a client.
struct TlsMaterial {
    identity: Option<reqwest::Identity>,
    root_cert: Option<reqwest::Certificate>,
    /// Digest of the PEM bytes the pieces were parsed from, used to detect rotation.
    ///
    /// Content, not modification time: rotation is commonly an atomic rename or a symlink swap.
    /// `None` when a configured piece failed to load, so that the next check retries it even if
    /// the files did not change again in the meantime.
    digest: Option<u64>,
}

impl TlsMaterial {
    fn new(
        identity: Option<(reqwest::Identity, u64)>,
        root_cert: Option<(reqwest::Certificate, u64)>,
    ) -> Self {
        // The fixed order keeps the identity and the CA certificate from being interchangeable.
        let mut hasher = DefaultHasher::new();
        identity
            .as_ref()
            .map(|(_, digest)| digest)
            .hash(&mut hasher);
        root_cert
            .as_ref()
            .map(|(_, digest)| digest)
            .hash(&mut hasher);

        Self {
            identity: identity.map(|(identity, _)| identity),
            root_cert: root_cert.map(|(root_cert, _)| root_cert),
            digest: Some(hasher.finish()),
        }
    }

    /// Load every configured piece, failing unless all of them load: half of a rotated
    /// generation is material the peer never issued together.
    fn load(tls: &TlsConfig) -> Result<Self> {
        Ok(Self::new(tls.load_identity()?, tls.load_root_cert()?))
    }

    /// Load every configured piece, skipping the ones that fail.
    ///
    /// Construction has no client in use to fall back to, so one unusable file must not drop the
    /// pieces that work. The missing digest makes the next check retry what was skipped.
    fn load_best_effort(tls: &TlsConfig) -> Self {
        let identity = tls.load_identity();
        let root_cert = tls.load_root_cert();
        let complete = identity.is_ok() && root_cert.is_ok();

        let mut material = Self::new(skip_unloadable(identity), skip_unloadable(root_cert));
        if !complete {
            material.digest = None;
        }
        material
    }
}

/// Digest of the material the current client was built from, and when disk was last read.
struct ReloadState {
    digest: Option<u64>,
    last_checked: Instant,
}

/// A `reqwest::Client` that is rebuilt when the TLS material it presents changes on disk.
pub(super) struct ReloadableClient {
    client: ArcSwap<reqwest::Client>,
    tls: TlsConfig,
    /// Minimum delay between two disk checks. `None` disables reloading entirely;
    /// `Some(Duration::ZERO)` checks on every request.
    reload_interval: Option<Duration>,
    state: Mutex<ReloadState>,
}

impl ReloadableClient {
    /// Build the initial client from the material currently on disk.
    ///
    /// Material that fails to load is logged and skipped, which keeps
    /// `RestNamespaceBuilder::build` infallible as it has always been.
    pub(super) fn new(tls: TlsConfig, reload_interval: Option<Duration>) -> Self {
        let (client, digest) = tls.build_best_effort(TlsMaterial::load_best_effort(&tls));

        // Without any PEM file configured there is nothing on disk to watch.
        let reload_interval = reload_interval.filter(|_| tls.watches_files());

        Self {
            client: ArcSwap::from_pointee(client),
            tls,
            reload_interval,
            state: Mutex::new(ReloadState {
                digest,
                last_checked: Instant::now(),
            }),
        }
    }

    /// The client to use for the next request, cloned because a reload can replace it at any
    /// point. Cloning is cheap: `reqwest::Client` is a handle around an `Arc`.
    pub(super) fn current(&self) -> reqwest::Client {
        self.reload_if_due();
        let client = self.client.load();
        reqwest::Client::clone(&client)
    }

    /// The interval reloading actually runs at, `None` when the client never reloads.
    #[cfg(test)]
    pub(super) fn reload_interval(&self) -> Option<Duration> {
        self.reload_interval
    }

    /// Re-read the PEM files if the check interval has elapsed, and replace the client if their
    /// content changed.
    ///
    /// Replacing the client drops its connection pool, which is correct - pooled connections
    /// were authenticated with the previous certificate - and is why it is gated on a real change.
    fn reload_if_due(&self) {
        let Some(interval) = self.reload_interval else {
            return;
        };

        // Another request is already checking; whoever loses the race is one interval behind.
        let Ok(mut state) = self.state.try_lock() else {
            return;
        };
        if state.last_checked.elapsed() < interval {
            return;
        }
        state.last_checked = Instant::now();

        let material = match TlsMaterial::load(&self.tls) {
            Ok(material) => material,
            Err(e) => {
                log::warn!(
                    "Failed to load the TLS material of the REST namespace client from disk, \
                     keeping the current client: {e}"
                );
                return;
            }
        };

        let digest = material.digest;
        if state.digest == digest {
            return;
        }

        match self.tls.build_client(material) {
            Ok(client) => {
                self.client.store(Arc::new(client));
                state.digest = digest;
                log::info!("Reloaded the TLS material of the REST namespace client from disk");
            }
            Err(e) => log::warn!(
                "Rotated TLS material of the REST namespace client is unusable, \
                 keeping the current client: {e}"
            ),
        }
    }
}

/// Read a PEM file, reporting failures with the path and the role the file plays.
fn read_pem_file(role: &str, path: &str) -> Result<Vec<u8>> {
    std::fs::read(path)
        .map_err(|e| tls_config_error(format!("Failed to read the TLS {role} '{path}': {e}")))
}

fn digest(pem: &[u8]) -> u64 {
    let mut hasher = DefaultHasher::new();
    pem.hash(&mut hasher);
    hasher.finish()
}

/// Report a piece of TLS material that could not be loaded and continue without it.
fn skip_unloadable<T>(loaded: Result<Option<T>>) -> Option<T> {
    loaded.unwrap_or_else(|e| {
        log::warn!(
            "Failed to load part of the TLS configuration of the REST namespace client, \
             continuing without it: {e}"
        );
        None
    })
}

fn tls_config_error(message: String) -> Error {
    NamespaceError::InvalidInput { message }.into()
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::path::Path;

    use rstest::rstest;
    use tempfile::tempdir;

    /// Self-signed certificate and matching key, only ever used to exercise PEM loading.
    const CERT_A: &str = r#"-----BEGIN CERTIFICATE-----
MIIBhTCCASugAwIBAgIUFnETCuqfjMXeoqoLZMxXObQElu8wCgYIKoZIzj0EAwIw
FzEVMBMGA1UEAwwMbGFuY2UtdGVzdC1hMCAXDTI2MDgxMDE5NTU1N1oYDzIxMjYw
NzE3MTk1NTU3WjAXMRUwEwYDVQQDDAxsYW5jZS10ZXN0LWEwWTATBgcqhkjOPQIB
BggqhkjOPQMBBwNCAATBBAzMgIrkusG3gOZuyboh+RKancco6xC03O2mMyykqq52
JRQnzlZm37BomXPtFlHW9mU7su1KcIBEKgRZfLnPo1MwUTAdBgNVHQ4EFgQUwKTV
fIJnAXSuL2imQuMvyQfsK2MwHwYDVR0jBBgwFoAUwKTVfIJnAXSuL2imQuMvyQfs
K2MwDwYDVR0TAQH/BAUwAwEB/zAKBggqhkjOPQQDAgNIADBFAiB6yQpGVALMCLxm
8ekf9DmNDuSv36JRsG7l6yS/d4yqPgIhAPv4bKcpKGqE8PsQVXhnWweltcUrnJ+D
H027pEBl+pKJ
-----END CERTIFICATE-----
"#;

    const KEY_A: &str = r#"-----BEGIN PRIVATE KEY-----
MIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBG0wawIBAQQge8mgs5FZcchWUNkr
JrJWhDOe7eOmgB03AFrpsQo1I5uhRANCAATBBAzMgIrkusG3gOZuyboh+RKancco
6xC03O2mMyykqq52JRQnzlZm37BomXPtFlHW9mU7su1KcIBEKgRZfLnP
-----END PRIVATE KEY-----
"#;

    /// A second self-signed pair, standing in for a re-minted certificate.
    const CERT_B: &str = r#"-----BEGIN CERTIFICATE-----
MIIBhDCCASugAwIBAgIUfo81kW2IN2pBuO3jhwazB12Pd04wCgYIKoZIzj0EAwIw
FzEVMBMGA1UEAwwMbGFuY2UtdGVzdC1iMCAXDTI2MDgxMDE5NTU1N1oYDzIxMjYw
NzE3MTk1NTU3WjAXMRUwEwYDVQQDDAxsYW5jZS10ZXN0LWIwWTATBgcqhkjOPQIB
BggqhkjOPQMBBwNCAARdbZsJo7FrzCRTFQQkGUNw3mllD0D2vlUSTstZkV9DqFsq
GgtQZsdPpD/ndAKCeILH618omGtfieRWbNEfmLnDo1MwUTAdBgNVHQ4EFgQUFERt
Mj+ePIr2LK2oV37TgpnKDvowHwYDVR0jBBgwFoAUFERtMj+ePIr2LK2oV37TgpnK
DvowDwYDVR0TAQH/BAUwAwEB/zAKBggqhkjOPQQDAgNHADBEAiAPBW3FOuyZbNnL
Jccb7r4yzAOWl25IX8mwiqL+IeMt4wIgSain0EVWElOQcBSaGcQTCmeB+v1yqg1a
AW7D1q0IGho=
-----END CERTIFICATE-----
"#;

    const KEY_B: &str = r#"-----BEGIN PRIVATE KEY-----
MIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBG0wawIBAQQguOhONlmNsse91H8r
XYris7tqFIImbS16joxzji/SDT2hRANCAARdbZsJo7FrzCRTFQQkGUNw3mllD0D2
vlUSTstZkV9DqFsqGgtQZsdPpD/ndAKCeILH618omGtfieRWbNEfmLnD
-----END PRIVATE KEY-----
"#;

    /// Write `cert` and `key` to the fixed paths a client is configured with, the way a
    /// certificate agent re-mints them in place.
    fn write_identity(dir: &Path, cert: &str, key: &str) -> TlsConfig {
        let cert_file = dir.join("cert.pem");
        let key_file = dir.join("key.pem");
        std::fs::write(&cert_file, cert).unwrap();
        std::fs::write(&key_file, key).unwrap();

        TlsConfig {
            cert_file: Some(cert_file.to_str().unwrap().to_string()),
            key_file: Some(key_file.to_str().unwrap().to_string()),
            ssl_ca_cert: None,
            assert_hostname: true,
        }
    }

    /// Point `tls` at a CA certificate in `dir`, writing `contents` there when there are any.
    fn with_ca_cert(dir: &Path, tls: &mut TlsConfig, contents: Option<&str>) {
        let ca_cert_file = dir.join("ca.pem");
        if let Some(contents) = contents {
            std::fs::write(&ca_cert_file, contents).unwrap();
        }
        tls.ssl_ca_cert = Some(ca_cert_file.to_str().unwrap().to_string());
    }

    #[rstest]
    #[case::interval_elapsed(Some(Duration::ZERO), true)]
    #[case::within_the_interval(Some(Duration::from_secs(300)), false)]
    #[case::reloading_disabled(None, false)]
    fn test_rotation_is_picked_up_once_the_interval_elapses(
        #[case] reload_interval: Option<Duration>,
        #[case] expected: bool,
    ) {
        let dir = tempdir().unwrap();
        let tls = write_identity(dir.path(), CERT_A, KEY_A);

        let reloadable = ReloadableClient::new(tls, reload_interval);
        let initial = reloadable.client.load_full();

        write_identity(dir.path(), CERT_B, KEY_B);
        reloadable.current();

        let rebuilt = !Arc::ptr_eq(&initial, &reloadable.client.load_full());
        assert_eq!(rebuilt, expected, "the client was rebuilt: {rebuilt}");
    }

    #[test]
    fn test_unchanged_certificate_keeps_client() {
        let dir = tempdir().unwrap();
        let tls = write_identity(dir.path(), CERT_A, KEY_A);

        let reloadable = ReloadableClient::new(tls, Some(Duration::ZERO));
        let initial = reloadable.client.load_full();

        // Rewriting the same bytes changes modification times but not the identity, so the
        // client - and its connection pool - must survive.
        write_identity(dir.path(), CERT_A, KEY_A);
        reloadable.current();
        reloadable.current();

        assert!(
            Arc::ptr_eq(&initial, &reloadable.client.load_full()),
            "unchanged material should not rebuild the client"
        );
    }

    #[test]
    fn test_reload_disabled_without_files_to_watch() {
        let tls = TlsConfig {
            cert_file: None,
            key_file: None,
            ssl_ca_cert: None,
            assert_hostname: true,
        };

        let reloadable = ReloadableClient::new(tls, Some(Duration::ZERO));
        assert!(reloadable.reload_interval.is_none());
    }

    #[test]
    fn test_unusable_rotated_material_keeps_client() {
        let dir = tempdir().unwrap();
        let tls = write_identity(dir.path(), CERT_A, KEY_A);

        let reloadable = ReloadableClient::new(tls, Some(Duration::ZERO));
        let initial = reloadable.client.load_full();

        // A truncated read, as can happen when a certificate is written in place, must not
        // replace a working identity with one the server would reject.
        write_identity(dir.path(), &CERT_B[..CERT_B.len() / 2], KEY_B);
        reloadable.current();
        assert!(
            Arc::ptr_eq(&initial, &reloadable.client.load_full()),
            "unusable material should not replace the client"
        );

        // Once the complete material lands, the next check picks it up.
        write_identity(dir.path(), CERT_B, KEY_B);
        reloadable.current();
        assert!(
            !Arc::ptr_eq(&initial, &reloadable.client.load_full()),
            "the retry should pick up the complete material"
        );
    }

    /// One unusable file must not drop the piece that loaded: a stale `ssl_ca_cert` would
    /// otherwise turn into a rejected mTLS handshake.
    #[rstest]
    #[case::unreadable_ca("ca.pem", None)]
    #[case::unreadable_key("key.pem", None)]
    #[case::malformed_key("key.pem", Some("-----BEGIN PRIVATE KEY-----\nnot a key\n"))]
    fn test_an_unloadable_file_keeps_the_other_piece(
        #[case] broken: &str,
        #[case] contents: Option<&str>,
    ) {
        let dir = tempdir().unwrap();
        let mut tls = write_identity(dir.path(), CERT_A, KEY_A);
        with_ca_cert(dir.path(), &mut tls, Some(CERT_B));

        let broken_file = dir.path().join(broken);
        match contents {
            Some(contents) => std::fs::write(&broken_file, contents).unwrap(),
            None => std::fs::remove_file(&broken_file).unwrap(),
        }

        let material = TlsMaterial::load_best_effort(&tls);
        let ca_is_broken = broken == "ca.pem";
        assert_eq!(material.identity.is_some(), ca_is_broken);
        assert_eq!(material.root_cert.is_some(), !ca_is_broken);
        assert!(
            material.digest.is_none(),
            "an incomplete load must be retried"
        );

        // Reloading holds the generation to a higher bar: it has a working client to keep.
        let Err(error) = TlsMaterial::load(&tls) else {
            panic!("an unloadable file should fail the atomic load");
        };
        let error = error.to_string();
        assert!(
            error.contains(broken),
            "the error should name the file: {error}"
        );
    }

    /// A CA certificate is only parsed when the client is built, so an unusable one survives the
    /// load and fails later. Construction drops it without the client identity.
    #[test]
    fn test_malformed_ca_certificate_is_dropped_on_its_own() {
        let dir = tempdir().unwrap();
        let mut tls = write_identity(dir.path(), CERT_A, KEY_A);
        with_ca_cert(
            dir.path(),
            &mut tls,
            Some("-----BEGIN CERTIFICATE-----\nnot a certificate\n"),
        );

        let material = TlsMaterial::load_best_effort(&tls);
        let identity = material.identity.clone();
        assert!(tls.build_client(material).is_err());

        // What the retry falls back to.
        assert!(identity.is_some());
        assert!(
            tls.build_client(TlsMaterial {
                identity,
                root_cert: None,
                digest: None,
            })
            .is_ok()
        );

        let (_, digest) = tls.build_best_effort(TlsMaterial::load_best_effort(&tls));
        assert!(
            digest.is_none(),
            "material dropped at build time must be retried"
        );
    }

    #[test]
    fn test_incomplete_initial_load_is_retried() {
        let dir = tempdir().unwrap();
        let mut tls = write_identity(dir.path(), CERT_A, KEY_A);
        with_ca_cert(dir.path(), &mut tls, None);
        let ca_cert_file = tls.ssl_ca_cert.clone().unwrap();

        let reloadable = ReloadableClient::new(tls, Some(Duration::ZERO));
        let initial = reloadable.client.load_full();
        assert!(reloadable.state.lock().unwrap().digest.is_none());

        reloadable.current();
        assert!(
            Arc::ptr_eq(&initial, &reloadable.client.load_full()),
            "the CA certificate is still missing, there is nothing to load"
        );

        // A piece missing at startup is not permanent: once the file lands, the next check
        // upgrades to a client built from the complete material.
        std::fs::write(&ca_cert_file, CERT_B).unwrap();
        reloadable.current();
        assert!(!Arc::ptr_eq(&initial, &reloadable.client.load_full()));
        assert!(reloadable.state.lock().unwrap().digest.is_some());
    }

    #[test]
    fn test_digest_tracks_content() {
        let dir = tempdir().unwrap();
        let tls = write_identity(dir.path(), CERT_A, KEY_A);

        let initial = TlsMaterial::load(&tls).unwrap().digest;
        assert_eq!(initial, TlsMaterial::load(&tls).unwrap().digest);

        write_identity(dir.path(), CERT_B, KEY_B);
        assert_ne!(initial, TlsMaterial::load(&tls).unwrap().digest);

        // The same bytes are different material in the two roles, so swapping one for the other
        // has to register as a change.
        let identity = reqwest::Identity::from_pem(&[CERT_A, KEY_A].concat().into_bytes()).unwrap();
        let root_cert = reqwest::Certificate::from_pem(CERT_A.as_bytes()).unwrap();
        assert_ne!(
            TlsMaterial::new(Some((identity, 1)), None).digest,
            TlsMaterial::new(None, Some((root_cert, 1))).digest,
        );
    }
}
