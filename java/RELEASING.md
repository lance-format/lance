# Publishing Java packages

The `Build and publish Java packages` workflow publishes stable, numbered beta,
and RC releases to both Maven Central and the R2 repository at
`https://maven.lance.org`. The R2 job runs after Central succeeds and uploads the
same signed Central bundle. A release is complete only when both jobs succeed.

Pull requests and manual `dry_run` runs build the Java package without publishing
to either Maven repository.

## Repository configuration

Keep the existing Central and GPG credentials. Add these GitHub Actions secrets:

| Secret | Value |
| --- | --- |
| `R2_ACCESS_KEY_ID` | R2 S3 access key ID |
| `R2_SECRET_ACCESS_KEY` | R2 S3 secret access key |

Use an R2 User or Account API Token with **Object Read & Write** permission scoped
to the `lance-maven` bucket. Both provide the S3 credentials needed by this workflow.
A User token depends on its owner's account access; an Account token is independent
of an individual maintainer's membership.

Set these GitHub Actions variables:

| Variable | Value |
| --- | --- |
| `R2_ENDPOINT` | `https://047ddb663cacb353161dd02eaa6f9566.r2.cloudflarestorage.com` |
| `R2_BUCKET` | `lance-maven` |
| `R2_PUBLIC_URL` | `https://maven.lance.org` |

Keep the custom domain enabled. No CORS configuration is needed for Maven or
Gradle clients. Cache rules must respect the uploaded `Cache-Control` headers:
version files are immutable, while `maven-metadata.xml` and its checksums require
revalidation. Do not configure a lifecycle rule that deletes published versions.

## Retry an R2 failure

Use **Re-run failed jobs** on the original workflow run. The signed
`java-central-bundle` artifact is retained for 30 days, so the R2 job can retry
without rebuilding or republishing to Central. Do not re-run all jobs after
Central has accepted the version.

The publisher accepts identical existing files and rejects conflicting bytes.
It uploads all version files before merging metadata. Interrupted metadata or
checksum updates are repaired by retrying. The workflow serializes R2 jobs with
one concurrency group; manual publishers must not run alongside those jobs.
After upload, the job checks that the new POM is accessible through the public
domain.

## Consuming the R2 repository

Add this repository alongside Maven Central and use an explicit published
version of `org.lance:lance-core`:

```xml
<repositories>
    <repository>
        <id>lance</id>
        <url>https://maven.lance.org</url>
    </repository>
</repositories>
```

For Gradle:

```kotlin
repositories {
    mavenCentral()
    maven { url = uri("https://maven.lance.org") }
}
```
