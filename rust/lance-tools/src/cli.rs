// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use clap::{Args, Parser, Subcommand};
use lance_core::Result;

#[derive(Parser, Debug)]
#[command(
    name = "lance-tools",
    about = "Tools for interacting with Lance files and tables",
    version
)]
pub struct LanceToolsArgs {
    /// Subcommand to run
    #[command(subcommand)]
    command: LanceToolsCommand,
}

#[derive(Subcommand, Debug)]
pub enum LanceToolsCommand {
    /// Commands for interacting with Lance files.
    File(LanceFileArgs),
    /// Commands for interacting with Lance tables.
    Table(LanceTableArgs),
    /// Commands for inspecting and recovering Lance datasets.
    Dataset(LanceDatasetArgs),
}

/// Dataset inspection and recovery commands.
#[derive(Parser, Debug)]
pub struct LanceDatasetArgs {
    #[command(subcommand)]
    command: LanceDatasetCommand,
}

/// Operations on manifest-referenced files and dataset versions.
#[derive(Subcommand, Debug)]
pub enum LanceDatasetCommand {
    /// Check referenced data files for missing objects and incorrect sizes.
    VerifyDataFiles(LanceDatasetVerifyArgs),
    /// Remove entire fragments referencing explicitly selected data files.
    RepairManifest(LanceDatasetRepairArgs),
    /// Find the earliest retained version referencing a data file.
    LocateDataFile(LanceDatasetLocateArgs),
    /// Restore a historical version as a new latest version.
    Restore(LanceDatasetRestoreArgs),
}

/// Connection settings shared by dataset commands.
#[derive(Args, Debug)]
pub struct LanceDatasetSourceArgs {
    /// Dataset URI or local directory.
    #[arg(short = 's', long, required = true)]
    pub(crate) source: String,

    /// JSON object of string storage options, passed to the dataset's object stores.
    #[arg(long, value_name = "JSON")]
    pub(crate) storage_options: Option<String>,
}

/// Options for checking the files referenced by a dataset version.
#[derive(Args, Debug)]
pub struct LanceDatasetVerifyArgs {
    #[command(flatten)]
    pub(crate) dataset: LanceDatasetSourceArgs,

    /// Version to inspect (defaults to latest).
    #[arg(long)]
    pub(crate) version: Option<u64>,

    /// Also read current-format file metadata. Does not verify all data pages or blob payloads.
    #[arg(long)]
    pub(crate) deep: bool,

    /// Exit with an error if any file fails verification.
    #[arg(long)]
    pub(crate) fail: bool,
}

/// Explicit file selections for removing affected fragments.
#[derive(Args, Debug)]
pub struct LanceDatasetRepairArgs {
    #[command(flatten)]
    pub(crate) dataset: LanceDatasetSourceArgs,

    /// Exact manifest paths, `data/<path>`, or resolved object keys. Removes all rows
    /// in every matching fragment, including its other columns. Repeat or comma-separate.
    #[arg(
        long,
        required = true,
        value_delimiter = ',',
        value_parser = clap::builder::NonEmptyStringValueParser::new()
    )]
    pub(crate) remove_data_files: Vec<String>,

    /// List affected fragments and files without committing a new version.
    #[arg(long)]
    pub(crate) dry_run: bool,
}

/// File identity to search for in retained dataset versions.
#[derive(Args, Debug)]
pub struct LanceDatasetLocateArgs {
    #[command(flatten)]
    pub(crate) dataset: LanceDatasetSourceArgs,

    /// Exact manifest path, `data/<path>`, or resolved object key to find.
    #[arg(
        short = 'f',
        long,
        required = true,
        value_parser = clap::builder::NonEmptyStringValueParser::new()
    )]
    pub(crate) data_file: String,
}

/// Historical snapshot to publish as the latest dataset version.
#[derive(Args, Debug)]
pub struct LanceDatasetRestoreArgs {
    #[command(flatten)]
    pub(crate) dataset: LanceDatasetSourceArgs,

    /// Historical version to restore. The referenced files must still exist.
    #[arg(long)]
    pub(crate) version: u64,
}

#[derive(Parser, Debug)]
pub struct LanceFileArgs {
    #[command(subcommand)]
    command: LanceFileCommand,
}

#[derive(Subcommand, Debug)]
pub enum LanceFileCommand {
    /// Display Lance file metadata.
    Meta(LanceFileMetaArgs),
}

#[derive(Parser, Debug)]
pub struct LanceTableArgs {
    #[command(subcommand)]
    command: LanceTableCommand,
}

#[derive(Subcommand, Debug)]
pub enum LanceTableCommand {
    /// Display Lance table manifest contents.
    Manifest(LanceTableManifestArgs),
}

#[derive(Args, Debug)]
pub struct LanceFileMetaArgs {
    // The Lance file to examine.
    #[arg(short = 's', long, value_name = "source")]
    pub(crate) source: String,
}

#[derive(Args, Debug)]
pub struct LanceTableManifestArgs {
    /// The Lance manifest file to examine.
    #[arg(short = 's', long, value_name = "source")]
    pub(crate) source: String,
}

impl LanceToolsArgs {
    pub async fn run(&self, writer: impl std::io::Write) -> Result<()> {
        match &self.command {
            LanceToolsCommand::File(args) => match &args.command {
                LanceFileCommand::Meta(args) => crate::meta::show_file_meta(writer, args).await,
            },
            LanceToolsCommand::Dataset(args) => match &args.command {
                LanceDatasetCommand::VerifyDataFiles(args) => {
                    crate::dataset::verify_data_files(writer, args).await
                }
                LanceDatasetCommand::RepairManifest(args) => {
                    crate::dataset::repair_manifest(writer, args).await
                }
                LanceDatasetCommand::LocateDataFile(args) => {
                    crate::dataset::locate_data_file(writer, args).await
                }
                LanceDatasetCommand::Restore(args) => crate::dataset::restore(writer, args).await,
            },
            LanceToolsCommand::Table(args) => match &args.command {
                LanceTableCommand::Manifest(args) => {
                    crate::manifest::show_table_manifest(writer, args).await
                }
            },
        }
    }
}
