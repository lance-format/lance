// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Whether a pair conflicts because of *what it touches*, or only because of
//! *what it is*.
//!
//! The grid in [`super::cases`] stages every operation against fragment 0, field
//! `c`, so every fragment-scoped pair in it overlaps completely. That makes it a
//! good detector of under-rejection — a pair that should have conflicted and did
//! not — and a poor one for the opposite mistake. A conflict detector that
//! rejected every pair outright would score nearly as well on it as one that
//! reasons about footprints, because the matrix never asks a question whose
//! answer depends on the footprint.
//!
//! This module asks it, at both granularities a footprint has:
//!
//! - [`Footprint::OtherFragment`] — `theirs` works on fragment 1, which `ours`
//!   never touches. Visible in the fragment ids a transaction carries.
//! - [`Footprint::OtherField`] — `theirs` works on the *same* fragment, but on
//!   column `d` while `ours` works on `c`. Visible only to something that
//!   reasons about fields, so a resolver can distinguish the fragment case and
//!   still miss this one.
//!
//! A cell where the answers differ is a pair legacy resolves by footprint. A
//! cell where they agree is one it resolves by opcode alone — which may be
//! correct, or may be conservatism worth revisiting, but either way it is now
//! recorded rather than unasked.
//!
//! Only the fragment-scoped operations appear, and the field column applies only
//! to the field-scoped ones; everything else produces the same transaction
//! whichever field it is pointed at, so asserting there would cost runtime and
//! prove nothing. See [`Scenario::is_fragment_scoped`] and
//! [`Scenario::is_field_scoped`].

use std::collections::HashMap;
use std::sync::LazyLock;

use rstest::rstest;

use super::cases::{known_bug, run};
use super::scenarios::{Footprint, Scenario};
use super::{Isolation, Outcome};
use super::{invariants, oracle};

/// What one ordered pair does at one footprint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Cell {
    /// The footprint does not apply to this `theirs`, so there is nothing to
    /// run: staging it there would produce the same transaction as `Same`.
    NotApplicable,
    /// Excluded because what the code does today is wrong; see
    /// `cases::KNOWN_BUGS`.
    Excluded,
    /// The observed outcome.
    Expected(Outcome),
}

impl Cell {
    fn symbol(self) -> char {
        match self {
            Self::NotApplicable => '-',
            Self::Excluded => '!',
            Self::Expected(outcome) => outcome.symbol(),
        }
    }

    fn from_symbol(symbol: &str) -> Self {
        match symbol {
            "-" => Self::NotApplicable,
            "!" => Self::Excluded,
            other => Self::Expected(Outcome::from_symbol(other)),
        }
    }
}

/// What each fragment-scoped ordered pair does at each footprint, stated under
/// [`Isolation::Legacy`].
///
/// Rows are `ours`, columns are `theirs`, keyed by [`Scenario::code`]. Each cell
/// is `same/field/fragment`: what happens when `theirs` touches exactly what
/// `ours` does, when it touches a different column of the same fragment, and
/// when it touches a different fragment. `L` lands, `R` retryable, `X`
/// incompatible, `!` excluded as a known bug, `-` not applicable.
///
/// A cell reading `R/L/L` is fully footprint-sensitive. `R/R/L` distinguishes
/// fragments but not fields. `R/R/R` rejects on opcode alone.
///
/// Regenerate with the `discover_footprints` test below.
const FOOTPRINT_MATRIX: &str = "\
                       | dl      df      ur      uc      ov      dr      ci      rw
delete                 | R/-/L   R/-/L   L/-/L   R/R/L   !/!/!   R/R/L   L/-/L   R/-/L
delete_whole_fragment  | R/-/L   R/-/L   R/-/L   R/R/L   L/L/L   R/R/L   L/-/L   R/-/L
update_rewrite_rows    | L/-/L   R/-/L   R/-/L   R/R/L   R/R/L   R/R/L   L/-/L   R/-/L
update_rewrite_columns | R/-/L   R/-/L   R/-/L   R/R/L   R/L/L   R/R/L   L/-/L   R/-/L
data_overlay           | L/-/L   R/-/L   R/-/L   L/L/L   L/L/L   L/L/L   L/-/L   R/-/L
data_replacement       | L/-/L   X/-/L   R/-/L   R/L/L   L/L/L   R/L/L   L/-/L   R/-/L
create_index           | L/-/L   L/-/L   L/-/L   L/L/L   L/L/L   L/L/L   R/-/R   R/-/L
rewrite                | R/-/L   R/-/L   R/-/L   R/R/L   R/R/L   R/R/L   R/-/L   R/-/L
";

/// The level [`FOOTPRINT_MATRIX`] was observed under.
const FOOTPRINT_ISOLATION: Isolation = Isolation::Legacy;

/// Parsed form of [`FOOTPRINT_MATRIX`], keyed by pair then footprint.
static FOOTPRINT_CELLS: LazyLock<HashMap<(Scenario, Scenario, Footprint), Cell>> =
    LazyLock::new(parse);

/// The operations this module covers, in matrix order.
fn scoped() -> Vec<Scenario> {
    Scenario::ALL
        .into_iter()
        .filter(|scenario| scenario.is_fragment_scoped())
        .collect()
}

/// What a cell must be without running it, if anything. `None` means run it.
fn predetermined(theirs: Scenario, footprint: Footprint, bug: bool) -> Option<Cell> {
    if footprint == Footprint::OtherField && !theirs.is_field_scoped() {
        Some(Cell::NotApplicable)
    } else if bug {
        Some(Cell::Excluded)
    } else {
        None
    }
}

fn parse() -> HashMap<(Scenario, Scenario, Footprint), Cell> {
    let scenario_by = |field: fn(Scenario) -> &'static str, value: &str| {
        Scenario::ALL
            .into_iter()
            .find(|s| field(*s) == value)
            .unwrap_or_else(|| panic!("the footprint matrix names an unknown scenario {value:?}"))
    };

    let mut lines = FOOTPRINT_MATRIX
        .lines()
        .filter(|line| !line.trim().is_empty());
    let header = lines.next().expect("the footprint matrix has a header row");
    let columns: Vec<Scenario> = header
        .split_once('|')
        .expect("the header row is delimited by `|`")
        .1
        .split_whitespace()
        .map(|code| scenario_by(Scenario::code, code))
        .collect();
    assert_eq!(
        columns,
        scoped(),
        "the footprint matrix columns must be the fragment-scoped scenarios, in order",
    );

    let mut cells = HashMap::new();
    for line in lines {
        let (label, rest) = line
            .split_once('|')
            .expect("a footprint matrix row is delimited by `|`");
        let ours = scenario_by(Scenario::name, label.trim());
        let entries: Vec<&str> = rest.split_whitespace().collect();
        assert_eq!(
            entries.len(),
            columns.len(),
            "footprint matrix row {} has {} cells but there are {} columns",
            ours.name(),
            entries.len(),
            columns.len(),
        );
        for (theirs, entry) in columns.iter().zip(entries) {
            let symbols: Vec<&str> = entry.split('/').collect();
            assert_eq!(
                symbols.len(),
                Footprint::ALL.len(),
                "cell {entry:?} does not name all {} footprints",
                Footprint::ALL.len(),
            );
            for (footprint, symbol) in Footprint::ALL.into_iter().zip(symbols) {
                assert!(
                    cells
                        .insert((ours, *theirs, footprint), Cell::from_symbol(symbol))
                        .is_none(),
                    "the footprint matrix has two rows for {}",
                    ours.name(),
                );
            }
        }
    }
    cells
}

/// One row of the footprint matrix: `ours` against every fragment-scoped
/// `theirs`, at every footprint.
#[rstest]
#[case::delete(Scenario::Delete)]
#[case::delete_whole_fragment(Scenario::DeleteWholeFragment)]
#[case::update_rewrite_rows(Scenario::UpdateRewriteRows)]
#[case::update_rewrite_columns(Scenario::UpdateRewriteColumns)]
#[case::data_overlay(Scenario::DataOverlay)]
#[case::data_replacement(Scenario::DataReplacement)]
#[case::create_index(Scenario::CreateIndex)]
#[case::rewrite(Scenario::Rewrite)]
#[tokio::test]
async fn footprint_row(#[case] ours: Scenario) {
    for theirs in scoped() {
        for footprint in Footprint::ALL {
            let cell = *FOOTPRINT_CELLS
                .get(&(ours, theirs, footprint))
                .unwrap_or_else(|| {
                    panic!(
                        "({}, {}) at {} is missing from the footprint matrix; re-run \
                         `discover_footprints`",
                        ours.name(),
                        theirs.name(),
                        footprint.name(),
                    )
                });
            let Cell::Expected(expected) = cell else {
                continue;
            };
            let context = format!(
                "({} over {} touching {})",
                ours.name(),
                theirs.name(),
                footprint.name(),
            );
            let (outcome, landed) = run(ours, theirs, footprint).await;
            assert_eq!(
                outcome, expected,
                "{context}: expected {expected:?} but observed {outcome:?}",
            );
            if let Some((before, after)) = landed {
                invariants::check_all(&before, &after, ours, &context).await;
                // Boxed for the same reason as in `cases`: a second commit
                // sequence makes the future large enough to trip clippy.
                Box::pin(oracle::check(
                    FOOTPRINT_ISOLATION,
                    ours,
                    theirs,
                    footprint,
                    &after,
                    &context,
                ))
                .await;
            }
        }
    }
}

/// Every pair must be covered at every footprint, and the cells that are not
/// asserted must be ones the harness can justify not asserting.
///
/// `-` has to line up with [`Scenario::is_field_scoped`] and `!` with
/// `KNOWN_BUGS`, or the grid is quietly skipping cells. `KNOWN_BUGS` is keyed on
/// the pair rather than the footprint, so a bug that only reproduces when the
/// footprints overlap exempts the disjoint cells too; requiring all three to be
/// excluded together keeps that visible rather than letting it pass unnoticed.
#[test]
fn footprint_matrix_is_total() {
    for ours in scoped() {
        for theirs in scoped() {
            let bug = known_bug(ours, theirs).is_some();
            for footprint in Footprint::ALL {
                let cell = *FOOTPRINT_CELLS
                    .get(&(ours, theirs, footprint))
                    .unwrap_or_else(|| {
                        panic!(
                            "({}, {}) at {} is missing from the footprint matrix",
                            ours.name(),
                            theirs.name(),
                            footprint.name(),
                        )
                    });
                match (cell, predetermined(theirs, footprint, bug)) {
                    (Cell::Expected(_), None) => {}
                    (actual, Some(wanted)) if actual == wanted => {}
                    (actual, _) => panic!(
                        "({}, {}) at {} is {actual:?}, which does not match what the harness \
                         says about it: field-scoped {}, known bug {bug}",
                        ours.name(),
                        theirs.name(),
                        footprint.name(),
                        theirs.is_field_scoped(),
                    ),
                }
            }
        }
    }
}

/// Print the observed outcome for every fragment-scoped pair at every footprint,
/// as a grid.
///
/// Not an assertion — this regenerates [`FOOTPRINT_MATRIX`]. Run it with:
///
/// ```text
/// cargo test -p lance --lib conflict_matrix::footprint::discover_footprints -- --ignored --nocapture
/// ```
#[tokio::test]
#[ignore = "discovery tool, not an assertion; regenerates the FOOTPRINT_MATRIX grid"]
// Stdout is the output format here, as in `cases::discover`.
#[allow(clippy::print_stdout)]
async fn discover_footprints() {
    let scoped = scoped();
    let width = scoped
        .iter()
        .map(|s| s.name().len())
        .max()
        .expect("there is at least one fragment-scoped scenario")
        + 1;
    // One symbol per footprint plus the separators between them, and a gutter.
    let column = Footprint::ALL.len() * 2 + 2;

    let mut out = format!("{:width$}| ", "");
    for theirs in &scoped {
        out.push_str(&format!("{:<column$}", theirs.code()));
    }
    for ours in &scoped {
        out = out.trim_end().to_string();
        out.push('\n');
        out.push_str(&format!("{:width$}| ", ours.name()));
        for theirs in &scoped {
            let bug = known_bug(*ours, *theirs).is_some();
            let mut entry = String::new();
            for footprint in Footprint::ALL {
                if !entry.is_empty() {
                    entry.push('/');
                }
                let cell = match predetermined(*theirs, footprint, bug) {
                    Some(cell) => cell,
                    None => Cell::Expected(run(*ours, *theirs, footprint).await.0),
                };
                entry.push(cell.symbol());
            }
            out.push_str(&format!("{entry:<column$}"));
        }
    }
    println!("{}", out.trim_end());
}
