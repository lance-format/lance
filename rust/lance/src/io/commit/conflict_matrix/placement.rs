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
//! reasons about what each side touches, because the matrix never asks a
//! question whose answer depends on it.
//!
//! This module asks it, at both granularities two operations can be disjoint at:
//!
//! - [`Placement::OtherFragment`] — `theirs` works on fragment 1, which `ours`
//!   never touches. Visible in the fragment ids a transaction carries.
//! - [`Placement::OtherField`] — `theirs` works on the *same* fragment, but on
//!   column `d` while `ours` works on `c`. Visible only to something that
//!   reasons about fields, so a resolver can distinguish the fragment case and
//!   still miss this one.
//!
//! A cell where the answers differ is a pair legacy resolves by what it touches. A
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
use super::scenarios::{Placement, Scenario};
use super::{Isolation, Outcome};
use super::{invariants, oracle};

/// What one ordered pair does at one placement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Cell {
    /// The placement does not apply to this `theirs`, so there is nothing to
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

/// What each fragment-scoped ordered pair does at each placement, stated under
/// [`Isolation::Legacy`].
///
/// Rows are `ours`, columns are `theirs`, keyed by [`Scenario::code`]. Each cell
/// is `same/field/fragment`: what happens when `theirs` touches exactly what
/// `ours` does, when it touches a different column of the same fragment, and
/// when it touches a different fragment. `L` lands, `R` retryable, `X`
/// incompatible, `!` excluded as a known bug, `-` not applicable.
///
/// A cell reading `R/L/L` is fully placement-sensitive. `R/R/L` distinguishes
/// fragments but not fields. `R/R/R` rejects on opcode alone.
///
/// Regenerate with the `discover_placements` test below.
const PLACEMENT_MATRIX: &str = "\
                       | dl      df      ur      uc      ov      dr      ci      rw
delete                 | R/-/L   R/-/L   L/-/L   R/R/L   L/L/L   R/R/L   L/-/L   R/-/L
delete_whole_fragment  | R/-/L   R/-/L   R/-/L   R/R/L   L/L/L   R/R/L   L/-/L   R/-/L
update_rewrite_rows    | L/-/L   R/-/L   R/-/L   R/R/L   R/R/L   R/R/L   L/-/L   R/-/L
update_rewrite_columns | R/-/L   R/-/L   R/-/L   R/R/L   R/L/L   R/R/L   L/-/L   R/-/L
data_overlay           | L/-/L   R/-/L   R/-/L   L/L/L   L/L/L   L/L/L   L/-/L   R/-/L
data_replacement       | L/-/L   X/-/L   R/-/L   R/L/L   L/L/L   R/L/L   L/-/L   R/-/L
create_index           | L/-/L   L/-/L   L/-/L   L/L/L   L/L/L   L/L/L   R/-/R   R/-/L
rewrite                | R/-/L   R/-/L   R/-/L   R/R/L   R/R/L   R/R/L   R/-/L   R/-/L
";

/// The level [`PLACEMENT_MATRIX`] was observed under.
const PLACEMENT_ISOLATION: Isolation = Isolation::Legacy;

/// Parsed form of [`PLACEMENT_MATRIX`], keyed by pair then placement.
static PLACEMENT_CELLS: LazyLock<HashMap<(Scenario, Scenario, Placement), Cell>> =
    LazyLock::new(parse);

/// The operations this module covers, in matrix order.
fn scoped() -> Vec<Scenario> {
    Scenario::ALL
        .into_iter()
        .filter(|scenario| scenario.is_fragment_scoped())
        .collect()
}

/// What a cell must be without running it, if anything. `None` means run it.
fn predetermined(theirs: Scenario, placement: Placement, bug: bool) -> Option<Cell> {
    if placement == Placement::OtherField && !theirs.is_field_scoped() {
        Some(Cell::NotApplicable)
    } else if bug {
        Some(Cell::Excluded)
    } else {
        None
    }
}

fn parse() -> HashMap<(Scenario, Scenario, Placement), Cell> {
    let scenario_by = |field: fn(Scenario) -> &'static str, value: &str| {
        Scenario::ALL
            .into_iter()
            .find(|s| field(*s) == value)
            .unwrap_or_else(|| panic!("the placement matrix names an unknown scenario {value:?}"))
    };

    let mut lines = PLACEMENT_MATRIX
        .lines()
        .filter(|line| !line.trim().is_empty());
    let header = lines.next().expect("the placement matrix has a header row");
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
        "the placement matrix columns must be the fragment-scoped scenarios, in order",
    );

    let mut cells = HashMap::new();
    for line in lines {
        let (label, rest) = line
            .split_once('|')
            .expect("a placement matrix row is delimited by `|`");
        let ours = scenario_by(Scenario::name, label.trim());
        let entries: Vec<&str> = rest.split_whitespace().collect();
        assert_eq!(
            entries.len(),
            columns.len(),
            "placement matrix row {} has {} cells but there are {} columns",
            ours.name(),
            entries.len(),
            columns.len(),
        );
        for (theirs, entry) in columns.iter().zip(entries) {
            let symbols: Vec<&str> = entry.split('/').collect();
            assert_eq!(
                symbols.len(),
                Placement::ALL.len(),
                "cell {entry:?} does not name all {} placements",
                Placement::ALL.len(),
            );
            for (placement, symbol) in Placement::ALL.into_iter().zip(symbols) {
                assert!(
                    cells
                        .insert((ours, *theirs, placement), Cell::from_symbol(symbol))
                        .is_none(),
                    "the placement matrix has two rows for {}",
                    ours.name(),
                );
            }
        }
    }
    cells
}

/// One row of the placement matrix: `ours` against every fragment-scoped
/// `theirs`, at every placement.
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
async fn placement_row(#[case] ours: Scenario) {
    for theirs in scoped() {
        for placement in Placement::ALL {
            let cell = *PLACEMENT_CELLS
                .get(&(ours, theirs, placement))
                .unwrap_or_else(|| {
                    panic!(
                        "({}, {}) at {} is missing from the placement matrix; re-run \
                         `discover_placements`",
                        ours.name(),
                        theirs.name(),
                        placement.name(),
                    )
                });
            let Cell::Expected(expected) = cell else {
                continue;
            };
            let context = format!(
                "({} over {} touching {})",
                ours.name(),
                theirs.name(),
                placement.name(),
            );
            let (outcome, landed) = run(ours, theirs, placement).await;
            assert_eq!(
                outcome, expected,
                "{context}: expected {expected:?} but observed {outcome:?}",
            );
            if let Some((before, after)) = landed {
                invariants::check_all(&before, &after, ours, &context).await;
                // Boxed for the same reason as in `cases`: a second commit
                // sequence makes the future large enough to trip clippy.
                Box::pin(oracle::check(
                    PLACEMENT_ISOLATION,
                    ours,
                    theirs,
                    placement,
                    &after,
                    &context,
                ))
                .await;
            }
        }
    }
}

/// Every pair must be covered at every placement, and the cells that are not
/// asserted must be ones the harness can justify not asserting.
///
/// `-` has to line up with [`Scenario::is_field_scoped`] and `!` with
/// `KNOWN_BUGS`, or the grid is quietly skipping cells. `KNOWN_BUGS` is keyed on
/// the pair rather than the placement, so a bug that only reproduces when the
/// placements overlap exempts the disjoint cells too; requiring all three to be
/// excluded together keeps that visible rather than letting it pass unnoticed.
#[test]
fn placement_matrix_is_total() {
    for ours in scoped() {
        for theirs in scoped() {
            let bug = known_bug(ours, theirs).is_some();
            for placement in Placement::ALL {
                let cell = *PLACEMENT_CELLS
                    .get(&(ours, theirs, placement))
                    .unwrap_or_else(|| {
                        panic!(
                            "({}, {}) at {} is missing from the placement matrix",
                            ours.name(),
                            theirs.name(),
                            placement.name(),
                        )
                    });
                match (cell, predetermined(theirs, placement, bug)) {
                    (Cell::Expected(_), None) => {}
                    (actual, Some(wanted)) if actual == wanted => {}
                    (actual, _) => panic!(
                        "({}, {}) at {} is {actual:?}, which does not match what the harness \
                         says about it: field-scoped {}, known bug {bug}",
                        ours.name(),
                        theirs.name(),
                        placement.name(),
                        theirs.is_field_scoped(),
                    ),
                }
            }
        }
    }
}

/// Print the observed outcome for every fragment-scoped pair at every placement,
/// as a grid.
///
/// Not an assertion — this regenerates [`PLACEMENT_MATRIX`]. Run it with:
///
/// ```text
/// cargo test -p lance --lib conflict_matrix::placement::discover_placements -- --ignored --nocapture
/// ```
#[tokio::test]
#[ignore = "discovery tool, not an assertion; regenerates the PLACEMENT_MATRIX grid"]
// Stdout is the output format here, as in `cases::discover`.
#[allow(clippy::print_stdout)]
async fn discover_placements() {
    let scoped = scoped();
    let width = scoped
        .iter()
        .map(|s| s.name().len())
        .max()
        .expect("there is at least one fragment-scoped scenario")
        + 1;
    // One symbol per placement plus the separators between them, and a gutter.
    let column = Placement::ALL.len() * 2 + 2;

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
            for placement in Placement::ALL {
                if !entry.is_empty() {
                    entry.push('/');
                }
                let cell = match predetermined(*theirs, placement, bug) {
                    Some(cell) => cell,
                    None => Cell::Expected(run(*ours, *theirs, placement).await.0),
                };
                entry.push(cell.symbol());
            }
            out.push_str(&format!("{entry:<column$}"));
        }
    }
    println!("{}", out.trim_end());
}
