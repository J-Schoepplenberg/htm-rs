//! The `TemporalMemory` module implements a core component of HTM that learns and predicts temporal sequences.
//!
//! At a high level, it models a set of columns, where each column contains multiple cells. 
//! Each cell can form multiple dendritic segments, which in turn consist of synapses.
//! Unfortunately, it is much slower in practice than spatial pooling.
//! 
//! Column:
//! - A group of cells that share common input. 
//! - Each column processes feed-forward signals and contributes to forming sparse distributed representations.
//! 
//! Cell: 
//! - An individual processing unit within a column. 
//! - Cells are responsible for representing different contexts of the same input.
//! 
//! Dendritic Segment (Segment):
//! - A cluster of synapses on a cell that detects patterns of activity from other cells. 
//! - Each segment learns to recognize sequences by forming connections to presynaptic cells.
//! 
//! Synapse: 
//! - A connection from a presynaptic cell (i.e., a cell that provided input) to a dendritic segment. 
//! - Each synapse has a permanence value that indicates the strength of the connection and determines if the connection is active.
//! 
//! Presynaptic Cell: 
//! - A cell that provides input to another cell via a synapse. 
//! - In the context of temporal memory, presynaptic activity is used to predict future cell activations.
//! 
//! Bursting:
//! - When a column becomes active due to feed-forward input but no cell was correctly predicted, all cells in the column are activated. 
//! - This process allows the system to learn new sequences and is called bursting.
//! 
//! Winner Cells/Columns: 
//! - Cells/columns that have been selected based on their predictive state or through bursting, which then guide the learning process.
//!
//! How It Works:
//! - The Temporal Memory processes input in discrete time steps. 
//! - For each time step, it receives a set of active (feed-forward) columns.
//! - In each active column, it checks if any cell was correctly predicted by an active dendritic segment. 
//! - If so, those cells are activated; otherwise, the column bursts.
//! - The algorithm then updates dendritic segments by reinforcing synapses that correctly predicted activity.
//! - It also punishes those that did not, following Hebbian-like learning rules.
//! - Additionally, new synapses may be grown on segments to incorporate previously winning cells, 
//! - Thus, it is adapting the network to better capture temporal patterns over time.

use fxhash::{FxHashMap, FxHashSet};
use rand::prelude::*;
use std::mem;

/// Represents an address that uniquely identifies a cell by its column and cell indices.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CellAddress {
    col: usize,
    cell: usize,
}

/// Represents a synapse that connects a segment to a presynaptic cell, holding a permanence value.
#[derive(Clone, Debug)]
pub struct Synapse {
    presynaptic_cell: CellAddress,
    permanence: f64,
}

/// Represents a dendritic segment (or distal segment) which contains a list of synapses and belongs to a cell.
#[derive(Clone, Debug)]
pub struct Segment {
    synapses: Vec<Synapse>,
}

/// Represents a cell that contains one or more dendritic segments.
#[derive(Clone, Debug)]
pub struct Cell {
    segments: Vec<Segment>,
}

/// Represents a column which is a group of cells.
#[derive(Clone, Debug)]
pub struct Column {
    cells: Vec<Cell>,
}

/// Enumerates the actions that can be taken on a column during temporal memory processing.
pub enum Action {
    Activate,
    Burst,
    Punish,
}

/// Holds the parameters required for the Temporal Memory algorithm's learning and activation.
#[derive(Clone, Debug)]
pub struct TemporalMemoryParams {
    pub activation_threshold: usize,
    pub connected_permanence: f64,
    pub learning_threshold: usize,
    pub initial_permanence: f64,
    pub permanence_increment: f64,
    pub permanence_decrement: f64,
    pub predicted_decrement: f64,
    pub synapse_sample_size: usize,
    pub learning_enabled: bool,
}

/// Implements the Temporal Memory algorithm which models the activation and learning of temporal sequences.
///
/// The Temporal Memory processes feed-forward input by activating columns and cells, predicting future activity
/// based on past patterns, and adapting synapse permanences through Hebbian-like learning rules. It operates in
/// discrete time steps and incorporates phases such as activating predicted cells, bursting columns without predictions,
/// and punishing erroneous predictions to gradually learn the temporal structure of the input data.
pub struct TemporalMemory {
    columns: Vec<Column>,
    // Learning and activation parameters.
    activation_threshold: usize,
    connected_permanence: f64,
    learning_threshold: usize,
    initial_permanence: f64,
    permanence_increment: f64,
    permanence_decrement: f64,
    predicted_decrement: f64,
    synapse_sample_size: usize,
    learning_enabled: bool,

    // State from previous time step (t-1)
    prev_active_cells: FxHashSet<CellAddress>,
    prev_winner_cells: FxHashSet<CellAddress>,
    prev_active_segments: FxHashSet<(CellAddress, usize)>, // (cell address, segment index)
    prev_matching_segments: FxHashSet<(CellAddress, usize)>,

    // Current state (t)
    active_cells: FxHashSet<CellAddress>,
    winner_cells: FxHashSet<CellAddress>,
    active_segments: FxHashSet<(CellAddress, usize)>,
    matching_segments: FxHashSet<(CellAddress, usize)>,
    // For each segment, record the number of active potential synapses.
    num_active_potential_synapses: FxHashMap<(CellAddress, usize), usize>,

    // Current time step.
    t: u64,

    rand: StdRng,
}

impl TemporalMemory {
    /// Constructs a new Temporal Memory instance:
    /// - Initializes a specified number of columns, each containing a fixed number of cells.
    /// - Pre-allocates hash sets and maps for tracking cell and segment states to avoid re-hashing.
    #[inline]
    pub fn new(num_columns: usize, cells_per_column: usize, params: TemporalMemoryParams) -> Self {
        let mut columns = Vec::with_capacity(num_columns);
        for _ in 0..num_columns {
            let mut cells = Vec::with_capacity(cells_per_column);
            for _ in 0..cells_per_column {
                cells.push(Cell {
                    segments: Vec::new(),
                });
            }
            columns.push(Column { cells });
        }

        let capacity = num_columns * cells_per_column / 4;

        TemporalMemory {
            columns,
            activation_threshold: params.activation_threshold,
            connected_permanence: params.connected_permanence,
            learning_threshold: params.learning_threshold,
            initial_permanence: params.initial_permanence,
            permanence_increment: params.permanence_increment,
            permanence_decrement: params.permanence_decrement,
            predicted_decrement: params.predicted_decrement,
            synapse_sample_size: params.synapse_sample_size,
            learning_enabled: params.learning_enabled,
            prev_active_cells: FxHashSet::with_capacity_and_hasher(capacity, Default::default()),
            prev_winner_cells: FxHashSet::with_capacity_and_hasher(capacity, Default::default()),
            prev_active_segments: FxHashSet::with_capacity_and_hasher(capacity, Default::default()),
            prev_matching_segments: FxHashSet::with_capacity_and_hasher(
                capacity,
                Default::default(),
            ),
            active_cells: FxHashSet::with_capacity_and_hasher(capacity, Default::default()),
            winner_cells: FxHashSet::with_capacity_and_hasher(capacity, Default::default()),
            active_segments: FxHashSet::with_capacity_and_hasher(capacity, Default::default()),
            matching_segments: FxHashSet::with_capacity_and_hasher(capacity, Default::default()),
            num_active_potential_synapses: FxHashMap::with_capacity_and_hasher(
                capacity,
                Default::default(),
            ),
            t: 0,
            rand: StdRng::from_seed([42u8; 32]),
        }
    }

    /// Executes one time step of the Temporal Memory algorithm.
    ///
    /// - Converts the list of active (feed-forward) columns into a set for rapid lookup.
    /// - For each column, determines the appropriate action:
    ///   - **Activate:** if the column contains a cell that was correctly predicted.
    ///   - **Burst:** if no cell in the column was predicted.
    ///   - **Punish:** if a cell was predicted but did not activate.
    /// - Processes the determined actions to update cell states and adjust synapse permanences.
    /// - Evaluates segments across all cells to update active and matching segment sets based on active synapse counts.
    /// - Prepares the state for the next time step by swapping current and previous state variables.
    #[inline]
    pub fn step(&mut self, active_columns: &[usize]) {
        let active_column_set: FxHashSet<usize> = active_columns.iter().copied().collect();
        let mut actions = Vec::with_capacity(active_columns.len() * 2);

        for col_index in 0..self.columns.len() {
            let is_active = active_column_set.contains(&col_index);

            if is_active {
                let mut predicted = false;
                'outer: for cell_idx in 0..self.columns[col_index].cells.len() {
                    let addr = CellAddress {
                        col: col_index,
                        cell: cell_idx,
                    };

                    for seg_idx in 0..self.columns[col_index].cells[cell_idx].segments.len() {
                        if self.prev_active_segments.contains(&(addr, seg_idx)) {
                            predicted = true;
                            break 'outer;
                        }
                    }
                }

                actions.push((
                    col_index,
                    if predicted {
                        Action::Activate
                    } else {
                        Action::Burst
                    },
                ));
            } else {
                let mut matching = false;
                'outer: for cell_idx in 0..self.columns[col_index].cells.len() {
                    let addr = CellAddress {
                        col: col_index,
                        cell: cell_idx,
                    };

                    for seg_idx in 0..self.columns[col_index].cells[cell_idx].segments.len() {
                        if self.prev_matching_segments.contains(&(addr, seg_idx)) {
                            matching = true;
                            break 'outer;
                        }
                    }
                }

                if matching {
                    actions.push((col_index, Action::Punish));
                }
            }
        }

        for (col_index, action) in actions {
            match action {
                Action::Activate => self.activate_predicted_column(col_index),
                Action::Burst => self.burst_column(col_index),
                Action::Punish => self.punish_predicted_column(col_index),
            }
        }

        self.num_active_potential_synapses.clear();

        let active_cells_index: FxHashSet<_> = self.active_cells.iter().copied().collect();

        for (col_idx, column) in self.columns.iter().enumerate() {
            for (cell_idx, cell) in column.cells.iter().enumerate() {
                let addr = CellAddress {
                    col: col_idx,
                    cell: cell_idx,
                };

                for (seg_idx, segment) in cell.segments.iter().enumerate() {
                    let mut active_conn = 0;
                    let mut active_pot = 0;

                    for syn in &segment.synapses {
                        if active_cells_index.contains(&syn.presynaptic_cell) {
                            active_pot += 1;
                            if syn.permanence >= self.connected_permanence {
                                active_conn += 1;
                            }
                        }
                    }

                    if active_conn >= self.activation_threshold {
                        self.active_segments.insert((addr, seg_idx));
                    }

                    if active_pot >= self.learning_threshold {
                        self.matching_segments.insert((addr, seg_idx));
                    }

                    self.num_active_potential_synapses
                        .insert((addr, seg_idx), active_pot);
                }
            }
        }

        self.t += 1;
        mem::swap(&mut self.prev_active_cells, &mut self.active_cells);
        self.active_cells.clear();
        mem::swap(&mut self.prev_winner_cells, &mut self.winner_cells);
        self.winner_cells.clear();
        mem::swap(&mut self.prev_active_segments, &mut self.active_segments);
        self.active_segments.clear();
        mem::swap(
            &mut self.prev_matching_segments,
            &mut self.matching_segments,
        );
        self.matching_segments.clear();
    }

    /// Activates cells in a column that were predicted in the previous time step:
    /// - Iterates over each cell and its segments in the specified column.
    /// - If a segment was active in the previous time step (i.e., predicted), the cell is marked as active/winner.
    /// - Schedules new synapse growth if the segment has fewer synapses than the desired sample size.
    /// 
    /// If learning is enabled:
    /// - Adjusts the permanence of each synapse in the segment.
    /// - This is based on whether the corresponding presynaptic cell was active previously.
    #[inline]
    pub fn activate_predicted_column(&mut self, col_index: usize) {
        let mut grow_requests = Vec::new();
        let column = &mut self.columns[col_index];

        for (cell_idx, cell) in column.cells.iter_mut().enumerate() {
            let addr = CellAddress {
                col: col_index,
                cell: cell_idx,
            };

            for (seg_idx, segment) in cell.segments.iter_mut().enumerate() {
                if self.prev_active_segments.contains(&(addr, seg_idx)) {
                    self.active_cells.insert(addr);
                    self.winner_cells.insert(addr);

                    if self.learning_enabled {
                        let prev_active_lookup = &self.prev_active_cells;

                        for syn in segment.synapses.iter_mut() {
                            if prev_active_lookup.contains(&syn.presynaptic_cell) {
                                syn.permanence += self.permanence_increment;
                            } else {
                                syn.permanence -= self.permanence_decrement;
                            }
                        }

                        let current_active = *self
                            .num_active_potential_synapses
                            .get(&(addr, seg_idx))
                            .unwrap_or(&0);

                        let new_synapse_count =
                            self.synapse_sample_size.saturating_sub(current_active);

                        if new_synapse_count > 0 {
                            grow_requests.push((addr, seg_idx, new_synapse_count));
                        }
                    }
                }
            }
        }

        for (addr, seg_idx, count) in grow_requests {
            self.grow_synapses(addr, seg_idx, count);
        }
    }

    /// Bursts a column when no cell in the column was predicted to become active:
    /// - Marks all cells in the column as active.
    /// - Searches for a matching segment in the column to select a winning cell.
    /// - If no matching segment exists, selects the least used cell and (if learning is enabled) grows a new segment.
    /// - Adjusts synapse permanence for the chosen segment and grows additional synapses if needed.
    #[inline]
    pub fn burst_column(&mut self, col_index: usize) {
        let column = &mut self.columns[col_index];
        let cells_len = column.cells.len();

        for cell_idx in 0..cells_len {
            self.active_cells.insert(CellAddress {
                col: col_index,
                cell: cell_idx,
            });
        }

        let mut winner: Option<CellAddress> = None;
        let mut best_score: isize = -1;
        let mut chosen_seg: Option<usize> = None;

        for (cell_idx, cell) in column.cells.iter().enumerate() {
            let addr = CellAddress {
                col: col_index,
                cell: cell_idx,
            };

            for (seg_idx, _) in cell.segments.iter().enumerate() {
                if self.prev_matching_segments.contains(&(addr, seg_idx)) {
                    let score = *self
                        .num_active_potential_synapses
                        .get(&(addr, seg_idx))
                        .unwrap_or(&0) as isize;

                    if score > best_score {
                        best_score = score;
                        winner = Some(addr);
                        chosen_seg = Some(seg_idx);
                    }
                }
            }
        }

        if winner.is_none() {
            let cell_addr = self.least_used_cell(col_index);
            winner = Some(cell_addr);

            if self.learning_enabled {
                chosen_seg = Some(self.grow_new_segment(cell_addr));
            }
        }

        let winner_addr = winner.unwrap();
        self.winner_cells.insert(winner_addr);

        if let Some(seg_idx) = chosen_seg {
            let cell = &mut self.columns[winner_addr.col].cells[winner_addr.cell];

            if let Some(segment) = cell.segments.get_mut(seg_idx) {
                let prev_active_lookup = &self.prev_active_cells;

                for syn in segment.synapses.iter_mut() {
                    if prev_active_lookup.contains(&syn.presynaptic_cell) {
                        syn.permanence += self.permanence_increment;
                    } else {
                        syn.permanence -= self.permanence_decrement;
                    }
                }
            }

            let current_active = *self
                .num_active_potential_synapses
                .get(&(winner_addr, seg_idx))
                .unwrap_or(&0);

            let new_count = self.synapse_sample_size.saturating_sub(current_active);

            if new_count > 0 {
                self.grow_synapses(winner_addr, seg_idx, new_count);
            }
        }
    }

    /// Punishes segments in a column that were predicted (matching) but did not become active:
    /// - Iterates over each cell and its segments in the specified column.
    /// - For each segment that was previously matching, reduces the permanence of synapses whose presynaptic cell was active.
    /// - This punitive adjustment helps to weaken incorrect predictions during learning.
    #[inline]
    pub fn punish_predicted_column(&mut self, col_index: usize) {
        if !self.learning_enabled {
            return;
        }

        let column = &mut self.columns[col_index];
        let prev_active_lookup = &self.prev_active_cells;

        for (cell_idx, cell) in column.cells.iter_mut().enumerate() {
            let addr = CellAddress {
                col: col_index,
                cell: cell_idx,
            };

            for (seg_idx, segment) in cell.segments.iter_mut().enumerate() {
                if self.prev_matching_segments.contains(&(addr, seg_idx)) {
                    for syn in segment.synapses.iter_mut() {
                        if prev_active_lookup.contains(&syn.presynaptic_cell) {
                            syn.permanence -= self.predicted_decrement;
                        }
                    }
                }
            }
        }
    }

    /// Identifies and returns the cell with the fewest segments within a column:
    /// - Iterates through all cells in the column to find the minimum number of segments.
    /// - If multiple cells have the same minimum count, one is chosen at random.
    /// - Returns the address of the selected cell.
    #[inline]
    pub fn least_used_cell(&mut self, col_index: usize) -> CellAddress {
        let column = &self.columns[col_index];
        let mut min_segments = usize::MAX;
        let mut min_cells = Vec::new();

        for (idx, cell) in column.cells.iter().enumerate() {
            let seg_count = cell.segments.len();
            if seg_count < min_segments {
                min_segments = seg_count;
                min_cells.clear();
                min_cells.push(idx);
            } else if seg_count == min_segments {
                min_cells.push(idx);
            }
        }

        let cell_idx = if min_cells.len() > 1 {
            min_cells[self.rand.random_range(0..min_cells.len())]
        } else {
            min_cells[0]
        };

        CellAddress {
            col: col_index,
            cell: cell_idx,
        }
    }

    /// Grows a new dendritic segment on the specified cell and returns the new segment's index:
    /// - Creates a new segment with pre-allocated space for synapses based on the sample size.
    /// - Appends the new segment to the cell's segment list.
    /// - Returns the index of the newly added segment.
    #[inline]
    pub fn grow_new_segment(&mut self, cell_addr: CellAddress) -> usize {
        let segment = Segment {
            synapses: Vec::with_capacity(self.synapse_sample_size),
        };

        let cell = &mut self.columns[cell_addr.col].cells[cell_addr.cell];
        cell.segments.push(segment);
        cell.segments.len() - 1
    }

    /// Grows new synapses on a given segment to reach the desired synapse sample size:
    /// - Selects a random subset of previously winning cells as candidates for new synapses.
    /// - Filters out candidates that already have an existing synapse connection.
    /// - Adds new synapses with the initial permanence until the segment reaches the desired sample size.
    #[inline]
    pub fn grow_synapses(
        &mut self,
        cell_addr: CellAddress,
        seg_idx: usize,
        mut new_synapse_count: usize,
    ) {
        if new_synapse_count == 0 || self.prev_winner_cells.is_empty() {
            return;
        }

        let mut candidates: Vec<CellAddress> = self.prev_winner_cells.iter().copied().collect();
        candidates.shuffle(&mut self.rand);

        if let Some(segment) = self.columns[cell_addr.col].cells[cell_addr.cell]
            .segments
            .get_mut(seg_idx)
        {
            let existing_synapses: FxHashSet<CellAddress> = segment
                .synapses
                .iter()
                .map(|syn| syn.presynaptic_cell)
                .collect();

            segment.synapses.reserve(new_synapse_count);

            for presynaptic in candidates {
                if new_synapse_count == 0 {
                    break;
                }

                if !existing_synapses.contains(&presynaptic) {
                    segment.synapses.push(Synapse {
                        presynaptic_cell: presynaptic,
                        permanence: self.initial_permanence,
                    });
                    new_synapse_count -= 1;
                }
            }
        }
    }

    /// Returns a vector of cell indices for winner cells from the previous time step.
    pub fn winner_cells(&self) -> Vec<usize> {
        self.prev_winner_cells
            .iter()
            .map(|addr| addr.cell)
            .collect()
    }

    /// Returns a vector of column indices for winner cells from the previous time step.
    pub fn winner_columns(&self) -> Vec<usize> {
        self.prev_winner_cells.iter().map(|addr| addr.col).collect()
    }
}