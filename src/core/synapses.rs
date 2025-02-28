//! A `Synapse` models a single connection between a `Column` and an input bit.
//! Each synapse links exactly one input index to one column.
//! 
//! If the permanence is above a certain threshold, the synapse is considered "connected".
//! During learning, permanence is increases or decreased depening on whether the corresponding
//! input bit was active. A connected synapse counts toward the column's overlap score.
//! 
//! The centralized `Synapses` struct is a pool that stores all synapses for all columns 
//! in a single contiguous vec (array). Each column's synapses occupy a contiguous subrange 
//! within this array. The column keeps track of how many synapses it owns and where they lie.
//! 
//! This design allows efficient iteration over synapses (e.g. for updating),
//! quick global operatins (e.g. sorting connected vs. disconnected synapses), 
//! compact memory usage, and simple indexing.
//! 
//! The pool exposes methods to initialize a column's synapses with randomized permanence values,
//! sort the synapses of a column so connected synapses are at the front of its subrange,
//! and update or trim synapse permanence values.

use rand::Rng;
use std::ops::Range;

/// A synapse connecting an input index with an associated permanence value.
#[derive(Debug, Default, Clone, Copy)]
pub struct Synapse {
    /// Points to which input bit this synapse connects to.
    pub index: isize,

    /// Represents the strength of the connection between the synapse and the input bit.
    pub permanence: f32,
}

/// Options governing how synapse permanence is adjusted.
#[derive(Debug)]
pub struct SynapsePermenenceOptions {
    pub inactive_decrement: f32,
    pub active_increment: f32,
    pub connected: f32,
    pub below_stimulus_increment: f32,
    pub min: f32,
    pub max: f32,
    pub trim_threshold: f32,
}

/// A flat pool of potential synapses for all columns.
/// Each column is allotted a contiguous region in the internal synapses vector.
#[derive(Debug)]
pub struct Synapses {
    /// All potential synapses for each column.
    synapses: Vec<Synapse>,

    /// The number of synapses stored for each column.
    synapse_count_per_column: Vec<usize>,

    /// The number of connected synapses for each column after pivot sorting.
    connected_synapse_count_per_column: Vec<usize>,

    /// The maximum number of synapses allowed per column.
    max_synapses_per_column: usize,
}

impl Synapses {
    /// Creates a new synapse pool for `num_columns` columns with capacity `max_synapses` per column.
    pub fn new(num_columns: usize, max_potential: usize) -> Self {
        Self {
            synapses: vec![Synapse::default(); num_columns * max_potential],
            synapse_count_per_column: vec![0; num_columns],
            connected_synapse_count_per_column: vec![0; num_columns],
            max_synapses_per_column: max_potential,
        }
    }

    /// Inserts a new synapse into the specified column.
    pub fn insert(&mut self, column: usize, synapse: Synapse) -> usize {
        let count = self.synapse_count_per_column[column];
        assert!(
            count < self.max_synapses_per_column,
            "Attempting to insert more synapses than allowed for column {}",
            column
        );
        let index = column * self.max_synapses_per_column + count;
        self.synapses[index] = synapse;
        self.synapse_count_per_column[column] += 1;
        self.synapse_count_per_column[column]
    }

    /// Initializes the synapse pool for a column from the given candidate input indices.
    /// The permanence of each synapse is randomly initialized based on the `init_connected_pct` parameter.
    pub fn init_column<R: Rng>(
        &mut self,
        column: usize,
        potential: &[usize],
        init_connected_percentage: f32,
        options: &SynapsePermenenceOptions,
        rng: &mut R,
    ) {
        let synapses: Vec<Synapse> = potential
            .iter()
            .map(|&input_index| {
                let random = if rng.random::<f32>() <= init_connected_percentage {
                    options.connected + (options.max - options.connected) * rng.random::<f32>()
                } else {
                    options.connected * rng.random::<f32>()
                };

                let permanence = if random > options.trim_threshold {
                    (random * 100_000.0).round() / 100_000.0
                } else {
                    0.0
                };

                Synapse {
                    index: input_index as isize,
                    permanence,
                }
            })
            .collect();
        let column_start = column * self.max_synapses_per_column;
        self.synapses[column_start..column_start + synapses.len()].copy_from_slice(&synapses);
        self.synapse_count_per_column[column] = synapses.len();
        self.sort_column(column, options.connected);
    }

    /// Reorders the synapses in a column so that those with permanence ≥ `connected_threshold` come first.
    pub fn sort_column(&mut self, column: usize, connected_threshold: f32) {
        let range = self.col_range(column);
        let slice = &mut self.synapses[range];

        let mut pivot = 0;

        for i in 0..slice.len() {
            if slice[i].permanence >= connected_threshold {
                slice.swap(i, pivot);
                pivot += 1;
            }
        }

        self.connected_synapse_count_per_column[column] = pivot;
    }

    /// Updates permanence values in a column:
    /// - if `raise` is true, first raise values until `stimulus_threshold` is met,
    /// - then clamp values to [opts.min, opts.max] (and trim low values).
    /// - finally, sort the column synapses by permanence, so that connected synapses come first.
    pub fn update_column_permanences(
        &mut self,
        column: usize,
        raise_permanences: bool,
        stimulus_threshold: i32,
        options: &SynapsePermenenceOptions,
    ) {
        if raise_permanences {
            self.raise_column_permanances(column, stimulus_threshold, options);
        }

        for syn in self.column_mut(column) {
            if syn.index >= 0 {
                if syn.permanence <= options.trim_threshold {
                    syn.permanence = 0.0;
                } else {
                    syn.permanence = syn.permanence.clamp(options.min, options.max);
                }
            }
        }

        self.sort_column(column, options.connected);
    }

    /// Raises synapse permanence in a column until at least `stimulus_threshold` ≥ `options.connected`.
    pub fn raise_column_permanances(
        &mut self,
        column: usize,
        stimulus_threshold: i32,
        options: &SynapsePermenenceOptions,
    ) {
        let slice = self.column_mut(column);

        while slice
            .iter()
            .filter(|syn| syn.permanence >= options.connected)
            .count()
            < stimulus_threshold as usize
        {
            for syn in slice.iter_mut() {
                if syn.index >= 0 {
                    syn.permanence += options.below_stimulus_increment;
                }
            }
        }
    }

    /// Returns the index range corresponding to the synapses stored for the given column.
    fn col_range(&self, column: usize) -> Range<usize> {
        let start = column * self.max_synapses_per_column;
        let end = start + self.synapse_count_per_column[column];
        start..end
    }

    /// Returns the index range corresponding to the first `size` synapses stored for the given column.
    fn col_range_sized(&self, column: usize, size: usize) -> Range<usize> {
        let start = column * self.max_synapses_per_column;
        let end = start + size;
        start..end
    }

    /// Returns an immutable slice for all synapses in the given column.
    pub fn column(&self, col: usize) -> &[Synapse] {
        &self.synapses[self.col_range(col)]
    }

    /// Returns a mutable slice for all synapses in the given column.
    pub fn column_mut(&mut self, column: usize) -> &mut [Synapse] {
        let r = self.col_range(column);
        &mut self.synapses[r]
    }

    /// Returns an immutable slice for the connected synapses in the given column.
    pub fn colum_connected(&self, column: usize) -> &[Synapse] {
        &self.synapses
            [self.col_range_sized(column, self.connected_synapse_count_per_column[column])]
    }
}
