//! The `SpatialPooler` is a core component of HTM that:
//! - Initializes and maintains a set of columns, each with potential synapses into its own subset of the input space.
//! - Learns to increase/decrease synapse permanence (strength) values if the connected input bit was active/inactive.
//! - Computes an "overlap" score for each column based on how many connected synapses match the current input.
//! - Enforces sparse activity via inhibition, allowing only a subset of top columns to become "winner columns."
//!
//! Each column selectively "tunes" its connections to represent frequently encountered input patterns, leading to SDRs.
//!
//! What are duty cycles?
//! - They are rolling metrics that measure how often each column is meeting certain criteria over time.
//! - The SP tracks: overlap duty cycles (ODC) and active duty cycles (ADC).
//! - ODC tracks how frequently a column has a non-zero overlap score with the input.
//! - ADC tracks how frequently a column is chosen as a winner after inhibition.
//! - By comparing these metrics to thresholds, the SP can decide whether to boost columns.
//! - This prevents columns from becoming inactive or uncompetitive over time.

use super::{
    column::Column,
    synapses::{SynapsePermenenceOptions, Synapses},
    topology::Topology,
};
use rand::{
    rngs::StdRng,
    seq::{IteratorRandom, SliceRandom},
    SeedableRng,
};
use std::ops::Range;

/// The SpatialPooler manages a set of columns that compete to represent the input space.
/// It computes overlaps, applies inhibition, boosts weak columns, and adapts synapses during learning.
/// Synapse management is performed via the embedded `Synapses` pool.
pub struct SpatialPooler {
    /// A seeded pseudo-random number generator for reproducible randomness (e.g., synapse initialization).
    pub rand: StdRng,

    /// The total number of compute iterations performed so far (whether learning or not).
    pub iteration_num: u32,

    /// The number of compute iterations performed so far with learning enabled.
    pub iteration_learn_num: u32,

    /// Defines the radius (in input-space) around a column’s center from which potential synapses can be drawn. If -1, the entire input space is used.
    pub potential_radius: i32,

    /// Controls how many input bits within the `potential_radius` become potential synapses for each column.
    pub potential_percentage: f64,

    /// The minimum overlap a column must have to be considered for winning.
    pub stimulus_threshold: f32,

    /// Fraction of the maximum overlap duty cycle used as a threshold for deciding if a column’s overlap duty cycle is too low.
    pub min_percentage_overlap_duty_cycles: f32,

    /// Fraction of the maximum active duty cycle used as a threshold for deciding if a column’s active duty cycle is too low.
    pub min_percentage_active_duty_cycles: f32,

    /// The time window over which overlap and active duty cycles are updated. Smoothens out fluctuations.
    pub duty_cycle_period: u32,

    /// The maximum possible boost factor that can be applied to a column’s overlap if it is underactive.
    pub max_boost: f32,

    /// If true, neighborhoods "wrap around" the edges in topology calculations. The space behaves like a torus.
    pub wrap_around: bool,

    /// The total number of bits/inputs available.
    pub num_inputs: usize,

    /// The total number of columns in the Spatial Pooler.
    pub num_columns: usize,

    /// Settings for how synapse permanence is incremented/decremented and thresholds for trimming or connecting.
    pub synapse_permanence_options: SynapsePermenenceOptions,

    /// Fraction of each column’s synapses that initially start out above the "connected" threshold.
    pub init_connected_percentage: f32,

    pub density: f32,

    /// How often (in iterations) certain recalculations happen, like updating inhibition radius or min duty cycles.
    pub update_period: u32,

    /// The shape (dimensions) of the Spatial Pooler’s column grid.
    pub column_dimensions: Vec<usize>,

    /// The shape (dimensions) of the input space.
    pub input_dimensions: Vec<usize>,

    /// A helper structure that knows how to map 1D column indices to the nD column space.
    pub column_topology: Topology,

    /// A helper structure that knows how to map 1D input indices to the nD input space.
    pub input_topology: Topology,

    /// A vector of all column objects, each containing a unique index.
    pub columns: Vec<Column>,

    /// A pool managing all synapse data. Stores a contiguous block of synapses for every column in one big array.
    pub synapses: Synapses,

    /// Rolling average of how often each column has an overlap > 0 in the current time window.
    pub overlap_duty_cycles: Vec<f32>,

    /// Rolling average of how often each column is chosen as a winner. Used to identify columns that never get activated.
    pub active_duty_cycles: Vec<f32>,

    /// The threshold for each column’s overlap duty cycle, columns below this threshold may be permanence-boosted.
    pub min_overlap_duty_cycles: Vec<f32>,

    /// The threshold for each column’s active duty cycle, columns below this threshold may be overlap-boosted
    pub min_active_duty_cycles: Vec<f32>,

    /// A multiplier applied to a column’s overlap if it is underactive. Computed each iteration.
    pub boost_factors: Vec<f32>,

    // Represents each iteration how many connected synapses map to active input bits for each column.
    pub overlaps: Vec<f32>,

    /// The indices of columns that won the inhibition process this iteration (i.e., the active columns).
    pub winner_columns: Vec<usize>,

    /// Adjusted overlap array to break ties in local inhibition, so columns with the exact same overlap can be distinguished.
    pub tie_broken_overlaps: Vec<f32>,
}

impl SpatialPooler {
    /// Creates a new `SpatialPooler` with the given input and column dimensions.
    #[inline]
    pub fn new(input_dimensions: Vec<usize>, column_dimensions: Vec<usize>) -> Self {
        let num_columns = column_dimensions.iter().product();
        let num_inputs = input_dimensions.iter().product();
        let column_topology = Topology::new(&column_dimensions);
        let input_topology = Topology::new(&input_dimensions);

        Self {
            iteration_num: 0,
            iteration_learn_num: 0,
            potential_radius: 16,
            potential_percentage: 0.5,
            min_percentage_overlap_duty_cycles: 0.001,
            min_percentage_active_duty_cycles: 0.001,
            duty_cycle_period: 1000,
            max_boost: 10.0,
            wrap_around: true,
            num_inputs,
            num_columns,
            density: 0.2,
            update_period: 50,
            init_connected_percentage: 0.5,
            synapse_permanence_options: SynapsePermenenceOptions {
                inactive_decrement: 0.008,
                active_increment: 0.05,
                connected: 0.10,
                below_stimulus_increment: 0.10 / 10.0,
                min: 0.0,
                max: 1.0,
                trim_threshold: 0.05 / 2.0,
            },
            stimulus_threshold: 0.0,
            column_dimensions,
            input_dimensions,
            column_topology,
            input_topology,
            columns: Vec::with_capacity(num_columns),
            synapses: Synapses::new(num_columns, num_inputs),
            rand: StdRng::from_seed([42u8; 32]),
            overlap_duty_cycles: vec![0.0; num_columns],
            active_duty_cycles: vec![0.0; num_columns],
            min_overlap_duty_cycles: vec![0.0; num_columns],
            min_active_duty_cycles: vec![0.0; num_columns],
            boost_factors: vec![1.0; num_columns],
            overlaps: vec![0.0; num_columns],
            winner_columns: Vec::with_capacity(num_columns),
            tie_broken_overlaps: vec![0.0; num_columns],
        }
    }

    /// Performs any post-initialization adjustments:
    /// - Updates certain derived permanence parameters based on other fields
    /// - Ensures `potential_radius` is set to a valid default if unspecified.
    ///
    /// Keeps the SP’s internal parameters consistent after construction.
    #[inline]
    pub fn post_init(&mut self) {
        self.synapse_permanence_options.below_stimulus_increment =
            self.synapse_permanence_options.connected / 10.0;
        self.synapse_permanence_options.trim_threshold =
            self.synapse_permanence_options.active_increment / 2.0;
        if self.potential_radius == -1 {
            self.potential_radius = self.num_inputs as i32;
        }
    }

    /// Initializes the `SpatialPooler`:
    /// - Validates inhibition parameters.
    /// - Generates columns.
    /// - Builds and configures synapses for each column.
    #[inline]
    pub fn init(&mut self) {
        self.post_init();
        self.gen_columns();
        self.gen_column_potential_synapses();
        self.connect_and_configure_inputs();
    }

    /// Processes the current `input_vector`:
    /// - Updates iteration counters.
    /// - Calculates overlaps between columns and input subsets.
    /// - Applies boosting if learning is enabled.
    /// - Performs inhibition to pick winner columns.
    ///
    /// If learning is enabled:
    /// - Updates synapse permance values.
    /// - Updates duty cycles.
    /// - Resets thresholds and inhibition radii periodically.
    #[inline]
    pub fn compute(&mut self, input_pattern: &[bool], learn: bool) {
        self.update_iteration_number(learn);
        self.calculate_overlaps(input_pattern);
        self.boost(learn);
        self.inhibit_columns();

        if learn {
            self.adapt_synapses(input_pattern);
            self.update_duty_cycles();
            self.bump_up_weak_columns();
            self.update_boost_factors();
            if self.iteration_num % self.update_period == 0 {
                self.update_min_duty_cycles();
            }
        }
    }

    /// Adjusts synapses for each winner column after an input is processed:
    /// - Increments permanence of synapses whose input bit was active.
    /// - Decrements permanence of synapses whose input bit was inactive.
    /// - Ensures permanence values remain within valid bounds and sorts connected synapses.
    ///
    /// Implements Hebbian-like learning that shapes columns towards frequently active inputs.
    #[inline]
    pub fn adapt_synapses(&mut self, input_pattern: &[bool]) {
        for &col in &self.winner_columns {
            for syn in self.synapses.column_mut(col) {
                if input_pattern[syn.index as usize] {
                    syn.permanence += self.synapse_permanence_options.active_increment;
                } else {
                    syn.permanence -= self.synapse_permanence_options.inactive_decrement;
                }
            }
            self.synapses.update_column_permanences(
                col,
                true,
                (self.stimulus_threshold + 0.5) as i32,
                &self.synapse_permanence_options,
            );
        }
    }

    /// Updates the rolling duty cycles for overlap and active states:
    /// - Maintains an exponential moving average of how often each column overlaps or becomes active.
    /// - Increments the column’s active duty cycle if it’s a winner in the current iteration.
    ///
    /// Duty cycles measure how "healthy" or "engaged" each column is in representing inputs over time.
    #[inline]
    pub fn update_duty_cycles(&mut self) {
        let period = self.iteration_num.min(self.duty_cycle_period) as f32;
        let factor = (period - 1.0) / period;
        let boost = 1.0 / period;
        self.overlap_duty_cycles
            .iter_mut()
            .zip(&self.overlaps)
            .for_each(|(duty, &overlap)| {
                *duty = (*duty * (period - 1.0) + if overlap > 0.0 { 1.0 } else { 0.0 }) / period;
            });
        self.active_duty_cycles.iter_mut().for_each(|duty| {
            *duty *= factor;
        });
        self.winner_columns.iter().for_each(|&col| {
            self.active_duty_cycles[col] = self.active_duty_cycles[col].mul_add(factor, boost);
        });
    }

    /// Increases permanence on "weak" columns that have low overlap duty cycles:
    /// - For each column whose overlap duty cycle is below a threshold, bumps all its synapses’ permanence.
    /// - Then re-sorts that column’s synapse by permanence, so that connected synapses come first
    ///
    /// Prevents columns from perpetually remaining low-overlap, giving them a chance to learn and stay relevant.
    #[inline]
    pub fn bump_up_weak_columns(&mut self) {
        for (col, &overlap_dc) in self.overlap_duty_cycles.iter().enumerate() {
            if self.min_overlap_duty_cycles[col] > overlap_dc {
                for syn in self.synapses.column_mut(col) {
                    syn.permanence += self.synapse_permanence_options.below_stimulus_increment;
                }
                self.synapses.update_column_permanences(
                    col,
                    true,
                    (self.stimulus_threshold + 0.5) as i32,
                    &self.synapse_permanence_options,
                );
            }
        }
    }

    /// Recalculates each column’s boost factor based on its active duty cycle:
    /// - If a column’s activity is below the local min, its boost factor increases (up to a limit).
    /// - If its activity meets or exceeds the min, its boost factor is reset to 1.0.
    ///
    /// Boosting encourages underutilized columns to become active more easily.
    #[inline]
    pub fn update_boost_factors(&mut self) {
        if self.min_active_duty_cycles.iter().any(|&min| min > 0.0) {
            self.boost_factors
                .iter_mut()
                .zip(&self.min_active_duty_cycles)
                .zip(&self.active_duty_cycles)
                .for_each(|((boost, &min), &active)| {
                    *boost = if active > min {
                        1.0
                    } else {
                        let ma = min.max(f32::EPSILON);
                        ((1.0 - self.max_boost) / ma) * active + self.max_boost
                    };
                });
        }
    }

    /// Updates the minimum duty cycles for overlap and activity globally:
    /// - Calculates global maximum overlap and active duty cycles across all columns.
    /// - Sets each column’s minimum duty cycle to a fraction of these maximums.
    ///
    /// Ensures columns remain competitive globally, preventing "dead" columns.
    #[inline]
    pub fn update_min_duty_cycles(&mut self) {
        let m = self.min_percentage_overlap_duty_cycles
            * self
                .overlap_duty_cycles
                .iter()
                .fold(0.0, |acc: f32, &x| acc.max(x));
        let m2 = self.min_percentage_active_duty_cycles
            * self
                .active_duty_cycles
                .iter()
                .fold(0.0, |acc: f32, &x| acc.max(x));
        self.min_overlap_duty_cycles.fill(m);
        self.min_active_duty_cycles.fill(m2);
    }

    /// Multiplies each column’s overlap by its boost factor (if learning is on):
    /// - Scales overlap values before the inhibition step, so columns with higher boosts have a better chance to win.
    ///
    /// Part of the mechanism that ensures columns that have been "idle" get more opportunities to become winners.
    #[inline]
    pub fn boost(&mut self, learn: bool) {
        if learn {
            for (overlap, boost) in self.overlaps.iter_mut().zip(self.boost_factors.iter()) {
                *overlap *= *boost;
            }
        }
    }

    /// Calculates the raw overlap for each column with the current input:
    /// - Counts how many connected synapses map to an active input bit, storing that count in `self.overlaps[col]`.
    ///
    /// Overlap is the key metric determining a column’s initial activation level before inhibition.
    #[inline]
    pub fn calculate_overlaps(&mut self, input_pattern: &[bool]) {
        self.overlaps
            .iter_mut()
            .enumerate()
            .for_each(|(col, overlap)| {
                *overlap = self
                    .synapses
                    .colum_connected(col)
                    .iter()
                    .filter(|syn| input_pattern[syn.index as usize])
                    .count() as f32;
            });
    }

    /// Increments the global iteration counters, including a separate counter if `learn` is true.
    ///
    /// Tracks how many total iterations vs. learning iterations have occurred for reference and parameter updates.
    #[inline]
    pub fn update_iteration_number(&mut self, learn: bool) {
        self.iteration_num += 1;
        if learn {
            self.iteration_learn_num += 1;
        }
    }

    /// Implements global inhibition, columns are sorted by overlap, and the top fraction are selected:
    /// - Sorts every column by overlap descending.
    /// - Truncates the list when overlap < `stimulus_threshold` or once the fraction `density * num_columns` is reached.
    ///
    /// Ensures only a sparse subset of columns with the highest overlaps become active, ignoring local topology.
    #[inline]
    pub fn inhibit_columns(&mut self) {
        let mut candidates: Vec<_> = (0..self.num_columns).collect();
        candidates
            .sort_unstable_by(|&a, &b| self.overlaps[b].partial_cmp(&self.overlaps[a]).unwrap());
        let threshold = candidates
            .iter()
            .position(|&col| self.overlaps[col] < self.stimulus_threshold)
            .unwrap_or(candidates.len());
        self.winner_columns.clear();
        let winners_fraction = (self.density * self.num_columns as f32) as usize;
        self.winner_columns
            .extend(&candidates[..threshold.min(winners_fraction)]);
    }

    /// Allocates and configures each column’s synapses by sampling input bits within the potential radius:
    /// - Calls `map_potential()` for each column to select which input indices are in that column’s potential synapse pool.
    /// - Initializes each column’s synapses with random permanence values based on `init_connected_percentage`.
    ///
    /// Establishes each column’s initial “potential synapses,” which define where it can learn to connect.
    #[inline]
    pub fn connect_and_configure_inputs(&mut self) {
        let mut arr = vec![0usize; self.num_inputs];
        for column in 0..self.num_columns {
            let range = self.map_potential(column, true, &mut arr);
            self.synapses.init_column(
                column,
                &arr[range],
                self.init_connected_percentage,
                &self.synapse_permanence_options,
                &mut self.rand,
            );
            self.synapses.update_column_permanences(
                column,
                true,
                (self.stimulus_threshold + 0.5) as i32,
                &self.synapse_permanence_options,
            );
        }
    }

    /// Samples which input bits fall within a column’s potential radius, optionally wrapping around:
    /// - Determines the center input index for the column via `map_column()`.
    /// - Gathers all input indices within the potential radius from that center.
    /// - Randomly selects `potential_percentage` fraction of them as potential synapses.
    ///
    /// Defines which inputs a column could connect to, controlling local connectivity and receptive fields.
    #[inline]
    pub fn map_potential(
        &mut self,
        column: usize,
        wrap_around: bool,
        into: &mut [usize],
    ) -> Range<usize> {
        let center = self.map_column(column);
        let elements_around_center =
            self.input_topology
                .neighborhood(center, self.potential_radius as usize, wrap_around);
        let lower_bound = elements_around_center.size_hint().0;
        let size = self.potential_synapses(lower_bound);
        let mut sample = elements_around_center.choose_multiple(&mut self.rand, size);
        sample.shuffle(&mut self.rand);
        let len = sample.len();
        into[..len].copy_from_slice(&sample);
        0..len
    }

    /// Calculates how many potential synapses a column should have, given the potential radius neighborhood size.
    ///
    /// Ensures each column’s actual synapse count is consistent with the defined fraction of that local neighborhood.
    #[inline]
    pub fn potential_synapses(&self, input_size: usize) -> usize {
        ((input_size as f64 * self.potential_percentage) + 0.5) as usize
    }

    /// Maps a column index to the "center" input index in the input space:
    /// - Proportionally maps the column’s coordinates to the input grid coordinates.
    /// - Offset by half a cell for better distribution.
    /// - Clamps the result to the valid input range.
    ///
    /// This "center" determines where the potential radius is anchored for each column, shaping how local the input sampling is.
    #[inline]
    pub fn map_column(&self, column: usize) -> usize {
        let coords: Vec<usize> = self
            .column_topology
            .coordinates(column)
            .into_iter()
            .zip(self.column_dimensions.iter())
            .zip(self.input_dimensions.iter())
            .map(|((index, &col_dim), &in_dim)| {
                let new_index = ((index as f32 / col_dim as f32) * in_dim as f32
                    + (in_dim as f32 / col_dim as f32) * 0.5)
                    as usize;
                new_index.min(in_dim - 1)
            })
            .collect();
        self.input_topology.index_from_coordinates(&coords)
    }

    /// Generates the column objects and places them into `self.columns`.
    ///
    /// Represents the conceptual "slots" that the Spatial Pooler is training to detect features in the input.
    #[inline]
    pub fn gen_columns(&mut self) {
        self.columns = (0..self.num_columns).map(Column::new).collect();
    }

    /// Reinitializes the Synapses object (i.e., discards and recreates the entire synapse pool).
    ///
    /// Allows the Spatial Pooler to reset or re-generate synapses if needed (e.g., after parameter changes).
    #[inline]
    pub fn gen_column_potential_synapses(&mut self) {
        self.synapses = Synapses::new(self.num_columns, self.num_inputs);
    }
}
