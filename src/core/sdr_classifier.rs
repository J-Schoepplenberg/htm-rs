//! SDRClassifier module for HTM.
//!
//! Learns to map sparse distributed representations (SDRs) to future discrete output classes (buckets)
//! for multiple prediction steps. It trains a 3D weight matrix using error-driven learning,
//! associating past input patterns with future target buckets. When using inference,
//! it can classify a given input pattern and also predict future bucket probabilities.
//!
//! Mechanism:
//! - Records recent sparse input patterns with timestamps (learn_iteration) during training.
//! - Maintains a 3D weight matrix linking active input bits to output buckets for each prediction step.
//! - For each new sample:
//!     - Updates weights to predict the current bucket from past patterns (for each step in `steps`).
//!     - Adjusts the weight matrix dynamically if new buckets are encountered.
//!
//! Inference:
//! - Computes probability distributions for future buckets (per step) using the current input pattern.
//! - Sums weights from active bits for each bucket, squares activations and normalizes them.
//! - This emphasizes strong activation values and yiels a probability distribution (sums to 1).

use std::collections::VecDeque;

/// A small value used to prevent division by zero.
const EPSILON: f32 = 0.001;
/// A classifier that learns a mapping from a sparse distributed representation (SDR)
/// to target buckets, with support for multi-step prediction.
pub struct SDRClassifier {
    /// Learning rate for weight updates.
    alpha: f32,

    /// The current learning iteration (offset-adjusted record number).
    learn_iteration: u32,

    /// The record number of the first sample; used to compute the relative learning iteration.
    record_offset: Option<u32>,

    /// Highest input index seen (used for sizing weight vectors).
    max_input_idx: usize,

    /// Highest bucket index seen (tracks the number of output classes).
    max_bucket_idx: usize,

    /// 3D weight matrix indexed as [prediction_step][bucket][input_feature].
    weight_matrix: Vec<Vec<Vec<f32>>>,

    /// The prediction steps (horizons) that the classifier learns.
    steps: Vec<u8>,

    /// History of activation patterns as (learn_iteration, input pattern) pairs.
    pattern_history: VecDeque<(u32, Vec<usize>)>,
}

impl SDRClassifier {
    /// Creates a new SDRClassifier.
    ///
    /// # Arguments
    ///
    /// * `steps` - Defines how many prediction steps the classifier should learn.
    /// * `alpha` - The learning rate.
    /// * `column_size` - The number of input features (used to size weight vectors).
    #[inline]
    pub fn new(steps: Vec<u8>, alpha: f32, column_size: usize) -> Self {
        let max_step = steps.iter().copied().max().unwrap_or(0) as usize + 1;
        Self {
            alpha,
            learn_iteration: 0,
            record_offset: None,
            max_input_idx: if column_size > 0 { column_size - 1 } else { 0 },
            max_bucket_idx: 0,
            weight_matrix: vec![vec![vec![0.0; column_size]; 1]; max_step],
            steps,
            pattern_history: VecDeque::with_capacity(max_step),
        }
    }

    /// Processes one input sample.
    ///
    /// This method:
    /// - Adjusts the learning iteration based on the record number.
    /// - Stores the activation pattern in history.
    /// - If learning is enabled, updates the weight matrix.
    /// - If inference is enabled, returns the probability distribution for each prediction step.
    ///
    /// # Arguments
    ///
    /// * `record_num` - The record number for the input sample.
    /// * `bucket_idx` - The target bucket index.
    /// * `pattern` - A slice of active indices from the SDR.
    /// * `learn` - Whether to perform weight updates.
    /// * `infer` - Whether to compute an inference.
    ///
    /// # Returns
    ///
    /// A vector of tuples `(step, probability_distribution)` for each prediction step.
    /// Alternatively, an empty vector if inference is disabled.
    #[inline]
    pub fn compute(
        &mut self,
        record_num: u32,
        bucket_idx: usize,
        pattern: &[usize],
        learn: bool,
        infer: bool,
    ) -> Vec<(u8, Vec<f32>)> {
        if self.record_offset.is_none() {
            self.record_offset = Some(record_num);
        }

        self.learn_iteration = record_num - self.record_offset.unwrap();

        if self.pattern_history.len() == self.pattern_history.capacity() {
            self.pattern_history.pop_back();
        }

        self.pattern_history
            .push_front((self.learn_iteration, pattern.to_vec()));

        if learn {
            if bucket_idx > self.max_bucket_idx {
                let additional = bucket_idx - self.max_bucket_idx;

                for &step in &self.steps {
                    let step_idx = step as usize;
                    self.weight_matrix[step_idx]
                        .extend((0..additional).map(|_| vec![0.0; self.max_input_idx + 1]));
                }

                self.max_bucket_idx = bucket_idx;
            }

            let mut error = vec![0.0; self.max_bucket_idx + 1];

            for &(iter, ref pattern) in &self.pattern_history {
                let n_steps = (self.learn_iteration - iter) as usize;

                if self.steps.contains(&(n_steps as u8)) {
                    self.infer_single_step(pattern, n_steps, &mut error);

                    for (i, err_val) in error.iter_mut().enumerate() {
                        *err_val = if i == bucket_idx { 1.0 } else { 0.0 } - *err_val;
                    }

                    for (bucket, weights) in self.weight_matrix[n_steps].iter_mut().enumerate() {
                        for &bit in pattern {
                            weights[bit] += self.alpha * error[bucket];
                        }
                    }
                }
            }
        }

        if infer {
            return self.infer(pattern);
        } else {
            return vec![(0, vec![0.0])];
        };
    }

    /// Performs inference over all prediction steps.
    ///
    /// Returns a vector of tuples `(step, probability_distribution)`, where each distribution sums to 1.
    #[inline]
    pub fn infer(&self, pattern: &[usize]) -> Vec<(u8, Vec<f32>)> {
        self.steps
            .iter()
            .map(|&step| {
                let mut distribution = vec![0.0; self.max_bucket_idx + 1];
                self.infer_single_step(pattern, step as usize, &mut distribution);
                (step, distribution)
            })
            .collect()
    }

    /// Performs inference for a single prediction step.
    ///
    /// The method computes the activation for each bucket by summing the weights corresponding to
    /// active bits in the input pattern, then squares and normalizes these activations to form
    /// a probability distribution.
    ///
    /// # Arguments
    ///
    /// * `pattern` - A slice of active indices.
    /// * `step` - The prediction step (horizon) to use.
    /// * `into` - A mutable slice where the computed activations are stored.
    #[inline]
    pub fn infer_single_step(&self, pattern: &[usize], step: usize, into: &mut [f32]) {
        let matrix = &self.weight_matrix[step];

        for (bucket, activation) in into.iter_mut().enumerate() {
            *activation = pattern
                .iter()
                .map(|&input_idx| matrix[bucket][input_idx])
                .sum();
        }

        for val in into.iter_mut() {
            if *val < EPSILON {
                *val = 0.0;
            } else {
                *val *= *val;
            }
        }

        let total: f32 = into.iter().sum();

        if total > EPSILON {
            for val in into.iter_mut() {
                *val /= total;
            }
        }
    }
}
