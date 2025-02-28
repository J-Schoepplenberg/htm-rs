//! A `Column` in HTM represents one feature detector or receptive field in the Spatial Pooler.
//! 
//! Biological inspiration:
//! Columns in HTM are inspired by cortical mini-colmuns found in the brain.
//! They consist of a group of neurons, which in HTM are modeled as "cells".
//! 
//! Meaning in HTM:
//! Each column is responsible for receing input from a random subset of the input space 
//! (via its potential synapses), computing its overlap score with the current input, 
//! and competing with other columns (via inhibition of neighbors within a radius) to become the active 
//! "winner columns". Over multiple learning iterations, each column adjusts its synapse permanence values 
//! (strength of the connections to input bits) to become selective for particular input patterns. 
//! Together, all columns produce a sparse distributed representation of the input space.

/// Represents a cortical column in the HTM model.
#[derive(Debug)]
pub struct Column {
    /// The index of the column.
    pub index: usize,
}

impl Column {
    /// Creates a new Column.
    pub fn new(index: usize) -> Self {
        Self { index }
    }
}