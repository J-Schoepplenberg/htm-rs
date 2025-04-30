//! Topology represents an N-dimensional space through a list of dimensions and corresponding stride values.
//! The struct provides methods to convert between linear indices and coordinates in this N-dimensional space,
//! and also offers a way to iterate over neighborhoods of that space within a radius of a given center index.
//!
//! In the HTM Spatial Pooler context, the input and column spaces are N-dimensional.
//! SP uses typical neighborhood notions such as local inhibition and connections within a radius.
//! Topology helps manage the relationship between array-like indices and coordinates in these spaces.
//! It also enables arbitrary number of dimensions by keeping track of each dimension size and strides.

use serde::{Deserialize, Serialize};
use std::cmp::{max, min};

/// Represents the shape of an N-dimensional space, along with precomputed stride values for
/// linear index conversions. The `dims` field stores the size of each dimension, while `strides`
/// stores the cumulative product of dimension sizes to enable fast index calculations.
#[derive(Debug, Serialize, Deserialize)]
pub struct Topology {
    dims: Vec<usize>,
    strides: Vec<usize>,
}

impl Topology {
    /// Creates a new `Topology` from a slice of dimension sizes.
    #[inline]
    pub fn new(dimensions: &[usize]) -> Self {
        let dims = dimensions.to_vec();
        let strides = Self::strides(&dims);

        Self { dims, strides }
    }

    /// Computes the stride values for each dimension in a given slice of dimension sizes.
    /// Strides are used to convert coordinates in N-dimensional space into a single linear index.
    #[inline]
    fn strides(dims: &[usize]) -> Vec<usize> {
        let mut strides = vec![1; dims.len()];

        for i in (0..dims.len() - 1).rev() {
            strides[i] = strides[i + 1] * dims[i + 1];
        }

        strides
    }

    /// Converts a linear index into its corresponding set of coordinates in the topology's N-dimensional space.
    /// Each element of the returned `Vec<usize>` is the coordinate along one of the dimensions, in order.
    #[inline]
    pub fn coordinates(&self, index: usize) -> Vec<usize> {
        let mut remainder = index;

        self.strides
            .iter()
            .map(|&stride| {
                let coord = remainder / stride;
                remainder %= stride;
                coord
            })
            .collect()
    }

    /// Converts a set of coordinates in the topology's N-dimensional space to a single linear index.
    /// The length of `coords` must match the number of dimensions in the topology.
    #[inline]
    pub fn index_from_coordinates(&self, coords: &[usize]) -> usize {
        coords.iter().zip(&self.strides).map(|(&c, &s)| c * s).sum()
    }

    /// Returns an iterator over the neighborhood of indices within a given `radius` of the
    /// specified `center` index. If `wrapping` is true, the neighborhood wraps around edges of
    /// the topology dimensions; otherwise, it is clipped at boundaries.
    #[inline]
    pub fn neighborhood(&self, center: usize, radius: usize, wrapping: bool) -> NeighborhoodIter {
        let center_coords = self.coordinates(center);
        let radius = radius as isize;

        let bounds: Vec<(isize, isize)> = center_coords
            .iter()
            .zip(&self.dims)
            .map(|(&c, &dim)| {
                let c = c as isize;
                let dim = dim as isize;

                if wrapping {
                    (c - radius, c - radius + dim)
                } else {
                    (max(c - radius, 0), min(c + radius + 1, dim))
                }
            })
            .collect();

        let current = bounds.iter().map(|&(low, _)| low).collect();

        NeighborhoodIter {
            topology: self,
            bounds,
            current: Some(current),
            wrapping,
        }
    }
}

/// An iterator that yields all valid indices within a neighborhood of a central index in the `Topology`.
/// The neighborhood is defined by a radius around the center, with optional wrapping behavior.
pub struct NeighborhoodIter<'a> {
    topology: &'a Topology,
    bounds: Vec<(isize, isize)>,
    current: Option<Vec<isize>>,
    wrapping: bool,
}

impl Iterator for NeighborhoodIter<'_> {
    type Item = usize;

    /// Returns the next index within the neighborhood. When all indices have been visited, it returns `None`.
    fn next(&mut self) -> Option<Self::Item> {
        let current = self.current.as_mut()?;

        let coords: Vec<usize> = current
            .iter()
            .enumerate()
            .map(|(i, &val)| {
                let dim = self.topology.dims[i] as isize;

                if self.wrapping {
                    val.rem_euclid(dim) as usize
                } else {
                    val.clamp(0, dim - 1) as usize
                }
            })
            .collect();

        let result = self.topology.index_from_coordinates(&coords);

        for i in (0..current.len()).rev() {
            if current[i] + 1 < self.bounds[i].1 {
                current[i] += 1;

                current
                    .iter_mut()
                    .enumerate()
                    .skip(i + 1)
                    .for_each(|(j, item)| *item = self.bounds[j].0);

                return Some(result);
            }
        }

        self.current.take();

        Some(result)
    }

    /// Provides the lower and upper bounds for the remaining elements in the neighborhood iterator.
    fn size_hint(&self) -> (usize, Option<usize>) {
        let count = self
            .bounds
            .iter()
            .map(|&(low, high)| (high - low) as usize)
            .product();

        (count, Some(count))
    }
}
