//! Saccadic‑attention HTM pipeline for handwritten‑digit recognition
//! -----------------------------------------------------------------
//! 28×28 greyscale digit
//!         │
//! step 1:  choose a small “fovea” (4×4 patch) at the digit’s centre of mass
//!         │
//! step 2:  encode that patch with a tiny Spatial Pooler (SP₁)
//!         │
//! step 3:  feed SP₁ sparse output into a Temporal Memory (TM) → sequence state
//!         │
//! step 4:  TM predicts *which patch* is likely to appear next
//!         │
//! step 5:  an attention policy scores the 24 candidates around the current
//!         │     fixation and jumps to the location with maximal overlap
//!         │     between TM prediction and actual patch code
//!         │               ↑
//!         │   loop 4. and 5. up to 49 times   (saccades)
//!         │
//! step 6:  OR all visited 4×4 patches into a 28×28 binary canvas
//!         │
//! step 7:  encode that canvas with a large Spatial Pooler (SP₂)
//!         │
//! step 8:  classify the resulting SDR with an SDRClassifier (supervised)
//!                ↓
//!           predicted digit
//! -----------------------------------------------------------------

use fxhash::FxHashSet;
use htm_rs::core::{
    sdr_classifier::SDRClassifier,
    spatial_pooler::SpatialPooler,
    temporal_memory::{TemporalMemory, TemporalMemoryParams},
};
use image::{DynamicImage, GrayImage, Rgb};
use imageproc::{drawing::draw_hollow_rect_mut, rect::Rect};
use mnist::{Mnist, MnistBuilder};
use std::collections::HashSet;

const IMAGE_DIM: usize = 28;
const PATCH_SIZE: usize = 4;
const MAX_SACCADES: usize = 49;
const MIN_ACTIVE_PIXELS: usize = 1;

/// Extract a 4×4 boolean patch at (x,y) (top‑left corner, 0‑based).
fn extract_patch(image: &[u8], x: usize, y: usize) -> [bool; PATCH_SIZE * PATCH_SIZE] {
    let mut p = [false; PATCH_SIZE * PATCH_SIZE];

    for r in 0..PATCH_SIZE {
        for c in 0..PATCH_SIZE {
            p[r * PATCH_SIZE + c] = image[(y + r) * IMAGE_DIM + (x + c)] > 127;
        }
    }

    p
}

/// Centre of mass of all on‑pixels.
fn first_saccade(image: &[u8]) -> (usize, usize) {
    let mut sx = 0usize;
    let mut sy = 0usize;
    let mut n = 0usize;

    for y in 0..IMAGE_DIM {
        for x in 0..IMAGE_DIM {
            if image[y * IMAGE_DIM + x] > 127 {
                sx += x;
                sy += y;
                n += 1;
            }
        }
    }

    if n == 0 {
        return (
            IMAGE_DIM / 2 - PATCH_SIZE / 2,
            IMAGE_DIM / 2 - PATCH_SIZE / 2,
        );
    }

    let cx = sx / n;
    let cy = sy / n;

    (
        cx.saturating_sub(PATCH_SIZE / 2)
            .min(IMAGE_DIM - PATCH_SIZE),
        cy.saturating_sub(PATCH_SIZE / 2)
            .min(IMAGE_DIM - PATCH_SIZE),
    )
}

/// generate candidate saccades around (x,y) with strides {±2, ±4} pixels
fn candidate_locations(x: usize, y: usize) -> Vec<(usize, usize)> {
    const OFFSETS: [isize; 5] = [-4, -2, 0, 2, 4];
    let mut v = Vec::with_capacity(24);

    for &dx in &OFFSETS {
        for &dy in &OFFSETS {
            if dx == 0 && dy == 0 {
                continue;
            }

            let nx = x as isize + dx;
            let ny = y as isize + dy;

            if nx >= 0
                && ny >= 0
                && nx <= (IMAGE_DIM - PATCH_SIZE) as isize
                && ny <= (IMAGE_DIM - PATCH_SIZE) as isize
            {
                v.push((nx as usize, ny as usize));
            }
        }
    }

    v
}

/// Overlap of two sparse SDRs (sets of indices).
fn sdr_overlap(a: &[usize], b: &[usize]) -> usize {
    let sa: HashSet<_> = a.iter().collect();
    b.iter().filter(|&&i| sa.contains(&i)).count()
}

/// OR‑write a given patch into the 28×28 canvas.
fn write_patch(
    canvas: &mut [bool],
    x: usize,
    y: usize,
    dense_patch: &[bool; PATCH_SIZE * PATCH_SIZE],
) {
    for r in 0..PATCH_SIZE {
        for c in 0..PATCH_SIZE {
            if dense_patch[r * PATCH_SIZE + c] {
                canvas[(y + r) * IMAGE_DIM + (x + c)] = true;
            }
        }
    }
}

/// Draw the given image with red rectangles around the saccades.
fn _visualize(img: &[u8], saccades: &[(usize, usize)], out: &str) {
    let g = GrayImage::from_vec(IMAGE_DIM as u32, IMAGE_DIM as u32, img.to_vec()).unwrap();
    let mut rgb = DynamicImage::ImageLuma8(g).to_rgb8();
    let red = Rgb([255, 0, 0]);

    for &(x, y) in saccades {
        let rect = Rect::at(x as i32, y as i32).of_size(PATCH_SIZE as u32, PATCH_SIZE as u32);
        draw_hollow_rect_mut(&mut rgb, rect, red);
    }

    rgb.save(out).unwrap();
}

fn main() {
    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(60_000)
        .test_set_length(10_000)
        .finalize();

    let mut sp1 = SpatialPooler::new(vec![PATCH_SIZE * PATCH_SIZE], vec![16]);
    sp1.potential_radius = sp1.num_inputs as i32;
    sp1.synapse_permanence_options.active_increment = 0.001;
    sp1.synapse_permanence_options.inactive_decrement = 0.01;
    sp1.synapse_permanence_options.connected = 0.4;
    sp1.synapse_permanence_options.max = 1.0;
    sp1.stimulus_threshold = 1.0;
    sp1.density = 0.6;
    sp1.potential_percentage = 15.0 / sp1.potential_radius as f64;
    sp1.init();

    let tm_params = TemporalMemoryParams {
        activation_threshold: 2,
        connected_permanence: 0.5,
        learning_threshold: 4,
        initial_permanence: 0.4,
        permanence_increment: 0.1,
        permanence_decrement: 0.05,
        predicted_decrement: 0.01,
        synapse_sample_size: 10,
        learning_enabled: true,
    };

    let mut tm = TemporalMemory::new(sp1.num_columns, 8, tm_params);

    let mut sp2 = SpatialPooler::new(vec![IMAGE_DIM * IMAGE_DIM], vec![64 * 64 * 4]);
    sp2.potential_radius = sp2.num_inputs as i32;
    sp2.synapse_permanence_options.connected = 0.2;
    sp2.synapse_permanence_options.max = 0.2;
    sp2.synapse_permanence_options.active_increment = 0.001;
    sp2.synapse_permanence_options.inactive_decrement = 0.01;
    sp2.stimulus_threshold = 2.9;
    sp2.potential_percentage = 15.0 / sp2.potential_radius as f64;
    sp2.init();

    let mut clf = SDRClassifier::new(vec![0], /* α */ 0.1, sp2.num_columns);

    println!("Training...");

    let image_len = IMAGE_DIM * IMAGE_DIM;
    let mut canvas = vec![false; image_len];

    for (i, (&label, img)) in trn_lbl.iter().zip(trn_img.chunks(image_len)).enumerate() {
        tm.reset_state();
        canvas.fill(false);

        let (mut x, mut y) = first_saccade(img);

        let mut visited = FxHashSet::default();

        visited.insert((x, y));

        for _ in 0..MAX_SACCADES {
            let patch = extract_patch(img, x, y);

            sp1.compute(&patch, true);
            tm.step(sp1.winner_columns());

            write_patch(&mut canvas, x, y, &patch);

            let prediction = tm.predicted_columns();

            let mut best_candidate = None;
            let mut best_score = 0usize;

            for (nx, ny) in candidate_locations(x, y) {
                if visited.contains(&(nx, ny)) {
                    continue;
                }

                let candidate_patch = extract_patch(img, nx, ny);

                if candidate_patch.iter().filter(|&&b| b).count() < MIN_ACTIVE_PIXELS {
                    continue;
                }

                sp1.compute(&candidate_patch, false);

                let candidate = sp1.winner_columns();

                let candidate_score = if prediction.is_empty() {
                    candidate_patch.iter().filter(|&&b| b).count()
                } else {
                    sdr_overlap(&prediction, candidate)
                };

                if candidate_score > best_score {
                    best_candidate = Some((nx, ny));
                    best_score = candidate_score;
                }
            }

            match best_candidate {
                Some(p) => {
                    x = p.0;
                    y = p.1;
                    visited.insert(p);
                }

                None => break,
            }
        }

        sp2.compute(&canvas, true);
        clf.compute(i as u32, label as usize, sp2.winner_columns(), true, false);

        if (i + 1) % 1_000 == 0 {
            println!("  trained {}", i + 1);
        }
    }

    println!("Testing...");

    let mut correct = 0usize;

    for (i, (&label, img)) in tst_lbl.iter().zip(tst_img.chunks(image_len)).enumerate() {
        tm.reset_state();
        canvas.fill(false);

        let (mut x, mut y) = first_saccade(img);

        //let mut saccades = vec![(x, y)];
        let mut visited = FxHashSet::default();

        visited.insert((x, y));

        for _ in 0..MAX_SACCADES {
            let patch = extract_patch(img, x, y);

            sp1.compute(&patch, false);
            tm.step(sp1.winner_columns());

            write_patch(&mut canvas, x, y, &patch);

            let prediction = tm.predicted_columns();

            let mut best_candidate = None;
            let mut best_score = 0usize;

            for (nx, ny) in candidate_locations(x, y) {
                if visited.contains(&(nx, ny)) {
                    continue;
                }

                let candidate_patch = extract_patch(img, nx, ny);

                if candidate_patch.iter().filter(|&&b| b).count() < MIN_ACTIVE_PIXELS {
                    continue;
                }

                sp1.compute(&candidate_patch, false);

                let candidate = sp1.winner_columns();

                let candidate_score = if prediction.is_empty() {
                    candidate_patch.iter().filter(|&&b| b).count()
                } else {
                    sdr_overlap(&prediction, candidate)
                };

                if candidate_score > best_score {
                    best_candidate = Some((nx, ny));
                    best_score = candidate_score;
                }
            }

            match best_candidate {
                Some(p) => {
                    x = p.0;
                    y = p.1;
                    visited.insert(p);
                    //saccades.push((x, y));
                }

                None => break,
            }
        }

        sp2.compute(&canvas, false);

        let probabilities =
            clf.compute(i as u32, label as usize, sp2.winner_columns(), false, true);
            
        let predicted_label = probabilities[0]
            .1
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;

        if predicted_label == label as usize {
            correct += 1;
        }

        /* let file_name = format!("mnist_wtf_{}.png", i);
        _visualize(img, &saccades, &file_name); */

        if (i + 1) % 1_000 == 0 {
            println!("  tested {}", i + 1);
        }
    }

    let acc = 100.0 * correct as f32 / 10_000.0;
    println!("Accuracy: {:.2}%  ({} / 10000)", acc, correct);
}
