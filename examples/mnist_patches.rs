//! This example reaches 95.55% accuracy on the MNIST dataset.
//! It uses 4x4 patch size and a 7x7 grid of non-overlapping patches.

use htm_rs::core::{sdr_classifier::SDRClassifier, spatial_pooler::SpatialPooler};
use mnist::{Mnist, MnistBuilder};
use rand::{rngs::StdRng, SeedableRng};

const IMAGE_DIM: usize = 28;
const PATCH_SIZE: usize = 4;

/// Extracts all 4x4 patches from a 28x28 MNIST image.
/// Returns 49 patches (7x7), each 16 bits in length (4x4 = 16).
fn extract_4x4_patches(image: &[u8]) -> Vec<Vec<bool>> {
    let patch_rows = IMAGE_DIM / PATCH_SIZE; // Using constants
    let patch_cols = IMAGE_DIM / PATCH_SIZE; // Using constants
    let mut patches = Vec::with_capacity(patch_rows * patch_cols);

    for pr in 0..patch_rows {
        for pc in 0..patch_cols {
            let mut patch = Vec::with_capacity(PATCH_SIZE * PATCH_SIZE);

            for r in 0..PATCH_SIZE {
                for c in 0..PATCH_SIZE {
                    let row = pr * PATCH_SIZE + r;
                    let col = pc * PATCH_SIZE + c;
                    let pixel = image[row * IMAGE_DIM + col];
                    patch.push(pixel > 127);
                }
            }
            patches.push(patch);
        }
    }
    patches
}

// Added write_patch function from the first snippet
/// OR-write a given patch into the 28×28 canvas.
fn write_patch(
    canvas: &mut [bool],
    x: usize,
    y: usize,
    dense_patch: &[bool], // Changed to &[bool] to accept Vec<bool> slices
) {
    for r in 0..PATCH_SIZE {
        for c in 0..PATCH_SIZE {
            if dense_patch[r * PATCH_SIZE + c] {
                canvas[(y + r) * IMAGE_DIM + (x + c)] = true;
            }
        }
    }
}

fn main() {
    println!("Loading MNIST dataset...");

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

    let mut r1 = StdRng::from_seed([42u8; 32]);
    let mut r2 = StdRng::from_seed([42u8; 32]);

    let training_len = trn_lbl.len();
    let testing_len = tst_lbl.len();

    let image_len = IMAGE_DIM * IMAGE_DIM;
    let input_dimensions_sp1 = vec![PATCH_SIZE * PATCH_SIZE];
    let column_dimensions_sp1 = vec![16];

    println!(
        "Initializing Spatial Pooler 1 with {} columns...",
        column_dimensions_sp1[0]
    );

    let mut spatial_pooler_1 = SpatialPooler::new(input_dimensions_sp1, column_dimensions_sp1);

    spatial_pooler_1.potential_radius = spatial_pooler_1.num_inputs as i32;
    spatial_pooler_1.synapse_permanence_options.active_increment = 0.001;
    spatial_pooler_1
        .synapse_permanence_options
        .inactive_decrement = 0.01;
    spatial_pooler_1.density = 0.6;
    spatial_pooler_1.stimulus_threshold = 1.0;
    spatial_pooler_1.synapse_permanence_options.connected = 0.4;
    spatial_pooler_1.synapse_permanence_options.max = 1.0;
    spatial_pooler_1.potential_percentage = 15.0 / spatial_pooler_1.potential_radius as f64;

    spatial_pooler_1.init(&mut r1);

    println!("Initializing Spatial Pooler 2...");

    let mut spatial_pooler_2 = SpatialPooler::new(vec![IMAGE_DIM * IMAGE_DIM], vec![64 * 64 * 4]); // SP2 input is full canvas

    spatial_pooler_2.potential_radius = spatial_pooler_2.num_inputs as i32;
    spatial_pooler_2.synapse_permanence_options.active_increment = 0.001;
    spatial_pooler_2
        .synapse_permanence_options
        .inactive_decrement = 0.01;
    spatial_pooler_2.stimulus_threshold = 2.9;
    spatial_pooler_2.synapse_permanence_options.connected = 0.2;
    spatial_pooler_2.synapse_permanence_options.max = 0.2;
    spatial_pooler_2.potential_percentage = 15.0 / spatial_pooler_2.potential_radius as f64;

    spatial_pooler_2.init(&mut r2);

    println!("Initializing Classifier...");

    let prediction_steps = vec![0];
    let learning_rate = 0.1;
    let column_size_sp2 = spatial_pooler_2.num_columns;

    let mut classifier = SDRClassifier::new(prediction_steps, learning_rate, column_size_sp2);

    println!(
        "Training Spatial Pooler and Classifier on {} images...",
        training_len
    );

    let mut canvas = vec![false; image_len];
    let patches_per_row_dim = IMAGE_DIM / PATCH_SIZE; // e.g. 28/4 = 7

    for i in 0..training_len {
        let image_slice = &trn_img[i * image_len..(i + 1) * image_len];
        let label = trn_lbl[i] as usize;

        canvas.fill(false);

        let dense_patches = extract_4x4_patches(image_slice);

        for (patch_idx, dense_patch) in dense_patches.iter().enumerate() {
            // Calculate x, y for the current patch based on its index
            let patch_grid_row = patch_idx / patches_per_row_dim;
            let patch_grid_col = patch_idx % patches_per_row_dim;
            let x = patch_grid_col * PATCH_SIZE;
            let y = patch_grid_row * PATCH_SIZE;

            spatial_pooler_1.compute(dense_patch, true);

            write_patch(&mut canvas, x, y, dense_patch);
        }

        // SP2 computes on the canvas built from dense patches
        spatial_pooler_2.compute(&canvas, true);

        classifier.compute(
            i as u32,
            label as usize,
            &spatial_pooler_2.winner_columns,
            true,
            false,
        );

        if (i + 1) % 1000 == 0 {
            // Progress indicator
            println!("  Trained {}/{} images", i + 1, training_len);
        }
    }

    println!("Training complete.");

    println!(
        "Testing Spatial Pooler and Classifier on {} images...",
        testing_len
    );

    let mut correct_predictions = 0;

    for i in 0..testing_len {
        let image_slice = &tst_img[i * image_len..(i + 1) * image_len];
        let label = tst_lbl[i] as usize;

        canvas.fill(false); // Reset canvas for each image

        let dense_patches = extract_4x4_patches(image_slice);

        for (patch_idx, dense_patch) in dense_patches.iter().enumerate() {
            let patch_grid_row = patch_idx / patches_per_row_dim;
            let patch_grid_col = patch_idx % patches_per_row_dim;
            let x = patch_grid_col * PATCH_SIZE;
            let y = patch_grid_row * PATCH_SIZE;

            spatial_pooler_1.compute(dense_patch, false);
            write_patch(&mut canvas, x, y, dense_patch);
        }

        spatial_pooler_2.compute(&canvas, false);

        let predictions = classifier.compute(
            i as u32,
            label as usize,
            &spatial_pooler_2.winner_columns,
            false,
            true,
        );

        for &(_step, ref probabilities) in &predictions {
            let prediction = probabilities
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0;

            if prediction == label {
                correct_predictions += 1;
            }
        }
        if (i + 1) % 1000 == 0 {
            // Progress indicator
            println!("  Tested {}/{} images", i + 1, testing_len);
        }
    }

    println!("Testing complete.");

    println!(
        "Accuracy: {:.2}%, Total: {} images, Correct: {}",
        100.0 * (correct_predictions as f32 / testing_len as f32),
        testing_len,
        correct_predictions
    );
}
