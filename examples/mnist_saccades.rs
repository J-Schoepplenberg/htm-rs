//! This example demonstrates how to identify MNIST digits.
//! The approach reaches around 95.18% accuracy in this configuration.
//! 
//! It implements a hierarchical pipeline for MNIST digit classification
//! using two layers of spatial poolers followed by an SDR classifier.
//! 
//! MNIST Data:
//! - The MNIST dataset is loaded with separate training and test sets.
//! 
//! Patch Extraction:
//! - Each 28x28 image is divided into 49 non-overlapping 4x4 patches.
//! - Each patch is converted into a 16-bit boolean vector.
//! - Each bit indicates if the pixel (0-255) exceeds a threshold (127).
//! 
//! First Layer:
//! - The patches are fed into the first spatial pooler layer.
//! - It is configured with 16 columns, matching the 4x4 patch structure.
//! - This produces an SDR in the form of winner column indices.
//! 
//! Aggregation:
//! - The patch SDRs are aggregated back into a 28x28 boolean array.
//! - This preserves the original spatial arrangement of the image.
//! 
//! Second Layer:
//! - The aggregated SDR is fed into a second spatial pooler layer.
//! - This integrates features across the entire image.
//! - The layer learns higher-level representations by processing the global spatial layout.
//! 
//! SDR Classifier:
//! - An SDRClassifier is trained on the output of the second spatial pooler.
//! - It uses the final SDR to predict the digit class for each image.


use mnist::{Mnist, MnistBuilder};
use htm_rs::core::{sdr_classifier::SDRClassifier, spatial_pooler::SpatialPooler};

/// Extracts all 4x4 patches from a 28x28 MNIST image.
/// Returns 49 patches (7x7), each 16 bits in length (4x4 = 16).
fn extract_4x4_patches(image: &[u8]) -> Vec<Vec<bool>> {
    let patch_rows = 7;
    let patch_cols = 7;
    let mut patches = Vec::with_capacity(patch_rows * patch_cols);

    for pr in 0..patch_rows {
        for pc in 0..patch_cols {
            let mut patch = Vec::with_capacity(16);
            
            for r in 0..4 {
                for c in 0..4 {
                    let row = pr * 4 + r;
                    let col = pc * 4 + c;
                    let pixel = image[row * 28 + col];
                    patch.push(pixel > 127);
                }
            }

            patches.push(patch);
        }
    }

    patches
}

/// Converts a vector of patch SDR indices into a global 28x28 boolean array.
fn indices_to_global_sdr(patches: &Vec<Vec<usize>>) -> [bool; 28 * 28] {
    let mut result = [false; 28 * 28];

    let patches_per_row = 28 / 4;

    for (patch_index, patch) in patches.iter().enumerate() {
        let patch_row = patch_index / patches_per_row;
        let patch_col = patch_index % patches_per_row;
        let patch_offset = (patch_row * 4 * 28) + (patch_col * 4);

        for &local_index in patch {
            let local_row = local_index / 4;
            let local_col = local_index % 4;
            let global_index = patch_offset + (local_row * 28) + local_col;

            if global_index < 28 * 28 {
                result[global_index] = true;
            }
        }
    }

    result
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

    let training_len = trn_lbl.len();
    let testing_len = tst_lbl.len();

    let image_size = 28 * 28;
    let input_dimensions = vec![16];
    let column_dimensions = vec![16];

    println!(
        "Initializing Spatial Pooler with {} columns...",
        column_dimensions[0]
    );

    let mut spatial_pooler_1 = SpatialPooler::new(input_dimensions, column_dimensions);

    spatial_pooler_1.potential_radius = spatial_pooler_1.num_inputs as i32;
    spatial_pooler_1.synapse_permanence_options.active_increment = 0.001;
    spatial_pooler_1.synapse_permanence_options.inactive_decrement = 0.01;
    spatial_pooler_1.density = 0.6;
    spatial_pooler_1.stimulus_threshold = 1.0;
    spatial_pooler_1.synapse_permanence_options.connected = 0.4;
    spatial_pooler_1.synapse_permanence_options.max = 1.0;
    spatial_pooler_1.potential_percentage = 15.0 / spatial_pooler_1.potential_radius as f64;

    spatial_pooler_1.init();

    println!("Initializing Spatial Pooler 2...");

    let mut spatial_pooler_2 = SpatialPooler::new(vec![28 * 28], vec![64 * 64 * 4]);
    
    spatial_pooler_2.potential_radius = spatial_pooler_2.num_inputs as i32;
    spatial_pooler_2.synapse_permanence_options.active_increment = 0.001;
    spatial_pooler_2.synapse_permanence_options.inactive_decrement = 0.01;
    spatial_pooler_2.stimulus_threshold = 2.9;
    spatial_pooler_2.synapse_permanence_options.connected = 0.2;
    spatial_pooler_2.synapse_permanence_options.max = 0.2;
    spatial_pooler_2.potential_percentage = 15.0 / spatial_pooler_2.potential_radius as f64;

    spatial_pooler_2.init();

    println!("Initializing Classifier...");

    let prediction_steps = vec![0];
    let learning_rate = 0.1;
    let column_size = spatial_pooler_2.num_columns;

    let mut classifier = SDRClassifier::new(prediction_steps, learning_rate, column_size);

    println!(
        "Training Spatial Pooler and Classifier on {} images...",
        training_len
    );

    for i in 0..training_len {
        let image = &trn_img[i * image_size..(i + 1) * image_size];
        let label = trn_lbl[i] as usize;

        let patches = extract_4x4_patches(image);

        let mut batch = Vec::new();

        for patch in patches {
            spatial_pooler_1.compute(&patch, true);
            batch.push(spatial_pooler_1.winner_columns.clone());
        }

        let bools = indices_to_global_sdr(&batch);

        spatial_pooler_2.compute(&bools, true);

        classifier.compute(
            i as u32,
            label as usize,
            &spatial_pooler_2.winner_columns,
            true,
            false,
        );
    }

    println!("Training complete.");

    println!(
        "Testing Spatial Pooler and Classifier on {} images...",
        testing_len
    );

    let mut correct_predictions = 0;

    for i in 0..testing_len {
        let image = &tst_img[i * image_size..(i + 1) * image_size];
        let label = tst_lbl[i] as usize;

        let patches = extract_4x4_patches(image);

        let mut batch = Vec::new();

        for patch in patches {
            spatial_pooler_1.compute(&patch, false);
            batch.push(spatial_pooler_1.winner_columns.clone());
        }

        let bools = indices_to_global_sdr(&batch);

        spatial_pooler_2.compute(&bools, false);

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
    }

    println!("Testing complete.");

    println!(
        "Accuracy: {:.2}%, Total: {} images, Correct: {}",
        100.0 * (correct_predictions as f32 / testing_len as f32),
        testing_len,
        correct_predictions
    );
}
