//! This example demonstrates how to use the HTM Spatial Pooler and SDR Classifier to classify
//! handwritten digits from the MNIST dataset. The Spatial Pooler is trained on the MNIST training
//! set, and the SDR Classifier is trained on the output of the Spatial Pooler. The trained models
//! are then tested on the MNIST test set.
//! 
//! This traditional HTM approach is a baseline for comparison with different techniques and pipelines.
//! The baseline reaches 95.52% accuracy in this configuration.

use mnist::{Mnist, MnistBuilder};
use htm_rs::core::{sdr_classifier::SDRClassifier, spatial_pooler::SpatialPooler};

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
    let input_dimensions = vec![image_size];
    let column_dimensions = vec![64 * 64 * 4];

    println!(
        "Initializing Spatial Pooler with {} columns...",
        column_dimensions[0]
    );

    let mut spatial_pooler = SpatialPooler::new(input_dimensions, column_dimensions);

    spatial_pooler.potential_radius = spatial_pooler.num_inputs as i32;
    spatial_pooler.synapse_permanence_options.active_increment = 0.001;
    spatial_pooler.synapse_permanence_options.inactive_decrement = 0.01;
    spatial_pooler.synapse_permanence_options.trim_threshold = 0.005;
    spatial_pooler.density = 1.0;
    spatial_pooler.stimulus_threshold = 2.9;
    spatial_pooler.synapse_permanence_options.connected = 0.2;
    spatial_pooler.synapse_permanence_options.max = 0.3;
    spatial_pooler.potential_percentage = 15.0 / spatial_pooler.potential_radius as f64;

    spatial_pooler.init();

    println!("Initializing Classifier...");

    let prediction_steps = vec![0];
    let learning_rate = 0.1;
    let column_size = spatial_pooler.num_columns;

    let mut classifier = SDRClassifier::new(prediction_steps, learning_rate, column_size);

    println!(
        "Training Spatial Pooler and Classifier on {} images...",
        training_len
    );

    let mut input_pattern_buffer = vec![false; spatial_pooler.num_inputs];

    for i in 0..training_len {
        let image = &trn_img[i * image_size..(i + 1) * image_size];
        let label = trn_lbl[i] as usize;

        for j in 0..image_size {
            input_pattern_buffer[j] = if image[j] > 127 { true } else { false };
        }

        spatial_pooler.compute(&input_pattern_buffer, true);
        classifier.compute(
            i as u32,
            label as usize,
            &spatial_pooler.winner_columns,
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
    input_pattern_buffer = vec![false; spatial_pooler.num_inputs];

    for i in 0..testing_len {
        let image = &tst_img[i * image_size..(i + 1) * image_size];
        let label = tst_lbl[i] as usize;

        for j in 0..image_size {
            input_pattern_buffer[j] = if image[j] > 127 { true } else { false };
        }

        spatial_pooler.compute(&input_pattern_buffer, false);
        let predictions = classifier.compute(
            i as u32,
            label as usize,
            &spatial_pooler.winner_columns,
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
