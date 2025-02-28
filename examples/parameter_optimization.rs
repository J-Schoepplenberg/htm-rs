use mnist::{Mnist, MnistBuilder};
use rand::Rng;
use htm_rs::core::{sdr_classifier::SDRClassifier, spatial_pooler::SpatialPooler};

#[derive(Debug, Copy, Clone)]
struct HyperParams {
    // Hyperparameters for the Spatial Pooler
    active_increment: f32,
    inactive_decrement: f32,
    trim_threshold: f32,
    density: f32,
    stimulus_threshold: f32,
    connected: f32,
    max: f32,
    potential_percentage_factor: f64,
    // Hyperparameter for the classifier
    learning_rate: f32,
}

// Given a hyperparameter configuration, initialize the Spatial Pooler and Classifier,
// train on the training set, and return the resulting accuracy on the test set.
fn evaluate_model(
    hyper: HyperParams,
    trn_img: &[u8],
    trn_lbl: &[u8],
    tst_img: &[u8],
    tst_lbl: &[u8],
) -> f32 {
    let training_len = trn_lbl.len();
    let testing_len = tst_lbl.len();
    let image_size = 28 * 28;

    // Global input dimensions and column count as in your base implementation.
    let input_dimensions = vec![image_size];
    let column_dimensions = vec![64 * 64 * 2];

    // Initialize and configure the spatial pooler with the hyperparameters.
    let mut sp = SpatialPooler::new(input_dimensions, column_dimensions);
    sp.potential_radius = sp.num_inputs as i32;
    sp.synapse_permanence_options.active_increment = hyper.active_increment;
    sp.synapse_permanence_options.inactive_decrement = hyper.inactive_decrement;
    sp.synapse_permanence_options.trim_threshold = hyper.trim_threshold;
    sp.density = hyper.density;
    sp.stimulus_threshold = hyper.stimulus_threshold;
    sp.synapse_permanence_options.connected = hyper.connected;
    sp.synapse_permanence_options.max = hyper.max;
    sp.potential_percentage = hyper.potential_percentage_factor / sp.potential_radius as f64;
    sp.init();

    // Initialize the classifier.
    let prediction_steps = vec![0];
    let column_size = sp.num_columns;
    let mut classifier = SDRClassifier::new(prediction_steps, hyper.learning_rate, column_size);

    let mut input_pattern_buffer = vec![false; sp.num_inputs];

    // Training loop.
    for i in 0..training_len {
        let image = &trn_img[i * image_size..(i + 1) * image_size];
        let label = trn_lbl[i] as usize;
        for j in 0..image_size {
            input_pattern_buffer[j] = image[j] > 127;
        }
        sp.compute(&input_pattern_buffer, true);
        classifier.compute(i as u32, label, &sp.winner_columns, true, false);
    }

    // Testing loop.
    let mut correct_predictions = 0;
    input_pattern_buffer = vec![false; sp.num_inputs];
    for i in 0..testing_len {
        let image = &tst_img[i * image_size..(i + 1) * image_size];
        let label = tst_lbl[i] as usize;
        for j in 0..image_size {
            input_pattern_buffer[j] = image[j] > 127;
        }
        sp.compute(&input_pattern_buffer, false);
        let predictions = classifier.compute(i as u32, label, &sp.winner_columns, false, true);

        for &(_step, ref probabilities) in &predictions {
            let pred = probabilities
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0;
            if pred == label {
                correct_predictions += 1;
            }
        }
    }

    100.0 * (correct_predictions as f32 / testing_len as f32)
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

    let mut rng = rand::rng();
    let num_trials = 100;
    let mut best_accuracy = 0.0;
    let mut best_params = None;

    // Random search loop.
    for trial in 0..num_trials {
        let hyper = HyperParams {
            active_increment: rng.random_range(0.005..0.03),
            inactive_decrement: rng.random_range(0.02..0.09),
            trim_threshold: rng.random_range(0.005..0.07),
            density: rng.random_range(0.5..1.3),
            stimulus_threshold: rng.random_range(2.0..3.5),
            connected: rng.random_range(0.2..0.45),
            max: rng.random_range(0.3..1.0),
            potential_percentage_factor: rng.random_range(12.0..17.0),
            learning_rate: rng.random_range(0.03..0.4),
        };

        println!("Trial {}: Testing hyperparameters: {:?}", trial + 1, hyper);

        let accuracy = evaluate_model(hyper, &trn_img, &trn_lbl, &tst_img, &tst_lbl);

        println!("Trial {}: Accuracy: {:.2}%", trial + 1, accuracy);

        if accuracy > best_accuracy {
            best_accuracy = accuracy;
            best_params = Some(hyper);
        }
    }

    println!("Best hyperparameters found: {:?}", best_params);
    println!("Best accuracy achieved: {:.2}%", best_accuracy);
}
