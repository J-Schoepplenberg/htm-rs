use std::collections::HashSet;

use htm_rs::core::{
    sdr_classifier::SDRClassifier,
    spatial_pooler::SpatialPooler,
    temporal_memory::{TemporalMemory, TemporalMemoryParams},
};
use image::{DynamicImage, GrayImage, Rgb};
use imageproc::{drawing::draw_hollow_rect_mut, rect::Rect};
use mnist::{Mnist, MnistBuilder};
use rand::seq::IndexedRandom;

const MAX_SACCADES: usize = 49;
const MIN_ACTIVE_PIXELS: usize = 1;
const IMAGE_DIM: usize = 28;
const PATCH_SIZE: usize = 4;

/// Extract a single 4x4 patch from a 28x28 MNIST image:
/// - Assumes (x, y) are grid-aligned (multiples of 4).
fn extract_patch(image: &[u8], x: usize, y: usize) -> [bool; 16] {
    let mut patch = [false; 16];
    for row in 0..PATCH_SIZE {
        for col in 0..PATCH_SIZE {
            let pixel = image[(y + row) * IMAGE_DIM + (x + col)];
            patch[row * PATCH_SIZE + col] = pixel > 127;
        }
    }
    patch
}

/// Finds a starting location for the first patch:
/// - Scans the image for a pixel above threshold.
/// - Snaps the coordinate to the nearest 4x4 grid location (i.e. multiples of 4).
fn find_start_point(image: &[u8]) -> (usize, usize) {
    for y in 0..IMAGE_DIM {
        for x in 0..IMAGE_DIM {
            if image[y * IMAGE_DIM + x] > 127 {
                // Snap to grid
                let x_grid = (x / PATCH_SIZE) * PATCH_SIZE;
                let y_grid = (y / PATCH_SIZE) * PATCH_SIZE;
                return (
                    x_grid.min(IMAGE_DIM - PATCH_SIZE),
                    y_grid.min(IMAGE_DIM - PATCH_SIZE),
                );
            }
        }
    }
    (
        IMAGE_DIM / 2 - PATCH_SIZE / 2,
        IMAGE_DIM / 2 - PATCH_SIZE / 2,
    )
}

/// Returns candidate locations for the next patch:
/// - Only returns grid-aligned candidates that do not exceed the bounds.
fn candidate_locations(x: usize, y: usize) -> Vec<(usize, usize)> {
    let mut candidates = Vec::new();
    // Define shifts in multiples of PATCH_SIZE
    let shifts = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ];
    for &(dx, dy) in &shifts {
        // Since our grid steps are PATCH_SIZE, add dx * PATCH_SIZE
        let new_x = x as isize + dx * PATCH_SIZE as isize;
        let new_y = y as isize + dy * PATCH_SIZE as isize;
        if new_x >= 0
            && new_x <= (IMAGE_DIM - PATCH_SIZE) as isize
            && new_y >= 0
            && new_y <= (IMAGE_DIM - PATCH_SIZE) as isize
        {
            candidates.push((new_x as usize, new_y as usize));
        }
    }
    candidates
}

/// Computes the overlap between two SDRs given as sparse lists of active indices.
fn sdr_overlap(sdr1: &[usize], sdr2: &[usize]) -> usize {
    let set1: std::collections::HashSet<_> = sdr1.iter().cloned().collect();
    sdr2.iter().filter(|&&x| set1.contains(&x)).count()
}

/// Visualizes the MNIST image with patch locations drawn.
fn visualize(image: &[u8], locations: &[(usize, usize)], output_path: &str) {
    let gray_img = GrayImage::from_vec(IMAGE_DIM as u32, IMAGE_DIM as u32, image.to_vec()).unwrap();
    let mut img = DynamicImage::ImageLuma8(gray_img).to_rgb8();
    let red = Rgb([255, 0, 0]);
    for &(x, y) in locations {
        let rect = Rect::at(x as i32, y as i32).of_size(PATCH_SIZE as u32, PATCH_SIZE as u32);
        draw_hollow_rect_mut(&mut img, rect, red);
    }
    img.save(output_path).unwrap();
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
        .training_set_length(10_000)
        .test_set_length(10)
        .finalize();

    let training_len = trn_lbl.len();
    let testing_len = tst_lbl.len();
    let image_size = IMAGE_DIM * IMAGE_DIM;

    // Our patch has 16 inputs.
    let input_dimensions = vec![16];
    let column_dimensions = vec![16];

    println!("Initializing Spatial Pooler...");
    let mut spatial_pooler = SpatialPooler::new(input_dimensions, column_dimensions);
    spatial_pooler.potential_radius = spatial_pooler.num_inputs as i32;
    spatial_pooler.synapse_permanence_options.active_increment = 0.001;
    spatial_pooler.synapse_permanence_options.inactive_decrement = 0.01;
    spatial_pooler.density = 0.6;
    spatial_pooler.stimulus_threshold = 1.0;
    spatial_pooler.synapse_permanence_options.connected = 0.4;
    spatial_pooler.synapse_permanence_options.max = 1.0;
    spatial_pooler.potential_percentage = 15.0 / spatial_pooler.potential_radius as f64;
    spatial_pooler.init();

    println!("Initializing Temporal Memory...");
    let cells_per_column = 8;
    let tm_params = TemporalMemoryParams {
        activation_threshold: 5,
        connected_permanence: 0.5,
        learning_threshold: 4,
        initial_permanence: 0.4,
        permanence_increment: 0.1,
        permanence_decrement: 0.05,
        predicted_decrement: 0.01,
        synapse_sample_size: 15,
        learning_enabled: true,
    };
    let mut temporal_memory =
        TemporalMemory::new(spatial_pooler.num_columns, cells_per_column, tm_params);

    println!("Initializing Classifier...");
    let prediction_steps = vec![0];
    let learning_rate = 0.1;
    let column_size = spatial_pooler.num_columns;
    let mut classifier = SDRClassifier::new(prediction_steps, learning_rate, column_size);

    println!("Training on {} images...", training_len);

    // Use random tie-breaking if multiple candidates have the same score.
    let mut rng = rand::rng();

    for i in 0..training_len {
        let image = &trn_img[i * image_size..(i + 1) * image_size];
        let label = trn_lbl[i] as usize;
        let (mut x, mut y) = find_start_point(image);

        let mut pool = Vec::new();

        let mut visited: HashSet<(usize, usize)> = HashSet::new();
        visited.insert((x, y));

        for _ in 0..MAX_SACCADES {
            let patch = extract_patch(image, x, y);
            spatial_pooler.compute(&patch, true);
            temporal_memory.step(&spatial_pooler.winner_columns());
            pool.push(temporal_memory.winner_columns());

            let predicted_sdr = temporal_memory.predicted_columns();

            let candidates = candidate_locations(x, y);
            let mut best_candidates = Vec::new();
            let mut max_overlap = 0;

            for (x1, y1) in candidates {
                if visited.contains(&(x1, y1)) {
                    continue;
                }

                let candidate_patch = extract_patch(image, x1, y1);

                let active_count = candidate_patch.iter().filter(|&&bit| bit).count();

                if active_count < MIN_ACTIVE_PIXELS {
                    continue;
                }

                spatial_pooler.compute(&candidate_patch, false);
                let candidate_sdr = spatial_pooler.winner_columns();
                let overlap = sdr_overlap(&predicted_sdr, &candidate_sdr);

                if overlap > max_overlap {
                    max_overlap = overlap;
                    best_candidates.clear();
                    best_candidates.push((x1, y1));
                } else if overlap == max_overlap {
                    best_candidates.push((x1, y1));
                }
            }

            if best_candidates.is_empty() {
                break;
            }

            let best_candidate = *best_candidates.choose(&mut rng).unwrap();

            if best_candidate == (x, y) {
                break;
            }

            x = best_candidate.0;
            y = best_candidate.1;
            visited.insert((x, y));
        }

        classifier.compute(i as u32, label as usize, &pool.concat(), true, false);

        if (i + 1) % 1000 == 0 {
            println!("Training image {}/{}", i + 1, training_len);
        }
    }

    println!("Training complete.");
    println!("Testing on {} images...", testing_len);

    let mut correct_predictions = 0;
    let mut saccades = Vec::new();

    for i in 0..testing_len {
        let image = &tst_img[i * image_size..(i + 1) * image_size];
        let label = tst_lbl[i] as usize;
        let (mut x, mut y) = find_start_point(image);

        saccades.clear();

        let mut pool = Vec::new();

        let mut visited: HashSet<(usize, usize)> = HashSet::new();
        visited.insert((x, y));

        for _ in 0..MAX_SACCADES {
            saccades.push((x, y));
            let patch = extract_patch(image, x, y);
            spatial_pooler.compute(&patch, true);
            temporal_memory.step(&spatial_pooler.winner_columns());
            pool.push(temporal_memory.winner_columns());

            let predicted_sdr = temporal_memory.predicted_columns();

            let candidates = candidate_locations(x, y);
            let mut best_candidates = Vec::new();
            let mut max_overlap = 0;

            for (x1, y1) in candidates {
                if visited.contains(&(x1, y1)) {
                    continue;
                }

                let candidate_patch = extract_patch(image, x1, y1);

                let active_count = candidate_patch.iter().filter(|&&bit| bit).count();

                if active_count < MIN_ACTIVE_PIXELS {
                    continue;
                }

                spatial_pooler.compute(&candidate_patch, false);
                let candidate_sdr = spatial_pooler.winner_columns();
                let overlap = sdr_overlap(&predicted_sdr, &candidate_sdr);

                if overlap > max_overlap {
                    max_overlap = overlap;
                    best_candidates.clear();
                    best_candidates.push((x1, y1));
                } else if overlap == max_overlap {
                    best_candidates.push((x1, y1));
                }
            }

            if best_candidates.is_empty() {
                break;
            }

            let best_candidate = *best_candidates.choose(&mut rng).unwrap();

            if best_candidate == (x, y) {
                break;
            }

            x = best_candidate.0;
            y = best_candidate.1;
            visited.insert((x, y));
        }

        println!("imgage {}: saccades: {:?}", i, saccades);

        let filename = format!("saccades_{}.png", i);
        visualize(image, &saccades, &filename);

        let predictions = classifier.compute(i as u32, label as usize, &pool.concat(), false, true);

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
