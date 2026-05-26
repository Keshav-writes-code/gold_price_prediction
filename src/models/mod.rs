use std::fs::File;

use csv::Reader;
use linfa::{
    Dataset, DatasetBase,
    prelude::SingleTargetRegression,
    traits::{Fit, Predict},
};
use ndarray::{Array1, Array2};
use tracing::{debug, error, info};

use crate::data::ingestion::GoldRecord;

pub fn train() {
    tracing_subscriber::fmt::init();
    let dataset = get_training_data().expect("Got a Problem while reading data");

    let model = linfa_linear::LinearRegression::default()
        .fit(&dataset)
        .expect("Cannot train");
    let pred = model.predict(&dataset);
    let r2 = pred.r2(&dataset).expect("Cannot calculate r2");
    let mse = pred
        .mean_squared_error(&dataset)
        .expect("Cannot cvalucalte MSE");

    info!(r2_score = r2, rmse = mse.sqrt(), "Model Evaluation Metrics")
}
fn get_training_data() -> Result<DatasetBase<Array2<f64>, Array1<f64>>, &'static str> {
    info!("Loading data from CSV...");
    let file = File::open("data.csv").expect("cannot open Dataset CSV");
    let mut reader = Reader::from_reader(file);

    let mut prices = Vec::new();
    for record in reader.deserialize() {
        let row: GoldRecord = record.expect("cannot parse row");
        if !row.inr.is_nan() {
            prices.push(row.inr);
        }
    }

    // Feature Engineering
    info!("Creating Features..");
    let x_cols_count: usize = 99;
    let window_size = 1 + x_cols_count; // +1 is to include the last element as target variable in a single window 

    if prices.len() < window_size {
        println!("Not enough sample to create even one window!");
        error!(
            prices = prices.len(),
            required = window_size,
            "Not enough sample to create even one window!"
        );
        return Err("Not enough sample to create even one window!");
    } else {
        debug!("Creating Traning data")
    }
    let num_window = prices.len() - window_size + 1;

    let mut flat_x = Vec::with_capacity(num_window * x_cols_count);
    let mut flat_y = Vec::with_capacity(window_size);

    for window in prices.windows(window_size) {
        flat_x.extend_from_slice(&window[..window_size - 1]);
        flat_y.push(window[window_size - 1]);
    }

    let x_train =
        Array2::from_shape_vec((num_window, x_cols_count), flat_x).expect("Shape Mismatch");
    let y_train = Array1::from_vec(flat_y);

    info!(
        samples = num_window,
        x_shape = ?x_train.shape(),
        y_shape = ?y_train.shape(),
        "Successfully Created traning dataset"
    );
    let dataset = Dataset::new(x_train, y_train);

    Ok(dataset)
}
