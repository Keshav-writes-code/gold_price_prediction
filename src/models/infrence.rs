use std::{collections::HashMap, fs::File};

use linfa::{Dataset, DatasetBase};
use linfa_linear::FittedLinearRegression;
use ndarray::{Array1, Array2};

use crate::models::training::SerealizedDataset;

pub struct PricePredictionModelInfrence {
    dataset_traning: DatasetBase<Array2<f64>, Array1<f64>>,
    cached_predictions: HashMap<usize, f64>,
    model: FittedLinearRegression<f64>,
}
impl Default for PricePredictionModelInfrence {
    fn default() -> Self {
        let file = File::open("gold_price_prediction.json").expect("Cannot Open File");
        let model: FittedLinearRegression<f64> =
            serde_json::from_reader(file).expect("cannot load model");

        let new_data = File::open("new_data.csv").expect("cannot open new data file");
        let serealized_data: SerealizedDataset =
            serde_json::from_reader(new_data).expect("Cammpt Deserealize file");
        let dataset_training = Dataset::new(serealized_data.records, serealized_data.targets);

        Self {
            dataset_traning: dataset_training,
            cached_predictions: HashMap::new(),
            model,
        }
    }
}

impl PricePredictionModelInfrence {
    pub fn predict(&self, unix_time: usize) -> f64 {
        todo!();
    }
    pub fn build_infrence_input(unix_time: usize) -> Self {
        todo!();
    }
}
