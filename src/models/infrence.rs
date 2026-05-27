use std::{collections::BTreeMap, fs::File, sync::RwLock};

use linfa::traits::Predict;
use linfa_linear::FittedLinearRegression;
use ndarray::Array2;

use crate::models::training::{DataLoader, RawData};

pub struct PricePredictionModelInfrence {
    dataset_traning: RawData,
    cached_predictions: RwLock<BTreeMap<i64, f64>>,
    model: FittedLinearRegression<f64>,
}
impl Default for PricePredictionModelInfrence {
    fn default() -> Self {
        let file = File::open("gold_price_prediction.json").expect("Cannot Open File");
        let model: FittedLinearRegression<f64> =
            serde_json::from_reader(file).expect("cannot load model");

        let dataset_training = DataLoader::new("data.csv").load();

        Self {
            dataset_traning: dataset_training,
            cached_predictions: RwLock::new(BTreeMap::new()),
            model,
        }
    }
}

impl PricePredictionModelInfrence {
    pub fn predict(&self, target_time: i64) -> f64 {
        let target_time = (target_time / 86400) * 86400;

        // if prediction is cached
        if let Some(&pred) = self.cached_predictions.read().unwrap().get(&target_time) {
            return pred;
        }

        // if User ask for a date from the traning dataset
        if let Some(idx) = self
            .dataset_traning
            .dates
            .iter()
            .position(|&d| d == target_time)
        {
            return self.dataset_traning.prices[idx];
        }

        let max_historical_time = *self
            .dataset_traning
            .dates
            .last()
            .expect("cannot find last element");
        let start_time = max_historical_time + 86400; // 86400 is for adding one day

        let mut current_time = start_time;

        while current_time <= target_time {
            if !self
                .cached_predictions
                .read()
                .unwrap()
                .contains_key(&current_time)
            {
                let features = self.build_infrence_input(current_time);
                let x_inputs = Array2::from_shape_vec((1, 99), features).unwrap();

                let prediction = self.model.predict(&x_inputs);
                self.cached_predictions
                    .write()
                    .unwrap()
                    .insert(current_time, prediction[0]);
            }
            current_time += 86400;
        }
        *self
            .cached_predictions
            .read()
            .unwrap()
            .get(&target_time)
            .unwrap()
    }
    pub fn build_infrence_input(&self, target_time: i64) -> Vec<f64> {
        let mut features = Vec::with_capacity(99);

        for (_ts, &price) in self
            .cached_predictions
            .read()
            .unwrap()
            .range(..target_time)
            .rev()
        {
            features.push(price);
            if features.len() == 99 {
                break;
            }
        }

        if features.len() < 99 {
            for &price in self.dataset_traning.prices.iter().rev() {
                features.push(price);
                if features.len() == 99 {
                    break;
                }
            }
        }
        features.reverse();

        features
    }
}
