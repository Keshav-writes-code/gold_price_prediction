use std::{collections::BTreeMap, sync::RwLock};

use crate::{
    cli::ModelArch,
    models::training::{
        data_loader::{DataLoader, RawData},
        model_architectures::PricePredictionModel,
    },
};

pub struct PricePredictionModelInfrence {
    dataset_training: RawData,
    cached_predictions: RwLock<BTreeMap<i64, f64>>,
    model: PricePredictionModel,
}

impl PricePredictionModelInfrence {
    pub fn new(arch: &ModelArch, dataset_path: &str) -> Self {
        let dataset_training = DataLoader::new(dataset_path).load();

        let cached_predictions = RwLock::new(BTreeMap::new());

        let mut model = PricePredictionModel::new(arch);
        model.load();

        Self {
            dataset_training,
            cached_predictions,
            model,
        }
    }
    pub fn predict(&self, target_time: i64) -> f64 {
        let target_time_rounded = (target_time / 86400) * 86400;

        // if prediction is cached
        if let Some(&pred) = self
            .cached_predictions
            .read()
            .unwrap()
            .get(&target_time_rounded)
        {
            return pred;
        }

        // if User ask for a date from the traning dataset
        if let Some(idx) = self
            .dataset_training
            .dates
            .iter()
            .position(|&d| d == target_time_rounded)
        {
            return self.dataset_training.prices[idx];
        }

        let max_historical_time = *self
            .dataset_training
            .dates
            .last()
            .expect("cannot find last element");
        let start_time = max_historical_time + 86400; // 86400 is for adding one day

        let mut current_time = start_time;

        while current_time <= target_time_rounded {
            if !self
                .cached_predictions
                .read()
                .unwrap()
                .contains_key(&current_time)
            {
                let features = self.build_infrence_input(current_time);
                let prediction = self.model.predict(&features);
                self.cached_predictions
                    .write()
                    .unwrap()
                    .insert(current_time, prediction);
            }
            current_time += 86400;
        }
        *self
            .cached_predictions
            .read()
            .unwrap()
            .get(&target_time_rounded)
            .unwrap_or(&0.0)
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
            for &price in self.dataset_training.prices.iter().rev() {
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
