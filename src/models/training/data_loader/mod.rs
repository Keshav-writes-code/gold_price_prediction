use crate::{config::open_artifact, data::ingestion::GoldRecord};
use chrono::NaiveDate;
use csv::Reader;
use ndarray::{Array1, Array2};
use tracing::{debug, error, info};

pub struct DataLoader {
    db_artifact_file_path: String,
}

impl DataLoader {
    pub fn new(file_path: &str) -> Self {
        Self {
            db_artifact_file_path: file_path.to_string(),
        }
    }
    pub fn load(self) -> RawData {
        info!("Loading data from CSV...");
        let file = open_artifact(&self.db_artifact_file_path);
        let mut reader = Reader::from_reader(file);

        let mut prices = Vec::new();
        let mut dates = Vec::new();
        for record in reader.deserialize() {
            let row: GoldRecord = record.expect("cannot parse row");
            if !row.inr.is_nan() {
                prices.push(row.inr);
                let unix_date = NaiveDate::parse_from_str(&row.date, "%m/%d/%Y")
                    .expect("Cannot covert string to unix timel")
                    .and_hms_opt(0, 0, 0)
                    .expect("problem setting hms")
                    .and_utc()
                    .timestamp();

                dates.push(unix_date);
            }
        }
        RawData { prices, dates }
    }
}
pub struct RawData {
    pub prices: Vec<f64>,
    pub dates: Vec<i64>,
}
impl RawData {
    pub fn build_features(self, window_size: usize) -> Result<TrainingData, &'static str> {
        info!("Creating Features..");

        if self.prices.len() < window_size {
            error!(
                prices = self.prices.len(),
                required = window_size,
                "Not enough sample to create even one window!"
            );
            return Err("Not enough sample to create even one window!");
        } else {
            debug!("Creating Traning data")
        }

        let x_cols_count: usize = window_size - 1;
        let num_window = self.prices.len() - window_size + 1;

        let mut flat_x = Vec::with_capacity(num_window * x_cols_count);
        let mut flat_y = Vec::with_capacity(window_size);

        for window in self.prices.windows(window_size) {
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
        Ok(TrainingData {
            x_train,
            y_train,
            x_dim: x_cols_count,
        })
    }
}

pub struct TrainingData {
    pub x_train: Array2<f64>,
    pub y_train: Array1<f64>,
    pub x_dim: usize,
}
