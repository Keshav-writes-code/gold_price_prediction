use self::internet_zip_ingestion::ZipIngestion;
use ndarray::Array1;
use serde::{Deserialize, Serialize};
mod internet_zip_ingestion;

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct GoldRecord {
    #[serde(rename = "Date")]
    pub date: String,
    #[serde(rename = "INR")]
    pub inr: f64,
}

pub trait IngestionStratergy {
    fn fetch_data(&self) -> Vec<GoldRecord>;
}

pub const FEATURE_STORE_PATH: &str = "data.csv";

pub fn hanlde_ingestion() -> Array1<f64> {
    let stratergies: Vec<Box<dyn IngestionStratergy>> = vec![Box::new(ZipIngestion)];

    let mut all_records = Vec::new();
    for stratergy in stratergies {
        all_records.extend(stratergy.fetch_data());
    }
    let mut writer = csv::Writer::from_path(FEATURE_STORE_PATH).expect("Error cann't init writer");

    all_records.iter().for_each(|record| {
        writer.serialize(record).expect("Failed");
    });
    writer.flush().expect("Cannot write to file");
    let prices = all_records.into_iter().map(|r| r.inr);
    Array1::from_iter(prices)
}
