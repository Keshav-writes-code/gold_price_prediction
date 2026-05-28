use crate::config::create_artifact;

use self::internet_zip_ingestion::ZipIngestion;
use ndarray::Array1;
use serde::{Deserialize, Deserializer, Serialize};
mod internet_zip_ingestion;

fn deserialize_f64_or_na<'de, D>(deserializer: D) -> Result<f64, D::Error>
where
    D: Deserializer<'de>,
{
    let s: Option<String> = Option::deserialize(deserializer)?;
    match s {
        Some(val) => {
            if val == "#N/A" || val.trim().is_empty() {
                Ok(f64::NAN)
            } else {
                val.replace(",", "")
                    .parse::<f64>()
                    .map_err(serde::de::Error::custom)
            }
        }
        None => Ok(f64::NAN),
    }
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct GoldRecord {
    #[serde(rename = "Date")]
    pub date: String,
    #[serde(rename = "INR", deserialize_with = "deserialize_f64_or_na")]
    pub inr: f64,
}

pub trait IngestionStratergy {
    fn fetch_data(&self) -> Vec<GoldRecord>;
}

pub const RAW_DATA_PATH: &str = "raw_data.csv";

pub fn hanlde_ingestion() -> Array1<f64> {
    let stratergies: Vec<Box<dyn IngestionStratergy>> = vec![Box::new(ZipIngestion)];

    let mut all_records = Vec::new();
    for stratergy in stratergies {
        all_records.extend(stratergy.fetch_data());
    }

    let file = create_artifact(RAW_DATA_PATH);
    let mut writer = csv::Writer::from_writer(file);

    all_records.iter().for_each(|record| {
        writer.serialize(record).expect("Failed");
    });
    writer.flush().expect("Cannot write to file");
    let prices = all_records.into_iter().map(|r| r.inr);
    Array1::from_iter(prices)
}
