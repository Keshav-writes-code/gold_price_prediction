use std::{
    fs::File,
    io::{Cursor, Read},
};

use csv::Writer;

use super::GoldRecord;
use super::IngestionStratergy;

const URL: &str = "https://www.kaggle.com/api/v1/datasets/download/rizkykiky/gold-price-dataset";
pub struct ZipIngestion;

impl IngestionStratergy for ZipIngestion {
    fn fetch_data(&self) -> Vec<GoldRecord> {
        let mut response = ureq::get(URL).call().expect("Failed to fetch URL");
        let mut data = Vec::new();
        response
            .body_mut()
            .as_reader()
            .read_to_end(&mut data)
            .expect("cannot read data");

        let mut archive = zip::ZipArchive::new(Cursor::new(data)).unwrap();
        let file = archive.by_name("Daily.csv").expect("cannot find file");

        let mut reader = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_reader(file);
        let records: Vec<GoldRecord> = reader
            .deserialize::<GoldRecord>()
            .map(|r| r.unwrap())
            .collect();

        let file = File::create("data.csv").expect("Cannot create csv");
        let mut writer = Writer::from_writer(file);

        for record in &records {
            writer.serialize(record).unwrap();
        }
        writer.flush().unwrap();
        records
    }
}
