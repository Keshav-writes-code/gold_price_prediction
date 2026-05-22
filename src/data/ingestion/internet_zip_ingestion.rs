use std::{
    fs::{self, File, create_dir_all},
    io::{Cursor, copy},
    path::Path,
};

use polars::lazy::{
    dsl::col,
    frame::{LazyCsvReader, LazyFileListReader, LazyFrame},
};

use super::IngestionStratergy;

const URL: &str = "https://www.kaggle.com/api/v1/datasets/download/rizkykiky/gold-price-dataset";
const DEST: &str = "/tmp/extracted_files";

pub struct ZipIngestion;

impl IngestionStratergy for ZipIngestion {
    fn fetch_data(&self) -> LazyFrame {
        Self::download_and_extract_zip();
        Self::read_csv_file()
    }
}
impl ZipIngestion {
    fn download_and_extract_zip() {
        fs::create_dir_all(DEST).unwrap();

        let data = reqwest::blocking::get(URL).unwrap().bytes().unwrap();
        let mut archive = zip::ZipArchive::new(Cursor::new(data)).unwrap();
        for i in 0..archive.len() {
            let mut file = archive.by_index(i).unwrap();
            let outpath = format!("{}/{}", DEST, file.name());
            if file.name().ends_with("/") {
                create_dir_all(&outpath).unwrap();
            } else {
                if let Some(p) = Path::new(&outpath).parent() {
                    create_dir_all(p).unwrap();
                }
                let mut new_file = File::create(&outpath).unwrap();
                copy(&mut file, &mut new_file).unwrap();
            }
        }
    }
    fn read_csv_file() -> LazyFrame {
        let path = Path::new(DEST).join("Daily.csv");
        let path_str = path.to_str().expect("Cannot");
        let lf = LazyCsvReader::new(path_str.into())
            .with_has_header(true)
            .finish()
            .unwrap();
        lf.select([col("Date"), col("INR")])
    }
}
