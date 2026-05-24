use std::fs::File;

use csv::Reader;

use crate::data::ingestion::GoldRecord;

pub fn train() {
    let file = File::open("data.csv").expect("cannot open Dataset CSV");
    let mut reader = Reader::from_reader(file);

    let mut prices = Vec::new();
    for record in reader.deserialize() {
        let row: GoldRecord = record.expect("cannot parse row");

        prices.push(row.inr);
    }

    println!("{:?}", prices);
}
