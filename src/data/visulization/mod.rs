use gtk4::{Application, ApplicationWindow, DrawingArea, prelude::*};
use plotters::prelude::*;
use plotters_cairo::CairoBackend;

pub fn init_visulization() {
    let app = gtk4::Application::new(
        Some("io.github.keshav-writes-code.gold-price-prediction"),
        Default::default(),
    );
    app.connect_activate(build_app);
    app.run_with_args(&Vec::<String>::new());
}

fn build_app(app: &Application) {
    let window = ApplicationWindow::builder()
        .application(app)
        .title("Gold Price Prediction")
        .default_width(800)
        .default_height(600)
        .build();

    let drawing_area = DrawingArea::new();
    drawing_area.set_draw_func(|_, cr, width, height| {
        let backend = CairoBackend::new(cr, (width as u32, height as u32)).unwrap();
        let root = backend.into_drawing_area();
        root.fill(&WHITE).unwrap();

        // Read the csv file
        let file = match std::fs::File::open("data.csv") {
            Ok(f) => f,
            Err(_) => return,
        };
        let mut rdr = csv::Reader::from_reader(file);

        let mut prices: Vec<f64> = Vec::new();
        for result in rdr.records() {
            if let Ok(record) = result {
                // Assume the second column is the price
                if let Some(price_str) = record.get(1) {
                    if let Ok(price) = price_str.parse::<f64>() {
                        if !price.is_nan() {
                            prices.push(price);
                        }
                    }
                }
            }
        }

        if prices.is_empty() {
            return;
        }

        let min_price = prices.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_price = prices.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let mut chart = ChartBuilder::on(&root)
            .caption("Gold Price", ("sans-serif", 30).into_font())
            .margin(20)
            .x_label_area_size(40)
            .y_label_area_size(60)
            .build_cartesian_2d(0f64..(prices.len() as f64), min_price..max_price)
            .unwrap();

        chart.configure_mesh()
            .x_desc("Time (Index)")
            .y_desc("Price")
            .draw()
            .unwrap();

        chart.draw_series(LineSeries::new(
            prices.iter().enumerate().map(|(i, &p)| (i as f64, p)),
            &RED,
        )).unwrap();

        root.present().unwrap();
    });

    window.set_child(Some(&drawing_area));
    window.present();
}
