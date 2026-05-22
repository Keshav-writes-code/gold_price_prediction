use gtk4::{Application, ApplicationWindow, DrawingArea, prelude::*};
use plotters::prelude::*;
use plotters_cairo::CairoBackend;
use polars::prelude::*;

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

        // Read the parquet file
        let df = match LazyFrame::scan_parquet("output.parquet".into(), Default::default()) {
            Ok(lf) => lf.collect().unwrap_or_else(|_| DataFrame::empty()),
            Err(_) => DataFrame::empty(),
        };

        let col_names = df.get_column_names();
        if col_names.len() < 2 {
            return;
        }

        // Assume the second column is the price
        let price_col_name = col_names[1];
        let price_series = df.column(price_col_name).unwrap().cast(&DataType::String).unwrap();
        
        let prices: Vec<f64> = match price_series.str() {
            Ok(ca) => ca.into_iter()
                .flatten()
                .filter_map(|s| s.replace(',', "").parse::<f64>().ok())
                .collect(),
            Err(_) => return,
        };

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
