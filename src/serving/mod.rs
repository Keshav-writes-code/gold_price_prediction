use actix_web::{App, HttpResponse, HttpServer, Responder, get, web};
use serde::{Deserialize, Serialize};

use crate::models::infrence::PricePredictionModelInfrence;

#[derive(Deserialize)]
struct PredictionReq {
    pub unix_time: usize,
}

#[derive(Serialize)]
struct PredictionResponse {
    pub predicted_price: f64,
}

#[actix_web::main]
pub async fn serve() {
    let model = PricePredictionModelInfrence::default();
    let model_arc = web::Data::new(model);
    HttpServer::new(move || App::new().app_data(model_arc.clone()).service(predict))
        .bind(("0.0.0.0", 8080))
        .expect("cannot connect to Socket")
        .run()
        .await
        .expect("cannot run server");
}

#[get("/predict")]
async fn predict(
    model: web::Data<PricePredictionModelInfrence>,
    req: web::Json<PredictionReq>,
) -> impl Responder {
    let req_data = req.into_inner();

    let pred = model.predict(req_data.unix_time);
    HttpResponse::Ok().json(PredictionResponse {
        predicted_price: pred,
    })
}
