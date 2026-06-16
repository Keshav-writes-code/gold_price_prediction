<h1 align="center" id="title">Gold Price Prediction</h1>

<p align="center"><img width="100" height="100" alt="Au" src="https://github.com/user-attachments/assets/176dcac0-c1ba-44db-875f-a3b272239229" /></p>

<p align="center" id="description">an End-to-End MLOps Project for Indian Gold price prediction</p>

<h2>🚀 Demo</h2>


<h2>Project Screenshots:</h2>
<img width="1356" height="862" alt="image" src="https://github.com/user-attachments/assets/4d7a1665-18b5-4f66-9e9c-b65857ea6b9f" />
  
<h2>🧐 Features</h2>

Here're some of the project's best features:

*   You can run this locally! see the [run locally](#run-locally) instruction
*   Support Mlp and Linear Regression Model Architectures
*   Training is configured by a training config file
*   Includes a simple web frontend to interract with the model

<h2>🛠️ Run locally:</h2>

* go to the [Releases page](https://github.com/Keshav-writes-code/gold_price_prediction/releases)
* Download the binary packages according to your OS and Cpu Architecture
* run the binary with these commands:

#### 1. Make the binary executable (Unix only) 

```sh
cd ~/Downloads
chmod +x gold-price-prediction
```

#### 2. Pull the data from built in data sources 

```sh
./gold-price-prediction --pull
```

#### 3. Train the model 

```sh
./gold-price-prediction --train 
```

- training can be configured by storing them in a `training_config.toml` file next to the `gold-price-prediction` binary
- full example config:
```toml
arch = "LinearRegression" # or "Mlp"
learning_rate = 0.016
```

#### 4. Serve the Model

```sh
./gold-price-prediction --serve 
```

it spins up a webserver to server an api route and a web frotnend to interract with the model.
just open `http://0.0.0.0:8080/` in your browser

<h2>💻 Built with</h2>

Technologies used in the project:

*   Rust
*   linfa
*   rust\_mlp
*   argh
*   actix
