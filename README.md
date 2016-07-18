<p align="center"><sup><strong>
DFT-CROSSFILTER
</strong></sup></p>
The Dft-Crossfiltering is a web platform that provides crossfiltering capabilities to users.
The aim of this platform is to support a dft benchmark data exposed to collaborators.
* **[INSTALL](INSTALLATION.md)** – installation instructions.
* **[LICENSE](LICENSE)** – the license.

# The platform components
## benchmark-db
Mongodb database for managing the dft data.
The models are design for ease of access from the API but also from the frontend.
It might be subjected to regular changes.

## benchmark-api
A python flask REST service that exposes the database access to the frontend.
The api endpoints have been organized to allow an ease for sending HTTP requests from
the frontend.

## benchmark
A python library for easily loading csv files by name without extension.
Not sure if this will not be deleted.

## bokeh
A bokeh app based on the crossfiltering use case.
The server loades the pandas dataframe of the dft data header.
Throughtout the execution of the app all the access to the data is replaced by calls to
the api.
