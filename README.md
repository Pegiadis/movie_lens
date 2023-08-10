# MovieLens Deep Learning Models

This repository focuses on developing multiple deep learning models to make recommendation systems on the MovieLens dataset.

## Structure

The project is organized as follows:
```
├── notebooks
│   ├── AE.ipynb
│   ├── DNN.ipynb
│   ├── eda.ipynb
│   ├── Hybrid_NN.ipynb
│   ├── LSTM.ipynb
│   ├── Matrix_factorization.ipynb
│   ├── MFNN.ipynb
│   ├── NCF.ipynb
│   ├── RNN.ipynb
│   └── transformations.ipynb
```


## Notebooks Description

- `AE.ipynb`: AutoEncoders Model
- `DNN.ipynb`: Deep Neural Network Model
- `eda.ipynb`: Exploratory Data Analysis
- `Hybrid_NN.ipynb`: Hybrid Neural Network Model
- `Matrix_factorization.ipynb`: Matrix Factorization Techniques
- `MFNN.ipynb`: Matrix Factorization Neural Network Model
- `NCF.ipynb`: Neural Collaborative Filtering
- `RNN.ipynb`: Recurrent Neural Network Model
- `transformations.ipynb`: Data Transformations and Preprocessing

# Hyperlink
From the following link, you can download the dataset used in this project:
[MovieLens](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)

## Data Description

The data folder contains both the raw and cleaned datasets used in this project:
- `data`
- `lens`: Folder containing MovieLens datasets
- `cleaned`: Contains cleaned versions of datasets
- `links_cleaned.csv`: Cleaned links data
- `movies_cleaned.csv`: Cleaned movies data
- `ratings_cleaned.csv`: Cleaned ratings data
- `tags_cleaned.csv`: Cleaned tags data
- Other files in `lens` include raw datasets such as links, movies, ratings, and tags.



# Results

The following table shows the results of the different models used in this project:

| Model                               | RMSE   | 
|-------------------------------------|--------|
| AutoEncoders                        | 0.4963 |
| Deep Neural Network                 | 0.9668 |
| Hybrid Neural Network               | 0.7956 |
| Matrix Factorization                | 0.8999 |
| Matrix Factorization Neural Network | 6.064  |
| Neural Collaborative Filtering      | 0.9231 |
| Recurrent Neural Network            | 0.7924 |

# visualizations

The following figures show the results of the different models used in this project:

## AutoEncoder
![](./metrics/rmse_ae.png)
![](./metrics/train_test_ae.png)

## Deep Neural Network
![](./metrics/rmse_dnn.png)
![](./metrics/train_test_dnn.png)

## Hybrid Neural Network
![](./metrics/rmse_hybrid.png)
![](./metrics/train_test_hybrid.png)


## Matrix Factorization Neural Network
![](./metrics/rmse_mfnn.png)
![](./metrics/train_test_mfnn.png)

## Neural Collaborative Filtering
![](./metrics/rmse_ncf.png)
![](./metrics/train_test_ncf.png)

## Recurrent Neural Network
![](./metrics/rmse_rnn.png)
![](./metrics/train_test_rnn.png)

## Matrix Factorization
![](./metrics/mf.png)


# Streamlit App
![](./metrics/streamlit_app.png)