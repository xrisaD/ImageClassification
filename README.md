# Machine Learning Assignment 1
@ AUEB 2021 <br>
***
Commands:
```console
python main.py DATASET_NAME
```
- DATASET_NAME: mnist or cifar <br>
***
## The datasets
mnist: dataset of handwritten single digits <br>
![](https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png) <br>
[cifar](https://www.cs.toronto.edu/~kriz/cifar.html): dataset which consists of the following 10 classes
![](https://matlab1.com/wp-content/uploads/2018/06/image1.jpg)
***
### Architecture:  
![](back.png)
***
### Results: 
| Dataset      | best hyperparameters | accuracy |
| ----------- | ----------- | ----------- |
| mnist      | lr = 1e-04, λ = 1e-05, Μ = 100, H3Activation, glorot | 0.9607 |
| cifar   | lr = 1e-04, λ = 1e-05, Μ = 300, H3Activation, glorot | 0.5292 |
