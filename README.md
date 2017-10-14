# SimplePythonCNN
This is Convolutional Neural Network only in python & numpy. It is simple and slow but will get the job done :+1:

## Specification
**Weight Initialization :** HE Normal

**Weight Update Policy :** ADAM, NAG, Momentum, Vanila

**Active Function :** ReLU, Sigmoid

**Regulization :** Droupout(only on fc), L2

**Pooling :** Max, Average

**Loss Function :** Softmax, Logistic

## Prerequisites
numpy (+ mkl for intel processors. recommend [anaconda](https://www.continuum.io/downloads))  
Used sklearn for LabelEncoder & utils.shuffle on examples.


## Example
AND gate and CIFAR-10 examples are included.

```python

lr = 1e-4
l2_reg = 8e-6

cnn = NeuralNetwork(train_images.shape[1:],
                    [
                        {'type': 'conv', 'k': 16, 'u_type': 'nag', 'f': 5, 's': 1, 'p': 2},
                        {'type': 'pool', 'method': 'average'},
                        {'type': 'conv', 'k': 20, 'u_type': 'nag', 'f': 5, 's': 1, 'p': 2},
                        {'type': 'pool', 'method': 'average'},
                        {'type': 'conv', 'k': 20, 'u_type': 'nag', 'f': 5, 's': 1, 'p': 2},
                        {'type': 'pool', 'method': 'average'},
                        {'type': 'output', 'k': len(le.classes_), 'u_type': 'adam'}
                    ]
                    , lr, l2_reg=l2_reg)

```

CIFAR-10 example gets ~72% test accuracy in 20 epoch.


## API Reference
```python
classes.NeuralNetwork(self, input_shape, layer_list, lr, l2_reg=0, loss='softmax'):
```
<br />

| Parameter | Description |
| --- | --- |
| input_shape | Data's numpy shape.  |
| layer_list | List of layers you want to be networked. All of properties goes to **kwargs. |
| lr | Learning rate. |
| l2_reg | L2 regularization|
| loss | Loss function. 'softmax', 'logistic' |


```python
# type fc, output
classes.NeuralLayer(input_size, k, f=3, s=1, p=1, u_type='adam', a_type='relu', dropout=1)

# type pool
classes.PoolLayer(input_size, f=2, s=2, method='max', dropout=1):

# type conv
classes.ConvLayer(input_size, k, f=3, s=1, p=1, u_type='adam', a_type='relu', dropout=1)
```
<br />



| Update Policy | u_type|
| --- | --- |
| ADAM | 'adam' |
| Momentum | 'm' |
| Vanilla | 'v' |
| NAG | 'nag' |
| RMSProp | 'rmsprop' |

| Activation Function |a_type|
| --- | --- |
| ReLU | 'relu' |
| Sigmoid | 'sigmoid' |

| Pooling |method|
| --- | --- |
| Max |'max' |
|Avverage |'average'|

## License
MIT

