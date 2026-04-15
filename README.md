# Forth Machine Learning

> Machine Learning Examples in Forth

## Examples

### Exclusive Or Neural Network (`xor.forth`)

2-2-1 network (2 inputs, 2 hidden neurons, 1 output) trained with online backpropagation and sigmoid activation.

```
$ poemk < xor.forth

epoch  1000  mse 0.232805
epoch  2000  mse 0.004768
...
epoch 10000  mse 0.000306

learned xor
0 ^ 0 = 0.019298
0 ^ 1 = 0.983447
1 ^ 0 = 0.983306
1 ^ 1 = 0.017283
```

### Sine Wave Regression (`sine.forth`)

1-8-1 network trained to approximate sin(x) over [0, 2π]. Hidden neurons use
tanh activation with biases evenly spaced to cover the input range (principled
RBF-like basis). 20 training samples, online gradient descent.

```
$ poemk < sine.forth
epoch   5000  mse 0.002533
epoch  10000  mse 0.000883
...
epoch  50000  mse 0.000331

sin(x) approximation
x=0.00000   actual=0.00000   predicted=-0.001068
x=0.785398   actual=0.707107   predicted=0.711194
x=1.57080   actual=1.00000   predicted=0.960106
x=2.35619   actual=0.707107   predicted=0.735541
x=3.14159   actual=0.000000   predicted=-0.010795
x=3.92699   actual=-0.707107   predicted=-0.720338
x=4.71239   actual=-1.00000   predicted=-0.960779
x=5.49779   actual=-0.707107   predicted=-0.727600
```
