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
