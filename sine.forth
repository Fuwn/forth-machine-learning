\ Architecture: 1 input -> 8 hidden neurons (tanh) -> 1 output (tanh)
\ Training: online gradient descent over uniformly spaced samples in [0, 2pi]

8  CONSTANT hidden-count
20 CONSTANT sample-count
8  CONSTANT display-count

0.05e0 FCONSTANT learning-rate
6.28318530e0 FCONSTANT two-pi
two-pi sample-count S>F F/ FCONSTANT training-step

CREATE input-to-hidden-weights  hidden-count FLOATS ALLOT
CREATE hidden-biases            hidden-count FLOATS ALLOT
CREATE hidden-to-output-weights hidden-count FLOATS ALLOT
FVARIABLE output-bias

CREATE hidden-activations hidden-count FLOATS ALLOT
FVARIABLE output-activation

FVARIABLE network-input
FVARIABLE target-output

FVARIABLE output-delta
CREATE hidden-deltas hidden-count FLOATS ALLOT

: float-ref  ( addr n -- ) ( f: -- x )   FLOATS + F@ ;
: float-set  ( addr n -- ) ( f: x -- )   FLOATS + F! ;

: tanh-derivative ( f: y -- f: 1-y^2 )
  FDUP F* 1.0e0 FSWAP F- ;

\ Each neuron's bias places its tanh inflection point at a different
\ position across [0, 2pi], giving uniform basis function coverage.
: initialize-weights
  1.0e0 input-to-hidden-weights 0 float-set
  1.0e0 input-to-hidden-weights 1 float-set
  1.0e0 input-to-hidden-weights 2 float-set
  1.0e0 input-to-hidden-weights 3 float-set
  1.0e0 input-to-hidden-weights 4 float-set
  1.0e0 input-to-hidden-weights 5 float-set
  1.0e0 input-to-hidden-weights 6 float-set
  1.0e0 input-to-hidden-weights 7 float-set

  -3.1416e0 hidden-biases 0 float-set
  -2.3562e0 hidden-biases 1 float-set
  -1.5708e0 hidden-biases 2 float-set
  -0.7854e0 hidden-biases 3 float-set
   0.0000e0 hidden-biases 4 float-set
   0.7854e0 hidden-biases 5 float-set
   1.5708e0 hidden-biases 6 float-set
   2.3562e0 hidden-biases 7 float-set

   0.1e0 hidden-to-output-weights 0 float-set
  -0.1e0 hidden-to-output-weights 1 float-set
   0.1e0 hidden-to-output-weights 2 float-set
  -0.1e0 hidden-to-output-weights 3 float-set
   0.1e0 hidden-to-output-weights 4 float-set
  -0.1e0 hidden-to-output-weights 5 float-set
   0.1e0 hidden-to-output-weights 6 float-set
  -0.1e0 hidden-to-output-weights 7 float-set

  0.0e0 output-bias F! ;

: compute-hidden-activation ( n -- )
  DUP
  input-to-hidden-weights OVER float-ref network-input F@ F*
  hidden-biases ROT float-ref F+
  FTANH
  hidden-activations SWAP float-set ;

: forward-pass
  0.0e0
  hidden-count 0 DO
    I compute-hidden-activation
    hidden-to-output-weights I float-ref
    hidden-activations I float-ref F*
    F+
  LOOP
  output-bias F@ F+
  FTANH
  output-activation F! ;

: backward-pass
  target-output F@ output-activation F@ F-
  output-activation F@ tanh-derivative F*
  FDUP output-delta F!                        \ f: [d_o]

  hidden-count 0 DO
    FDUP hidden-to-output-weights I float-ref F*
    hidden-activations I float-ref tanh-derivative F*
    hidden-deltas I float-set
  LOOP
  FDROP ;

: update-weights
  learning-rate output-delta F@ F*            \ f: [lr*d_o]

  hidden-count 0 DO
    hidden-to-output-weights I float-ref
    FOVER hidden-activations I float-ref F* F+
    hidden-to-output-weights I float-set
  LOOP

  output-bias F@ FOVER F+
  output-bias F!
  FDROP

  hidden-count 0 DO
    learning-rate hidden-deltas I float-ref F*  \ f: [lr*d_h_i]

    input-to-hidden-weights I float-ref
    FOVER network-input F@ F* F+
    input-to-hidden-weights I float-set

    hidden-biases I float-ref FOVER F+
    hidden-biases I float-set
    FDROP
  LOOP ;

: train-epoch
  0.0e0                                       \ f: [x = 0.0]
  sample-count 0 DO
    FDUP network-input F!
    FDUP FSIN target-output F!
    forward-pass
    backward-pass
    update-weights
    training-step F+
  LOOP
  FDROP ;

: predict ( f: x -- f: output )
  network-input F!
  forward-pass
  output-activation F@ ;

: mean-squared-error ( f: -- f: mse )
  0.0e0                                       \ f: [acc]
  0.0e0                                       \ f: [acc, x]
  sample-count 0 DO
    FDUP FDUP FSIN target-output F!
    network-input F!
    forward-pass
    target-output F@ output-activation F@ F-
    FDUP F*
    FROT F+ FSWAP
    training-step F+
  LOOP
  FDROP
  sample-count S>F F/ ;

: train ( epochs -- )
  0 DO
    train-epoch
    I 5000 MOD 4999 = IF
      ." epoch " I 1 + 6 .R ."   mse " mean-squared-error F. cr
    THEN
  LOOP ;

: print-at ( f: x -- )
  FDUP FSIN                                   \ f: [x, sin(x)]
  FOVER predict                               \ f: [x, sin(x), pred]
  ." x=" FROT F.
  ."   actual=" FSWAP F.
  ."   predicted=" F. cr ;

: print-results
  cr
  ." sin(x) approximation" cr
  0.0e0
  display-count 0 DO
    FDUP print-at
    two-pi display-count S>F F/ F+
  LOOP
  FDROP ;

initialize-weights
50000 train
print-results
