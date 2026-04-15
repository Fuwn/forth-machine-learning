\ Architecture: 2 inputs -> 2 hidden neurons (sigmoid) -> 1 output (sigmoid)
\ Training: online gradient descent, one weight update per example

0.5e0 FCONSTANT learning-rate

\ Weights stored flat; input-to-hidden[j*2+i] = input i -> hidden neuron j
CREATE input-to-hidden-weights  4 FLOATS ALLOT
CREATE hidden-biases            2 FLOATS ALLOT
CREATE hidden-to-output-weights 2 FLOATS ALLOT
FVARIABLE output-bias

CREATE hidden-activations 2 FLOATS ALLOT
FVARIABLE output-activation

FVARIABLE first-input
FVARIABLE second-input
FVARIABLE target-output

: float-ref  ( addr n -- ) ( f: -- x )   FLOATS + F@ ;
: float-set  ( addr n -- ) ( f: x -- )   FLOATS + F! ;

: sigmoid ( f: x -- f: y )
  FNEGATE FEXP 1.0e0 F+ 1.0e0 FSWAP F/ ;

: sigmoid-derivative ( f: y -- f: dy )
  FDUP 1.0e0 FSWAP F- F* ;

: initialize-weights
  -0.5e0 input-to-hidden-weights 0 float-set
   0.3e0 input-to-hidden-weights 1 float-set
   0.7e0 input-to-hidden-weights 2 float-set
  -0.2e0 input-to-hidden-weights 3 float-set

   0.0e0 hidden-biases 0 float-set
   0.0e0 hidden-biases 1 float-set

   0.4e0 hidden-to-output-weights 0 float-set
  -0.6e0 hidden-to-output-weights 1 float-set

   0.0e0 output-bias F! ;

\ Hidden activations kept live on the float stack to avoid reloading from memory.
: forward-pass
  input-to-hidden-weights 0 float-ref first-input  F@ F*
  input-to-hidden-weights 1 float-ref second-input F@ F* F+
  hidden-biases 0 float-ref F+
  sigmoid
  FDUP hidden-activations 0 float-set             \ f: [h0]

  input-to-hidden-weights 2 float-ref first-input  F@ F*
  input-to-hidden-weights 3 float-ref second-input F@ F* F+
  hidden-biases 1 float-ref F+
  sigmoid
  FDUP hidden-activations 1 float-set             \ f: [h0, h1]

  hidden-to-output-weights 1 float-ref F*         \ f: [h0, w1*h1]
  hidden-to-output-weights 0 float-ref FROT F*    \ f: [w1*h1, w0*h0]
  F+  output-bias F@ F+
  sigmoid
  output-activation F! ;

\ Deltas carried live on the float stack; lr*delta computed once per layer
\ and reused via FOVER to avoid redundant multiplications.
: backpropagate
  output-activation F@
  FDUP sigmoid-derivative                          \ f: [output, sigmoid'(output)]
  target-output F@ FROT F- F*                     \ f: [d_o]

  FDUP hidden-to-output-weights 0 float-ref F*
  hidden-activations 0 float-ref sigmoid-derivative F*   \ f: [d_o, d_h0]

  FOVER hidden-to-output-weights 1 float-ref F*
  hidden-activations 1 float-ref sigmoid-derivative F*   \ f: [d_o, d_h0, d_h1]

  FROT learning-rate F*                            \ f: [d_h0, d_h1, lr*d_o]

  hidden-to-output-weights 0 float-ref
  FOVER hidden-activations 0 float-ref F* F+
  hidden-to-output-weights 0 float-set

  hidden-to-output-weights 1 float-ref
  FOVER hidden-activations 1 float-ref F* F+
  hidden-to-output-weights 1 float-set

  output-bias F@ FOVER F+
  output-bias F!
  FDROP                                            \ f: [d_h0, d_h1]

  learning-rate F*                                 \ f: [d_h0, lr*d_h1]

  input-to-hidden-weights 2 float-ref
  FOVER first-input F@ F* F+
  input-to-hidden-weights 2 float-set

  input-to-hidden-weights 3 float-ref
  FOVER second-input F@ F* F+
  input-to-hidden-weights 3 float-set

  hidden-biases 1 float-ref FOVER F+
  hidden-biases 1 float-set
  FDROP                                            \ f: [d_h0]

  learning-rate F*                                 \ f: [lr*d_h0]

  input-to-hidden-weights 0 float-ref
  FOVER first-input F@ F* F+
  input-to-hidden-weights 0 float-set

  input-to-hidden-weights 1 float-ref
  FOVER second-input F@ F* F+
  input-to-hidden-weights 1 float-set

  hidden-biases 0 float-ref FOVER F+
  hidden-biases 0 float-set
  FDROP ;

: train-example ( f: x0 x1 target -- )
  target-output F!
  second-input F!
  first-input F!
  forward-pass
  backpropagate ;

: train-epoch
  0.0e0 0.0e0 0.0e0 train-example
  0.0e0 1.0e0 1.0e0 train-example
  1.0e0 0.0e0 1.0e0 train-example
  1.0e0 1.0e0 0.0e0 train-example ;

: predict ( f: x0 x1 -- f: output )
  second-input F!
  first-input F!
  forward-pass
  output-activation F@ ;

: mean-squared-error ( f: -- f: mse )
  0.0e0
  0.0e0 0.0e0 predict 0.0e0 F- FDUP F* F+
  0.0e0 1.0e0 predict 1.0e0 F- FDUP F* F+
  1.0e0 0.0e0 predict 1.0e0 F- FDUP F* F+
  1.0e0 1.0e0 predict 0.0e0 F- FDUP F* F+
  4.0e0 F/ ;

: train ( epochs -- )
  0 DO
    train-epoch
    I 1000 MOD 999 = IF
      ." epoch " I 1 + 5 .R ."   mse " mean-squared-error F. cr
    THEN
  LOOP ;

: print-predictions
  cr
  ." learned xor" cr
  ." 0 ^ 0 = " 0.0e0 0.0e0 predict F. cr
  ." 0 ^ 1 = " 0.0e0 1.0e0 predict F. cr
  ." 1 ^ 0 = " 1.0e0 0.0e0 predict F. cr
  ." 1 ^ 1 = " 1.0e0 1.0e0 predict F. cr ;

initialize-weights
10000 train
print-predictions
