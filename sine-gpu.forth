\ Architecture: 1 input -> 8 hidden neurons (tanh) -> 1 output (tanh)
\ Batched training: all samples processed per epoch in one forward/backward pass,
\ eliminating per-sample kernel dispatches and making GPU acceleration beneficial.

s" /Users/ebisu/Developer/Git/Fuwn/PoemForth/crates/poem-gpu/target/release/libpoem_gpu.dylib" LOAD-LIBRARY

GPU-DEVICE-METAL

8  CONSTANT hidden-count
20 CONSTANT sample-count
8  CONSTANT display-count

0.05e0 FCONSTANT learning-rate
6.28318530e0 FCONSTANT two-pi
two-pi sample-count S>F F/ FCONSTANT training-step

\ Weight tensors: W1 [8,1], b1 [8], W2 [1,8], b2 [1]
VARIABLE w1   VARIABLE b1   VARIABLE w2   VARIABLE b2

\ Training data, built once before training
VARIABLE train-x   \ [20,1]
VARIABLE train-y   \ [20,1]

\ Per-epoch forward activations (set by forward-pass, freed by step-backward)
VARIABLE fh   \ [20,8]
VARIABLE fo   \ [20,1]

\ Per-epoch hidden gradient
VARIABLE sdh  \ [20,8]

\ Scratch handles shared across non-overlapping words
VARIABLE ta   VARIABLE tb   VARIABLE tc

\ Scratch handles used exclusively inside tanh-deriv
VARIABLE td0  VARIABLE td1

: initialize-weights
  1.0e0  hidden-count 1 2  TENSOR-FULL                      w1 !
  6.28318530e0  hidden-count 1  TENSOR-RAND  TENSOR-SCALE   ta !
  -3.14159265e0  hidden-count 1  TENSOR-FULL                tb !
  ta @ tb @ TENSOR-ADD                                       b1 !
  ta @ TENSOR-FREE  tb @ TENSOR-FREE
  0.1e0  1 hidden-count 2  TENSOR-RANDN  TENSOR-SCALE       w2 !
  0.0e0  1 1  TENSOR-FULL                                    b2 ! ;

: build-training-data
  0.0e0 two-pi training-step  TENSOR-ARANGE-STEP   ta !
  ta @  sample-count 1 2  TENSOR-RESHAPE           train-x !
  ta @ TENSOR-FREE
  train-x @  TENSOR-SIN                            train-y ! ;

\ Does not consume the input handle; caller is responsible for freeing it.
: tanh-deriv ( h -- 1-h^2 )
  DUP DUP TENSOR-MUL  td0 !
  DROP
  td0 @  1.0e0  TENSOR-FULL-LIKE  td1 !
  td1 @ td0 @ TENSOR-SUB
  td0 @ TENSOR-FREE
  td1 @ TENSOR-FREE ;

\ Sets fh [20,8] and fo [20,1].
: forward-pass
  train-x @  w1 @  b1 @  TENSOR-LINEAR   ta !
  ta @  TENSOR-TANH                       fh !
  ta @  TENSOR-FREE
  fh @  w2 @  b2 @  TENSOR-LINEAR        ta !
  ta @  TENSOR-TANH                       fo !
  ta @  TENSOR-FREE ;

\ Uses original W2 for hidden gradient (computed before W2 update).
\ Frees fh, fo, and all intermediates.
: step-backward
  \ delta_O = (fo - train-y) * tanh'(fo)  [20,1]
  fo @  train-y @  TENSOR-SUB   ta !
  fo @  tanh-deriv               tb !
  ta @  tb @  TENSOR-MUL        tc !
  ta @ TENSOR-FREE  tb @ TENSOR-FREE

  \ delta_H = (delta_O @ W2) * tanh'(fh)  [20,8]
  tc @  w2 @  TENSOR-MATMUL    ta !
  fh @  tanh-deriv              tb !
  ta @  tb @  TENSOR-MUL       sdh !
  ta @ TENSOR-FREE  tb @ TENSOR-FREE

  \ Update W2 -= lr * (delta_O.T @ fh)
  tc @  0 1  TENSOR-TRANSPOSE   ta !
  ta @  fh @  TENSOR-MATMUL     tb !
  ta @ TENSOR-FREE
  tb @  learning-rate  TENSOR-SCALE   ta !
  tb @ TENSOR-FREE
  w2 @  ta @  TENSOR-SUB  tb !
  ta @ TENSOR-FREE  w2 @ TENSOR-FREE
  tb @  w2 !

  \ Update b2 -= lr * sum(delta_O, dim=0)
  tc @  0  TENSOR-SUM          ta !
  tc @ TENSOR-FREE
  ta @  learning-rate  TENSOR-SCALE  tb !
  ta @ TENSOR-FREE
  b2 @  tb @  TENSOR-SUB  ta !
  tb @ TENSOR-FREE  b2 @ TENSOR-FREE
  ta @  b2 !

  \ Update W1 -= lr * (delta_H.T @ train-x)
  sdh @  0 1  TENSOR-TRANSPOSE   ta !
  ta @  train-x @  TENSOR-MATMUL  tb !
  ta @ TENSOR-FREE
  tb @  learning-rate  TENSOR-SCALE  ta !
  tb @ TENSOR-FREE
  w1 @  ta @  TENSOR-SUB   tb !
  ta @ TENSOR-FREE  w1 @ TENSOR-FREE
  tb @  w1 !

  \ Update b1 -= lr * sum(delta_H, dim=0)
  sdh @  0  TENSOR-SUM          ta !
  ta @  learning-rate  TENSOR-SCALE  tb !
  ta @ TENSOR-FREE
  b1 @  tb @  TENSOR-SUB   ta !
  tb @ TENSOR-FREE  b1 @ TENSOR-FREE
  ta @  b1 !
  sdh @ TENSOR-FREE

  fo @ TENSOR-FREE
  fh @ TENSOR-FREE ;

: predict ( f: x -- f: output )
  1 1 2  TENSOR-FULL            ta !
  ta @  w1 @  b1 @  TENSOR-LINEAR   tb !
  ta @ TENSOR-FREE
  tb @  TENSOR-TANH              ta !
  tb @ TENSOR-FREE
  ta @  w2 @  b2 @  TENSOR-LINEAR   tb !
  ta @ TENSOR-FREE
  tb @  TENSOR-TANH              ta !
  tb @ TENSOR-FREE
  ta @  TENSOR-SCALAR
  ta @  TENSOR-FREE ;

: train-epoch
  forward-pass
  step-backward ;

: mean-squared-error ( f: -- f: mse )
  0.0e0 0.0e0
  sample-count 0 DO
    FDUP predict
    FOVER FSIN
    F- FDUP F*
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
  FDUP FSIN
  FOVER predict
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
build-training-data
50000 train
print-results
bye
