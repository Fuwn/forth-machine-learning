\ Matrix multiply benchmark: N×N matmul, CPU device vs Metal GPU.
\ Measures us/op and GFLOPS for increasing N to show GPU crossover point.

s" /Users/ebisu/Developer/Git/Fuwn/PoemForth/crates/poem-gpu/target/release/libpoem_gpu.dylib" LOAD-LIBRARY

VARIABLE ta   VARIABLE tb   VARIABLE tc
VARIABLE bench-start
VARIABLE bench-n

: matmul-bench ( n iters -- us )
  swap
  dup dup 2 tensor-randn  ta !
  dup dup 2 tensor-randn  tb !  drop

  \ warmup + sync to trigger Metal JIT and stabilise timings
  ta @ tb @ TENSOR-MATMUL  tc !
  tc @ TENSOR-MEAN-ALL FDROP  tc @ TENSOR-FREE

  UTIME DROP  bench-start !

  DUP 0 DO
    ta @ tb @ TENSOR-MATMUL  tc !  tc @ TENSOR-FREE
  LOOP

  \ one final op + readback to flush the GPU pipeline before stopping the clock
  ta @ tb @ TENSOR-MATMUL  tc !
  tc @ TENSOR-MEAN-ALL FDROP  tc @ TENSOR-FREE

  UTIME DROP  bench-start @ -
  SWAP /

  ta @ TENSOR-FREE  tb @ TENSOR-FREE ;

\ Prints 2*N^3 / (us*1000) as integer GFLOPS.
: .gflops ( n us -- )
  SWAP DUP DUP * * 2 *
  SWAP 1000 * /
  6 .R ;

: bench-row ( n iters -- )
  OVER bench-n !
  matmul-bench
  bench-n @ 5 .R ."  | "
  DUP 5 .R ."  | "
  bench-n @ SWAP .gflops cr ;

: run-bench
  ."     n | us/op | gflops" cr
  ." ------+-------+-------" cr
  32    200 bench-row
  64    100 bench-row
  128    50 bench-row
  256    20 bench-row
  512    10 bench-row
  1024    5 bench-row ;

." ======== cpu =========" cr
GPU-DEVICE-CPU
run-bench

cr ." ======= metal ========" cr
GPU-DEVICE-METAL
run-bench

bye

