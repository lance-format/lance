# Hierarchical Lloyd vs hierarchical e-means

8,388,608 DINO10B vectors | k=32,768 | d=1,024 | 18 threads

| | Lloyd | e-means |
|---|---|---|
| training time, median (s) | 19.3 (18.0-21.8) | 14.2 (13.4-15.3) |
| validation MSE / Lloyd | 1 | 0.9985 [0.9983, 0.9988] |
| validation MSE, mean | 196988.9 | 196698.2 |
| training MSE, mean | 187032.4 | 186637.6 |

## Hierarchical Lloyd (default)

One thread: 205.6 s; timers cover 96.6% of it as busy work.

| step (one thread) | seconds | share |
|---|---|---|
| node training: assign | 143.9 | 70.0% |
| node training: update | 19.7 | 9.6% |
| node training: other | 0.1 | 0.0% |
| membership (assign all node rows) | 35.2 | 17.1% |
| leaf means | 2.6 | 1.2% |
| sampling and gather | 4.0 | 2.0% |
| untimed | 0.2 | 0.1% |

Assign vs update, one thread: assign 89.0%, update 11.0% (node training plus membership and leaf means).

18 threads: 22.4 s wall, 403 thread-seconds, speedup 9.2x, efficiency 51.0% (an embarrassingly parallel distance computation reaches 12.9x here).

| thread-seconds | seconds | share |
|---|---|---|
| busy: assign | 231.1 | 57.3% |
| busy: gather | 30.0 | 7.4% |
| busy: other | 0.2 | 0.1% |
| busy: update | 74.0 | 18.4% |
| idle | 67.7 | 16.8% |

Idle 16.8% of thread-seconds. Busy work grew by 136.7 s (33.9% of thread-seconds) over the one-thread run: the same work runs slower when every core is loaded, including on the slower efficiency cores. 338,890 fork-joins; an empty 64-task one costs 47 µs here. At most a quarter of the threads were busy for 23.6% of the wall time.


## Hierarchical e-means

One thread: 114.7 s; timers cover 94.6% of it as busy work.

| step (one thread) | seconds | share |
|---|---|---|
| node training: assign | 72.0 | 62.7% |
| node training: update | 0.9 | 0.8% |
| node training: other | 0.2 | 0.1% |
| membership (assign all node rows) | 35.1 | 30.6% |
| leaf means | 2.5 | 2.2% |
| sampling and gather | 4.0 | 3.4% |
| untimed | 0.2 | 0.2% |

Assign vs update, one thread: assign 96.9%, update 3.1% (node training plus membership and leaf means).

18 threads: 14.8 s wall, 266 thread-seconds, speedup 7.8x, efficiency 43.2% (an embarrassingly parallel distance computation reaches 12.8x here).

| thread-seconds | seconds | share |
|---|---|---|
| busy: assign | 119.4 | 45.0% |
| busy: gather | 30.5 | 11.5% |
| busy: other | 0.3 | 0.1% |
| busy: update | 35.4 | 13.3% |
| idle | 80.0 | 30.1% |

Idle 30.1% of thread-seconds. Busy work grew by 77.2 s (29.1% of thread-seconds) over the one-thread run: the same work runs slower when every core is loaded, including on the slower efficiency cores. 464,501 fork-joins; an empty 64-task one costs 48 µs here. At most a quarter of the threads were busy for 13.1% of the wall time.

