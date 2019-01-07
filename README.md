# CUDA stream benchmark #

## requirement ##
* CUDA
* Python
* CuPy

## run ##
```sh
Usage: stream_benchmark.py [options]

Options:
  -h, --help            show this help message and exit
  -s SIZE, --size=SIZE  matrix size
  -q QUEUE, --queue=QUEUE
                        number of queue
  -t TASK, --task=TASK  number of task
```

for example:
```sh
$ nvvp python stream_benchmark.py -s 500 -q 16 -t 1000
```
