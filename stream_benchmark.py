import cupy as cp
from optparse import OptionParser
from pdb import set_trace

def main():
  parser = OptionParser()
  parser.add_option('-s', '--size', dest='size',
                    default='500',
                    help='matrix size')
  parser.add_option('-q', '--queue', dest='queue',
                    default='32',
                    help='number of queue')
  parser.add_option('-t', '--task', dest='task',
                    default='32',
                    help='number of task')
  (options, args) = parser.parse_args()

  xs = []
  ys = []
  zs = []
  streams = []
  mat_size = int(options.size)
  nb_stream = int(options.queue)
  nb_task = int(options.task)

  for i in range(nb_task):
    x = cp.ones((mat_size, mat_size), dtype = cp.float32)
    xs.append(x)
    y = cp.ones((mat_size, mat_size), dtype = cp.float32)
    ys.append(y)
    z = cp.empty((mat_size, mat_size), dtype = cp.float32)
    zs.append(z)
    stream = cp.cuda.stream.Stream()
    streams.append(stream)
  
  # warmup
  cp.dot(xs[0], ys[0], zs[0])
  cp.dot(xs[0], ys[0], zs[0])
  cp.dot(xs[0], ys[0], zs[0])
  cp.cuda.Device(0).synchronize()

  cp.cuda.profiler.start()

  for i in range(nb_task):
    q_id = i % nb_stream
    streams[q_id].use()
    cp.dot(xs[i], ys[i], zs[i])

  for i in range(nb_stream):
    streams[i].synchronize()

  cp.cuda.profiler.stop()

  print('done')

if __name__ == '__main__':
    main()