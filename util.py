import sys
import os

class Logger(object):
  def __init__(self, args):
    # NOTE: format: dataset/seed/record.log
    self.terminal = sys.stdout
    output_file = os.path.join(args.log_dir, "record-"+str(args.stage)+".log")
    self.log = open(output_file, "w")

  def write(self, message):
    print(message, end="", file=self.terminal, flush=True)
    print(message, end="", file=self.log, flush=True)

  def flush(self):
    self.terminal.flush()
    self.log.flush()

