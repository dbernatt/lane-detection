import collections

class Process(object):
  """Compose multiple process sequentially.
  Args:
      process (Sequence[callable]): Sequence of process object 
        to be composed.
  """
  def __init__(self, processes):
    assert isinstance(processes, collections.abc.Sequence)
    self.processes = []
    for process in processes:
      if callable(process):
          self.processes.append(process)
      else:
          raise TypeError('process must be callable')

  def __call__(self, data):
    """Call function to apply processes sequentially.
    Args:
        data (dict): Contains data(init_args) dict to the process.
    Returns:
        dict: Processed data.
    """

    for t in self.processes:
      data = t(data)
      if data is None:
        return None
    return data