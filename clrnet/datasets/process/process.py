import collections

class Process(object):
  """Compose multiple process sequentially.
  Args:
      process (Sequence[callable]): Sequence of process object 
        to be composed.
  """
  def __init__(self, processes, *args, **kwargs):
    print('Init Process...')
    # assert isinstance(processes, collections.abc.Sequence)
    self.processes = []
    for process in processes:
      self.processes.append(process)
      # if callable(process):
      #     self.processes.append(process)
      # else:
      #     raise TypeError('process must be callable')


    print('processes: ', processes)

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

  def __repr__(self):
    format_string = self.__class__.__name__ + '('
    for t in self.processes:
      format_string += '\n'
      format_string += f'    {t}'
    format_string += '\n)'
    return format_string

  