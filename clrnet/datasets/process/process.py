import collections
from clrnet.utils import build_from_cfg

from ..registry import PROCESS

class Process(object):
  """Compose multiple process sequentially.
  Args:
      process (Sequence[callable]): Sequence of process object 
        to be composed.
  """
  def __init__(self, processes):
    print('Init Process...')
    assert isinstance(processes, collections.abc.Sequence)
    print('processes in: ', processes)
    print('processes in len: ', len(processes))
    self.processes = []
    for process in processes:
        if isinstance(process, dict):
            print('process: ', process)
            process = build_from_cfg(process,
                                      PROCESS,
                                      default_args=None)
            self.processes.append(process)
        elif callable(process):
            self.processes.append(process)
        else:
            raise TypeError('process must be callable or a dict')
    print('processes out: ',  self.processes)

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

  