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
    self.processes = []
    for process in processes:
        if isinstance(process, dict):
            print('process: ', process)
            process_namespace = next(iter(process.values()))
            process_dict = vars(process_namespace)
            process = build_from_cfg(process_dict,
                                      PROCESS,
                                      default_args=None)
            self.processes.append(process)
        elif callable(process):
            self.processes.append(process)
        else:
            raise TypeError('process must be callable or a dict')
    print('processes out: ',  self.processes)
      # assert isinstance(cfg, dict) and 'type' in cfg
      # assert isinstance(default_args, dict) or default_args is None
      # args = cfg.copy()
      # obj_type = process_entriesargs.pop('type')
      # obj_type = 
      # if is_str(obj_type):
      #     obj_cls = registry.get(obj_type)
      #     if obj_cls is None:
      #         raise KeyError('{} is not in the {} registry'.format(
      #             obj_type, registry.name))
      # elif inspect.isclass(obj_type):
      #     obj_cls = obj_type
      # else:
      #     raise TypeError('type must be a str or valid type, but got {}'.format(
      #         type(obj_type)))
      # if default_args is not None:
      #     for name, value in default_args.items():
      #         args.setdefault(name, value)
      # return obj_cls(**args)
      # self.processes.append(process)
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

  