class Dict2Class(object):
    
  def __init__(self, my_dict):
    for key in my_dict:
      setattr(self, key, my_dict[key])
  
  def __repr__(self):
    format_string = self.__class__.__name__ + '('
    for k, v in vars(self).items():
      format_string += '\n'
      format_string += f'    {k}={v}'
    format_string += '\n)'
    return format_string