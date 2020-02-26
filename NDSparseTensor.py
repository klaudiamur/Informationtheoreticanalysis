import json
import numpy as np

class NDSparseTensor:
  def __init__(self, shape):
    self.elements = {}
    if isinstance(shape, int):
      self.shape = shape,
    elif isinstance(shape, tuple):
      self.shape = shape
    else:
      raise TypeError
    self.dimension = len(self.shape)

  def __index_exists__(self, i, arr):
    try:
        value = arr[i]
    except IndexError:
        return False
    return True

  def __numberofelements__(self):
    value = 1
    for v in self.shape:
      value *= v
    return value

  def __sliceindicies__(self, slice):
    return [x for x in range(*slice.indices(self.__numberofelements__()))]

  def __slicelength__(self, slice):
    return len(self.__sliceindicies__(slice))

  def __getslice__(self, slice):
    r = range(*slice.indices(self.__numberofelements__()))
    return [self.__getitem__(ii) for ii in r]

  def __getitem__(self, key):
    if isinstance(key, slice):
      return self.__getslice__(key)
    elif isinstance(key, tuple):
      if len(key) > self.dimension:
        raise IndexError
      if all([isinstance(x, int) and x < self.shape[i] for i, x in enumerate(key)]):
        try:
          return self.elements[key]
        except KeyError:
          return 0
      shape = []
      for k in key:
        if isinstance(k, int):
          shape.append(k)
        elif isinstance(k, slice):
          shape.append(self.__slicelength__(k))
      shape = tuple(shape)
      submatrix = NDSparseTensor(shape)
      
      indicies = []
      for k in key:
        if isinstance(k, int):
          indicies.append(k)
        elif isinstance(k, slice):
          indicies.append(self.__sliceindicies__(k))
      
      stack = [0 for _ in range(len(key))]
      p = len(stack) - 1
      while p >= 0:
        while self.__index_exists__(p + 1, stack):
          p += 1 # Move right

        new = tuple(stack)
        old = tuple([indicies[i][x] for i, x in enumerate(stack)])
        val = self[old]
        submatrix[new] = val

        stack[p] += 1

        while stack[p] >= shape[p] and p >= 0:
          p -= 1 # Move left
          stack[p + 1] = 0
          stack[p] += 1
        
      return submatrix
    else:
      try:
        value = self.elements[key]
      except KeyError:
        # could also be 0.0 if using floats...
        value = 0
      return value

  def __setitem__(self, key, value):
    if not value == 0:
      self.elements[key] = value
    else:
      try:
        del self.elements[key]
      except Exception:
        pass

  def __delitem__(self, key):
    del self.elements[key]

  def __repr__(self):
    return "NDSparseTensor()"

  def __str__(self):
    s = f'NDSparseTensor of size {self.shape}\n  '
    return s+'\n  '.join([str(x) for x in self.elements.items()])

  def getnonzero(self):
    return self.elements.items()

  def todense(self):
    dense = np.zeros(shape=self.shape)
    for key, value in self.elements.items():
      dense[key] = value
    return dense

  def tojson(self):
    return json.dumps({
      "NDSparseTensor": {
        "elements": [{' '.join([str(x) for x in key]): value} for key, value in self.getnonzero()],
        "shape": self.shape,
        "dimension": self.dimension
      }
    })

  def fromjson(self, json_string):
    dictionary = json.loads(json_string)
    shape = tuple(dictionary["NDSparseTensor"]["shape"])
    elements = dictionary["NDSparseTensor"]["elements"]
    matrix = NDSparseTensor(shape)
    
    for element in elements:
      for x in element.items():
        key, value = x
        key = tuple(key.split(' '))
        matrix[key] = value
    return matrix

if __name__ == "__main__":
  matrix = NDSparseTensor((5,5))
  matrix[0, 1] = 1
  matrix[1, 0] = 2
  matrix[1, 1] = 3
  matrix[1, 4] = 1
  matrix[2, 2] = 10
  print(matrix.todense())
  

  print(matrix[:3, :3].todense())

  json_string = matrix.tojson()
  print(json_string)

  print(matrix.fromjson(json_string))

  matrix = matrix.fromjson(json_string)