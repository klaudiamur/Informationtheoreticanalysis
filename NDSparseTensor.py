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
      raise TypeError(f"shape must be of type 'int' or 'tuple', not '{type(shape).__name__}'")
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

  def __sliceindicies__(self, slice, arraylength):
    return [x for x in range(*slice.indices(arraylength))]

  def __slicelength__(self, slice, arraylength):
    return len(self.__sliceindicies__(slice, arraylength))

  def __getslice__(self, slice, arraylength):
    r = range(*slice.indices(arraylength))
    return [self[ii] for ii in r]

  def __getitem__(self, key):
    if isinstance(key, slice):
      return self.__getslice__(key, self.shape[0])
    elif isinstance(key, tuple):
      if not len(key) == self.dimension:
        raise KeyError(f"Dimensions must match. Given key '{key}' should be of length {self.dimension}")
      if all([isinstance(x, int) and x < self.shape[i] for i, x in enumerate(key)]):
        try:
          return self.elements[key]
        except KeyError:
          return 0
      shape = []
      for dim, k in enumerate(key):
        if isinstance(k, int):
          shape.append(1)
        elif isinstance(k, slice):
          shape.append(self.__slicelength__(k, arraylength=self.shape[dim]))
      shape = tuple(shape)
      submatrix = NDSparseTensor(shape)
      
      indicies = []
      for dim, k in enumerate(key):
        if isinstance(k, int):
          indicies.append([k])
        elif isinstance(k, slice):
          indicies.append(self.__sliceindicies__(k, arraylength=self.shape[dim]))
      
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
        
      return submatrix.reducedimension()
    elif isinstance(key, int):
      try:
        value = self.elements[key,]
      except KeyError:
        # could also be 0.0 if using floats...
        value = 0
      return value
    else:
      raise TypeError(f"Key must be of type 'int', 'slice' or 'tuple' of intigers or slices, not '{type(key).__name__}'")

  def __setitem__(self, key, value):
    if isinstance(key, int):
      key = key,
    if isinstance(key, tuple):
      if not len(key) == self.dimension:
         raise KeyError(f"Dimensions must match. Given key '{key}' should be of length {self.dimension}")
      elif not all([k < self.shape[i] for i, k in enumerate(key)]):
        raise IndexError(f"Index out of range. Key {key}, must be within {self.shape}")
    else:
      raise TypeError(f"Key must be of type 'int' or 'tuple' of intigers or slices, not '{type(key).__name__}'")

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
    return f"NDSparseTensor({self.shape})"

  def __str__(self):
    s = self.__repr__()+'\n  '
    return s+'\n  '.join([str(x) for x in self.elements.items()])

  def __mul__(self, other):
    if isinstance(other, int) or isinstance(other, float):
      for key, value in self.getnonzero():
        self.elements[key] *= other
      return self
    elif isinstance(other, NDSparseTensor):
      new = NDSparseTensor(shape=self.shape)
      for key, value in self.getnonzero():
        new[key] = self[key] * other[key]
      return new
    else:
      raise TypeError

  def __truediv__(self, other):
    if isinstance(other, int) or isinstance(other, float):
      for key, value in self.getnonzero():
        self.elements[key] /= other
      return self
    elif isinstance(other, NDSparseTensor):
      new = NDSparseTensor(shape=self.shape)
      for key, value in self.getnonzero():
        if not other[key] == 0:
          new[key] = self[key] / other[key]
        else:
          raise ZeroDivisionError
      return new
    else:
      raise TypeError

  def sum(self):
    return sum(self.elements.values())

  def normalize(self):
    self / self.sum()
    return self

  def isempty(self):
    return len(self.getnonzero()) == 0

  def getnonzero(self):
    return self.elements.items()

  def todense(self):
    dense = np.zeros(shape=self.shape)
    for key, value in self.elements.items():
      dense[key] = value
    return dense

  def reducedimension(self):
    old_shape = self.shape    
    redundant_dimensions = [dim for dim, size in enumerate(old_shape) if size == 1]

    new_shape = tuple([size for dim, size in enumerate(old_shape) if dim not in redundant_dimensions])

    reduced = NDSparseTensor(new_shape)
    for old_key, value in self.getnonzero():
      new_key = tuple([x for dim, x in enumerate(old_key) if dim not in redundant_dimensions])
      reduced[new_key] = self.elements[old_key]

    return reduced

  def tojson(self):
    return json.dumps({
      "NDSparseTensor": {
        "elements": [{' '.join([str(x) for x in key]): value} for key, value in self.getnonzero()],
        "shape": self.shape,
        "dimension": self.dimension
      }
    })

  @staticmethod
  def fromjson(json_string):
    dictionary = json.loads(json_string)
    shape = tuple(dictionary["NDSparseTensor"]["shape"])
    elements = dictionary["NDSparseTensor"]["elements"]
    matrix = NDSparseTensor(shape)
    
    for element in elements:
      for x in element.items():
        key, value = x
        key = tuple([int(x) for x in key.split(' ')])
        matrix[key] = value
    return matrix

if __name__ == "__main__":
  matrix = NDSparseTensor((5,5))
  matrix[0, 1] = 1
  matrix[1, 0] = 2
  matrix[1, 1] = 3
  matrix[1, 4] = 1
  matrix[2, 2] = 10
  
  json_string = matrix.tojson()
  print(json_string)

  matrix2 = NDSparseTensor.fromjson(json_string)
  print('matrix2:')
  print(matrix2.todense())

  print('Extract row:', matrix2[1, :].todense()) # Extract row
  print('Extract column:', matrix2[:, 1].todense()) # Extract column

  print(matrix)
  print(matrix / 2)
  print(matrix.sum())
  print(matrix.normalize())
  print(matrix.isempty())
  matrix.elements = {}
  print(matrix.isempty())
  
  matrix[1, 0] = 10
  # print(matrix.normalize())
  print(matrix2)
  print(matrix2.normalize())

  # print('Division\n', matrix / matrix2)

  # print(matrix2)
  # print(matrix2.update(np.log2))

