from numpy import array as npa

# read obj file
def parse_face(st, vertices):
  def to_offset(x):
    ret = int(x)
    if ret >= 0:
      ret -= 1
    return ret
  def parse_tri(t):
    v1 = t.split("/")[0]
    return vertices[to_offset(v1)]
  return map(parse_tri, st.split()[1:])

def load_obj(obj_file):
  obj = map(lambda x: x.strip(), open(obj_file).read().split("\n"))

  # read vertices
  vertices = npa(map(lambda x: map(float, x.split()[1:]), filter(lambda x: x[0:2] == "v ", obj)))
  #vertices += K*10

  # read in faces
  tris = map(lambda x: parse_face(x, vertices), filter(lambda x: x[0:2] == "f ", obj))
  print len(tris), "triangles"

  return tris

