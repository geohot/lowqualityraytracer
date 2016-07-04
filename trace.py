#!/usr/bin/env python
import numpy as np
from numpy import array as npa
import scipy.misc

np.set_printoptions(suppress=True)


X = 60
Y = 60
arcrad_per_pixel = 0.007

# place the image point at the origin

I = np.array([1.0,0.0,0.0])
J = np.array([0.0,1.0,0.0])
K = np.array([0.0,0.0,1.0])

origin = -10*K + I + J

from scipy.linalg import expm3, norm

def M(axis, theta):
  return expm3(np.cross(np.eye(3), axis/norm(axis)*theta))

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

OBJ_FILE = "objs/cube.obj"
#OBJ_FILE = "objs/teapot.obj"
obj = map(lambda x: x.strip(), open(OBJ_FILE).read().split("\n"))

# read vertices
vertices = npa(map(lambda x: map(float, x.split()[1:]), filter(lambda x: x[0:2] == "v ", obj)))
#vertices += K*10

# read in faces
tris = map(lambda x: parse_face(x, vertices), filter(lambda x: x[0:2] == "f ", obj))
print len(tris), "triangles"

for i, tr in enumerate(tris):
  print i, tr

img = np.zeros((Y, X))

# do raytracing
for y in range(-Y/2, Y/2):
  for x in range(-X/2, X/2):
    rot = np.dot(M(I, x * arcrad_per_pixel), M(J, y * arcrad_per_pixel))
    ray = np.dot(rot, K)
    #print ray, x, y

    intersections = []
    # check ray for intersection with each triangle
    #   equation for triangle is point(u,v) = (1-u-v)*p0 + u*p1 + v*p2
    #     such that u >= 0, v >= 0, u + v <= 1.0
    #   equation for point is point(t) = p + t * d
    for tr in tris:
      e1 = tr[1] - tr[0]
      e2 = tr[2] - tr[0]
      h = np.cross(ray, e2)
      a = np.dot(e1, h)

      # 0 dot product = orthogonal
      if np.abs(a) < 0.00001:
        continue

      f = 1/a
      s = origin - tr[0]
      u = f * np.dot(s, h)

      if u < 0.0 or u > 1.0:
        continue

      q = np.cross(s, e1)
      v = f * np.dot(ray, q)

      if v < 0.0 or (u+v) > 1.0:
        continue

      t = f * np.dot(e2, q)

      # check work
      intersection_point = origin + t * ray
      triangle_point = (1-u-v)*tr[0] + u*tr[1] + v*tr[2]
      assert (intersection_point-triangle_point < 0.0001).all()

      # debugging
      """
      print "*** intersection"
      print "   ray", x, y, ray
      print "   tri", tr
      print "   intersects", t, intersection_point, u, v
      print triangle_point
      exit(0)
      """
      intersections.append(t)

    # err, shading?
    if len(intersections):
      img[y + Y/2][x + X/2] = (12.0-min(intersections))/4.0

# save the image
scipy.misc.imsave("out.png", img)

