import numpy as np
from numpy import array as npa
import scipy.misc


X = 80
Y = 80
arcrad_per_pixel = 0.005

# place the image point at the origin
origin = np.array([0.0,0.0,0.0])

I = np.array([1.0,0.0,0.0])
J = np.array([0.0,1.0,0.0])
K = np.array([0.0,0.0,1.0])

from scipy.linalg import expm3, norm

def M(axis, theta):
  return expm3(np.cross(np.eye(3), axis/norm(axis)*theta))

# read obj file
def parse_face(st, vertices):
  def parse_tri(t):
    v1, vt1, vn1 = t.split("/")
    return vertices[int(v1)]
  return map(parse_tri, st.split(" ")[1:])

obj = open("cube.obj").read().split("\n")

# read vertices
vertices = npa(map(lambda x: map(float, x.split(" ")[1:]), filter(lambda x: x[0:2] == "v ", obj)))
vertices += K*10

# read in faces
tris = map(lambda x: parse_face(x, vertices), filter(lambda x: x[0:2] == "f ", obj))
print len(tris), "triangles"

img = np.zeros((Y, X))

# do raytracing
for y in range(Y):
  for x in range(X):
    x_rad = (x - (X/2)) * arcrad_per_pixel
    y_rad = (y - (Y/2)) * arcrad_per_pixel
    rot = np.dot(M(I, x_rad), M(J, y_rad))
    ray = np.dot(rot, K)

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

      if v < 0.0 or v > 1.0:
        continue

      t = f * np.dot(e2, q)

      print t, tr, ray, x, y
      img[y][x] = 1.0 


# save the image
scipy.misc.imsave("out.png", img)

