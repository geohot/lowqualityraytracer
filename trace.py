#!/usr/bin/env python
import time
import numpy as np
from numpy import array as npa
import scipy.misc
from helpers import load_obj

from scipy.linalg import expm3, norm
from skimage.draw import line_aa

# *** basic math things ***

def M(axis, theta):
  return expm3(np.cross(np.eye(3), axis/norm(axis)*theta))

I = np.array([1.0,0.0,0.0])
J = np.array([0.0,1.0,0.0])
K = np.array([0.0,0.0,1.0])

np.set_printoptions(suppress=True)

# *** image params ***
#X,Y,arcrad_per_pixel = 80,80,0.007
X,Y,arcrad_per_pixel = 600,600,0.001

# *** drawing functions ***

# fast triangle mesh drawer draws triangle meshes and is hella fast
def fast_triangle_mesh_drawer(tris, origin, look):
  img = np.zeros((Y, X))
  lines = []
  for tr in tris:
    lines.append((tr[0], tr[1]))
    lines.append((tr[1], tr[2]))
    lines.append((tr[2], tr[0]))

  def project_point(pt):
    # vector from pt to origin
    v = pt - origin
    v /= norm(v)

    # real projection shit
    vx = npa((v[0], 0, v[2]))
    vy = npa((0, v[1], v[2]))
    vx /= norm(vx)
    vy /= norm(vy)

    x = (np.arccos(np.dot(vx, look)) / arcrad_per_pixel)
    if v[0] < 0.0:
      x = -x
    y = (np.arccos(np.dot(vy, look)) / arcrad_per_pixel)
    if v[1] < 0.0:
      y = -y

    """
    print " *** "
    print v, K
    print pt, x, y
    """

    return int(round(x + X/2)),int(round(Y/2 - y))

  for l1, l2 in lines:
    pt1, pt2 = project_point(l1), project_point(l2)
    rr, cc, val = line_aa(pt1[1], pt1[0], pt2[1], pt2[0])

    # filter
    rr[rr < 0] = 0
    cc[cc < 0] = 0
    rr[rr >= X] = 0
    cc[cc >= Y] = 0

    img[rr, cc] = val
  return img

# raytracer sends a ray out from the camera and is hella slow
def raytrace(tris, origin, look):
  img = np.zeros((Y, X))
  # do raytracing
  for y in range(-Y/2, Y/2):
    for x in range(-X/2, X/2):
      rot = np.dot(M(I, x * arcrad_per_pixel), M(J, y * arcrad_per_pixel))
      ray = np.dot(rot, look)
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
  return img

# *** do shit
#tris = load_obj("objs/cube.obj")
tris = load_obj("objs/teapot.obj")

SCALE = 1/10.0

origin = -500*K + I + J
look = K

import pygame
pygame.init()
screen = pygame.display.set_mode((X,Y), pygame.DOUBLEBUF)
surf = pygame.surface.Surface((X,Y),0,8)
surf.set_palette([(x,x,x) for x in range(256)])

while 1:
  img = fast_triangle_mesh_drawer(tris, origin, look)

  pygame.surfarray.blit_array(surf, img.swapaxes(0,1) * 255.0)
  screen.blit(surf, (0,0))
  pygame.display.flip()

  for event in pygame.event.get():
    if event.type == pygame.QUIT:
      pygame.quit()

  keys_pressed = pygame.key.get_pressed()

  if keys_pressed[pygame.K_LEFT]:
    origin += I*SCALE
  if keys_pressed[pygame.K_RIGHT]:
    origin -= I*SCALE
  if keys_pressed[pygame.K_UP]:
    origin -= J*SCALE
  if keys_pressed[pygame.K_DOWN]:
    origin += J*SCALE

# save the image
#scipy.misc.imsave("out.png", img)

