#!/usr/bin/env python
import time
import numpy as np
from numpy import array as npa
import scipy.misc
import random
from helpers import load_obj

from scipy.linalg import expm3, norm
from skimage.draw import line_aa
from skimage.draw import polygon

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
  # shouldn't do anything
  look /= norm(look)

  img = np.zeros((Y, X))

  def project_point(pt):
    # vector from pt to origin
    vv = pt - origin
    v = vv/norm(vv)

    # real projection shit
    vx = npa((v[0], 0, v[2]))
    lx = npa((look[0], 0, look[2]))
    vy = npa((0, v[1], v[2]))
    ly = npa((0, look[1], look[2]))
    def ang(v1, v2):
      v1 /= norm(v1)
      v2 /= norm(v2)
      angl = np.dot(v1, v2)
      crs = np.cross(v1, v2)
      if np.sum(crs) >= 0.0:
        return np.arccos(angl)
      else:
        return -np.arccos(angl)

    x = (ang(vx, lx) / arcrad_per_pixel)
    y = (ang(vy, ly) / arcrad_per_pixel)

    # add z for z-buffering
    # z is the distance of the point from the plane formed by look and origin

    # project v on to look
    z = np.dot(v, look) * norm(vv)

    """
    print " *** "
    print v, K
    print pt, x, y
    """

    return int(round(x + X/2)),int(round(Y/2 - y)), z

  # project the triangles into 2D space
  # does this projection preserve the u and v
  DRAW_WIREFRAME = False
  if DRAW_WIREFRAME:
    lines = []
    for tr in tris:
      p0, p1, p2 = project_point(tr[0]), project_point(tr[1]), project_point(tr[2])
      lines.append((p0[0:2], p1[0:2]))
      lines.append((p1[0:2], p2[0:2]))
      lines.append((p2[0:2], p0[0:2]))

    for pt1, pt2 in lines:
      rr, cc, val = line_aa(pt1[1], pt1[0], pt2[1], pt2[0])

      # filter
      rr[np.logical_or(rr < 0, rr >= Y)] = 0
      cc[np.logical_or(cc < 0, cc >= X)] = 0

      img[rr, cc] = val
  else:
    # z-buffering
    polys = []
    for tr in tris:
      xyz = npa(map(project_point, tr))
      polys.append((np.mean(xyz[:, 2]), xyz))
    polys = sorted(polys, reverse=True, key=lambda x: x[0])
    for _, xyz in polys:
      rr, cc = polygon(xyz[:, 1], xyz[:, 0])

      rr[np.logical_or(rr < 0, rr >= Y)] = 0
      cc[np.logical_or(cc < 0, cc >= X)] = 0
      img[rr, cc] = random.uniform(0.3, 1.0)
  return img

# *** do shit
tris = load_obj("objs/cube.obj")
#tris = load_obj("objs/teapot.obj")

SCALE = 1/10.0
LSCALE = 1/100.0

origin = -10*K + I + J
#origin = -500*K + I + J
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

  # moving
  if keys_pressed[pygame.K_a]:
    origin += I*SCALE
  if keys_pressed[pygame.K_d]:
    origin -= I*SCALE
  if keys_pressed[pygame.K_w]:
    origin -= J*SCALE
  if keys_pressed[pygame.K_s]:
    origin += J*SCALE

  # looking, DOESN'T WORK!
  if keys_pressed[pygame.K_LEFT]:
    look += I*LSCALE
  if keys_pressed[pygame.K_RIGHT]:
    look -= I*LSCALE
  if keys_pressed[pygame.K_UP]:
    look += J*LSCALE
  if keys_pressed[pygame.K_DOWN]:
    look -= J*LSCALE
  look /= norm(look)

  print "drawing", origin, look

# save the image
#scipy.misc.imsave("out.png", img)

