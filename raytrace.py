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

