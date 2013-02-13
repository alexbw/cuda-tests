import numpy as np
triangle = np.array([[0., 0.],
                    [0., 1.],
                    [1., 0.]])
"""
// This function is a member of my Triangle class, which holds 3 3D vectors,
// a, b, and c which denote the 3 points making up the triangle.  The 3
// values for the barycentric coordinates get stored in lambda
Vector<3, T> baryCoords(Vector<2, T> vec) const
{
    Vector<3, T> lambda;
    T den = 1 / ((b.y - c.y) * (a.x - c.x) + (c.x - b.x) * (a.y - c.y));

    lambda.x = ((b.y - c.y) * (vec.x - c.x) + (c.x - b.x) * (vec.y - c.y)) * den;
    lambda.y = ((c.y - a.y) * (vec.x - c.x) + (a.x - c.x) * (vec.y - c.y)) * den;
    lambda.z = 1 - lambda.x - lambda.y;

    return lambda;
}

// I use it kinda like this:
Triangle<float> t; // assign it's ABC points
Vector<2, float> p; // Give it some position
Vector<3, float> lambda = t.baryCoords(p);
Vector<3, float> final = lambda.x * t.getA() + lambda.y * t.getB() + lambda.z * t.getC();
"""

def barycoords(coord, tri):
    """
    Coord is a 2D cartesian point (1D, 2-element nparray) that will be converted to 
    barycentric coordinates
    Tri is the triangle reference (2D, 6-element nparray) frame for the barycentric coordinate conversion.
    """

    assert len(coord) == 2, "Just need two cartesian points"
    assert tri.shape == (3,2), "Pass in a 3x2 array to define the points of the triangle"
    a,b,c = tri[0], tri[1], tri[2]
    den = 1.0 / ( (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1]))

    barycoord =  np.zeros((3,), dtype='float32')
    barycoord[0] = ((b[1] - c[1]) * (coord[0] - c[0]) + (c[0] - b[0]) * (coord[1] - c[1])) * den;
    barycoord[1] = ((c[1] - a[1]) * (coord[0] - c[0]) + (a[0] - c[0]) * (coord[1] - c[1])) * den;
    barycoord[2] = 1 - barycoord[0] - barycoord[1];
    return barycoord

print barycoords((1.,1.), triangle)