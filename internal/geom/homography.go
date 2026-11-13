package geom

import "math"

// PerspectiveTransform represents a 2D planar projective transformation (homography)
// stored as a row-major 3x3 matrix H.
type PerspectiveTransform struct {
	H [9]float64
}

// SquareToQuadrilateral constructs the homography mapping the unit square
// (0,0), (1,0), (1,1), (0,1) to p0, p1, p2, p3 respectively.
func SquareToQuadrilateral(p0, p1, p2, p3 PointF) PerspectiveTransform {
	dx3 := p0.X - p1.X + p2.X - p3.X
	dy3 := p0.Y - p1.Y + p2.Y - p3.Y
	if math.Abs(dx3) < 1e-12 && math.Abs(dy3) < 1e-12 {
		// Affine case.
		return PerspectiveTransform{H: [9]float64{
			p1.X - p0.X, p2.X - p1.X, p0.X,
			p1.Y - p0.Y, p2.Y - p1.Y, p0.Y,
			0, 0, 1,
		}}
	}

	dx1 := p1.X - p2.X
	dx2 := p3.X - p2.X
	dy1 := p1.Y - p2.Y
	dy2 := p3.Y - p2.Y
	den := dx1*dy2 - dx2*dy1
	if math.Abs(den) < 1e-15 {
		return PerspectiveTransform{}
	}

	invDen := 1.0 / den
	g := (dx3*dy2 - dx2*dy3) * invDen
	h := (dx1*dy3 - dx3*dy1) * invDen
	return PerspectiveTransform{H: [9]float64{
		p1.X - p0.X + g*p1.X, p3.X - p0.X + h*p3.X, p0.X,
		p1.Y - p0.Y + g*p1.Y, p3.Y - p0.Y + h*p3.Y, p0.Y,
		g, h, 1,
	}}
}

// QuadrilateralToSquare constructs the homography mapping p0, p1, p2, p3
// to the unit square (0,0), (1,0), (1,1), (0,1) respectively.
func QuadrilateralToSquare(p0, p1, p2, p3 PointF) PerspectiveTransform {
	s := SquareToQuadrilateral(p0, p1, p2, p3)
	inv, ok := s.invert()
	if !ok {
		return PerspectiveTransform{}
	}
	return inv
}

// GeneralTransform computes the homography mapping src[i] -> dst[i] for i=0..3.
func GeneralTransform(src, dst Quad) (PerspectiveTransform, bool) {
	sSrc := SquareToQuadrilateral(src[0], src[1], src[2], src[3])
	srcToSq, ok := sSrc.invertNoCheck()
	if !ok {
		return PerspectiveTransform{}, false
	}
	sqToDst := SquareToQuadrilateral(dst[0], dst[1], dst[2], dst[3])
	if !sqToDst.IsValid() {
		return PerspectiveTransform{}, false
	}
	res := sqToDst.mul(srcToSq)
	if !res.IsValid() {
		return PerspectiveTransform{}, false
	}
	return res, true
}

// Project applies the homography to coordinates (u, v).
func (pt PerspectiveTransform) Project(u, v float64) PointF {
	den := pt.H[6]*u + pt.H[7]*v + pt.H[8]
	if math.Abs(den) < 1e-15 {
		return PointF{}
	}
	return PointF{
		X: (pt.H[0]*u + pt.H[1]*v + pt.H[2]) / den,
		Y: (pt.H[3]*u + pt.H[4]*v + pt.H[5]) / den,
	}
}

// ProjectPoint applies the homography to p.
func (pt PerspectiveTransform) ProjectPoint(p PointF) PointF {
	return pt.Project(p.X, p.Y)
}

// IsValid reports whether the homography matrix is finite and non-singular.
func (pt PerspectiveTransform) IsValid() bool {
	h := pt.H
	normSq := h[0]*h[0] + h[1]*h[1] + h[2]*h[2] +
		h[3]*h[3] + h[4]*h[4] + h[5]*h[5] +
		h[6]*h[6] + h[7]*h[7] + h[8]*h[8]
	if math.IsNaN(normSq) || math.IsInf(normSq, 0) || normSq == 0 {
		return false
	}
	det := pt.determinant()
	if normSq < 1e90 {
		return det*det > 1e-28*normSq*normSq*normSq
	}
	norm := math.Sqrt(normSq)
	return math.Abs(det) > 1e-14*norm*norm*norm
}

// RectangleQuad constructs a clockwise Quad for a rectangle of size (w, h)
// inset inward by inset on all sides.
func RectangleQuad(w, h, inset float64) Quad {
	return Quad{
		{X: inset, Y: inset},
		{X: w - inset, Y: inset},
		{X: w - inset, Y: h - inset},
		{X: inset, Y: h - inset},
	}
}

func (pt PerspectiveTransform) determinant() float64 {
	h := pt.H
	return h[0]*(h[4]*h[8]-h[5]*h[7]) -
		h[1]*(h[3]*h[8]-h[5]*h[6]) +
		h[2]*(h[3]*h[7]-h[4]*h[6])
}

func (pt PerspectiveTransform) invertNoCheck() (PerspectiveTransform, bool) {
	h := pt.H
	c00 := h[4]*h[8] - h[5]*h[7]
	c01 := -(h[3]*h[8] - h[5]*h[6])
	c02 := h[3]*h[7] - h[4]*h[6]
	c10 := -(h[1]*h[8] - h[2]*h[7])
	c11 := h[0]*h[8] - h[2]*h[6]
	c12 := -(h[0]*h[7] - h[1]*h[6])
	c20 := h[1]*h[5] - h[2]*h[4]
	c21 := -(h[0]*h[5] - h[2]*h[3])
	c22 := h[0]*h[4] - h[1]*h[3]

	det := h[0]*c00 + h[1]*c01 + h[2]*c02
	if math.IsNaN(det) || math.IsInf(det, 0) || math.Abs(det) < 1e-30 {
		return PerspectiveTransform{}, false
	}
	invDet := 1.0 / det
	return PerspectiveTransform{H: [9]float64{
		c00 * invDet, c10 * invDet, c20 * invDet,
		c01 * invDet, c11 * invDet, c21 * invDet,
		c02 * invDet, c12 * invDet, c22 * invDet,
	}}, true
}

func (pt PerspectiveTransform) invert() (PerspectiveTransform, bool) {
	if !pt.IsValid() {
		return PerspectiveTransform{}, false
	}
	res, ok := pt.invertNoCheck()
	if !ok || !res.IsValid() {
		return PerspectiveTransform{}, false
	}
	return res, true
}

func (pt PerspectiveTransform) mul(o PerspectiveTransform) PerspectiveTransform {
	a, b := pt.H, o.H
	return PerspectiveTransform{H: [9]float64{
		a[0]*b[0] + a[1]*b[3] + a[2]*b[6],
		a[0]*b[1] + a[1]*b[4] + a[2]*b[7],
		a[0]*b[2] + a[1]*b[5] + a[2]*b[8],
		a[3]*b[0] + a[4]*b[3] + a[5]*b[6],
		a[3]*b[1] + a[4]*b[4] + a[5]*b[7],
		a[3]*b[2] + a[4]*b[5] + a[5]*b[8],
		a[6]*b[0] + a[7]*b[3] + a[8]*b[6],
		a[6]*b[1] + a[7]*b[4] + a[8]*b[7],
		a[6]*b[2] + a[7]*b[5] + a[8]*b[8],
	}}
}
