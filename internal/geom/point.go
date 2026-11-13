// Package geom provides 2D geometric primitives, perspective homographies,
// orthogonal regression lines, polygon IoU evaluation, and curved surface transforms
// for the qore QR code engine.
package geom

import "math"

// PointF represents a 2D point or vector with float64 coordinates.
type PointF struct {
	X float64
	Y float64
}

// PointI represents a 2D point or pixel coordinate with integer coordinates.
type PointI struct {
	X int
	Y int
}

// Centered converts an integer pixel coordinate to its float64 pixel-center
// coordinate by adding 0.5 to each axis.
func Centered(p PointI) PointF {
	return PointF{X: float64(p.X) + 0.5, Y: float64(p.Y) + 0.5}
}

// Add returns the vector sum p + o.
func (p PointF) Add(o PointF) PointF {
	return PointF{X: p.X + o.X, Y: p.Y + o.Y}
}

// Sub returns the vector difference p - o.
func (p PointF) Sub(o PointF) PointF {
	return PointF{X: p.X - o.X, Y: p.Y - o.Y}
}

// Mul returns the scalar product p * s.
func (p PointF) Mul(s float64) PointF {
	return PointF{X: p.X * s, Y: p.Y * s}
}

// Div returns the scalar quotient p / s, or the zero vector if s == 0.
func (p PointF) Div(s float64) PointF {
	if s == 0 {
		return PointF{}
	}
	return PointF{X: p.X / s, Y: p.Y / s}
}

// Dot returns the dot product p.X*o.X + p.Y*o.Y.
func (p PointF) Dot(o PointF) float64 {
	return p.X*o.X + p.Y*o.Y
}

// Cross returns the 2D scalar cross product p.X*o.Y - p.Y*o.X.
func (p PointF) Cross(o PointF) float64 {
	return p.X*o.Y - p.Y*o.X
}

// Len returns the Euclidean norm of p.
func (p PointF) Len() float64 {
	return math.Hypot(p.X, p.Y)
}

// LenSq returns the squared Euclidean norm of p.
func (p PointF) LenSq() float64 {
	return p.X*p.X + p.Y*p.Y
}

// Dist returns the Euclidean distance between p and o.
func (p PointF) Dist(o PointF) float64 {
	return math.Hypot(p.X-o.X, p.Y-o.Y)
}

// DistSq returns the squared Euclidean distance between p and o.
func (p PointF) DistSq(o PointF) float64 {
	dx := p.X - o.X
	dy := p.Y - o.Y
	return dx*dx + dy*dy
}

// Normalize returns a unit vector in the direction of p, or the zero vector
// if p has zero length.
func (p PointF) Normalize() PointF {
	l := p.Len()
	if l == 0 {
		return PointF{}
	}
	return PointF{X: p.X / l, Y: p.Y / l}
}

// Midpoint returns the midpoint between p and o.
func (p PointF) Midpoint(o PointF) PointF {
	return PointF{X: (p.X + o.X) * 0.5, Y: (p.Y + o.Y) * 0.5}
}

// ToPointI rounds p to the nearest integer PointI coordinates.
func (p PointF) ToPointI() PointI {
	return PointI{
		X: int(math.Round(p.X)),
		Y: int(math.Round(p.Y)),
	}
}

// Neg returns the negated vector -p.
func (p PointF) Neg() PointF {
	return PointF{X: -p.X, Y: -p.Y}
}
