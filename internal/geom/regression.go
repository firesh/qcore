package geom

import (
	"math"
	"slices"
)

// RegressionLine accumulates 2D edge points and fits an orthogonal total-least-squares
// line of the form Normal.X*x + Normal.Y*y = C, with optional outlier trimming.
type RegressionLine struct {
	points    []PointF
	normal    PointF
	c         float64
	dirInward PointF
	valid     bool
}

// Add appends a sample point to the regression line.
func (l *RegressionLine) Add(p PointF) {
	l.points = append(l.points, p)
	l.valid = false
}

// Points returns a copy of the accumulated (or trimmed) sample points.
func (l RegressionLine) Points() []PointF {
	return slices.Clone(l.points)
}

// IsValid reports whether Evaluate succeeded and the line parameters are valid.
func (l RegressionLine) IsValid() bool {
	return l.valid
}

// Normal returns the unit normal vector of the fitted line.
func (l RegressionLine) Normal() PointF {
	return l.normal
}

// C returns the offset c in the line equation Normal.Dot(p) = c.
func (l RegressionLine) C() float64 {
	return l.c
}

// Distance returns the unsigned perpendicular distance from p to the fitted line.
func (l RegressionLine) Distance(p PointF) float64 {
	return math.Abs(l.normal.Dot(p) - l.c)
}

// SetDirectionInward sets a reference inward direction and orients the line normal
// so that Normal.Dot(d) >= 0.
func (l *RegressionLine) SetDirectionInward(d PointF) {
	l.dirInward = d
	if l.valid && l.normal.Dot(d) < 0 {
		l.normal = l.normal.Neg()
		l.c = -l.c
	}
}

// IsHighRes reports whether the fitted line has enough samples and spatial span
// to be treated as a high-resolution edge.
func (l RegressionLine) IsHighRes() bool {
	if !l.valid || len(l.points) < 5 {
		return false
	}
	tangent := PointF{X: -l.normal.Y, Y: l.normal.X}
	minProj := tangent.Dot(l.points[0])
	maxProj := minProj
	for _, p := range l.points[1:] {
		proj := tangent.Dot(p)
		if proj < minProj {
			minProj = proj
		}
		if proj > maxProj {
			maxProj = proj
		}
	}
	return maxProj-minProj >= 8.0
}

// Evaluate fits the orthogonal regression line to the accumulated points.
// When maxDist > 0, all points must lie within maxDist of the line; if trim is
// true, worst outliers are iteratively removed (preserving at least half of
// the original points and at least 2 points) until all remaining points satisfy
// maxDist.
func (l *RegressionLine) Evaluate(maxDist float64, trim bool) bool {
	l.valid = false
	if len(l.points) < 2 {
		return false
	}

	pts := slices.Clone(l.points)
	n, c, ok := fitOrthogonalLine(pts)
	if !ok {
		return false
	}

	if maxDist > 0 {
		worstIdx, worstDist := worstPointDist(pts, n, c)
		if worstDist > maxDist {
			if !trim {
				return false
			}
			minPts := (len(l.points) + 1) / 2
			if minPts < 2 {
				minPts = 2
			}
			for worstDist > maxDist && len(pts) > minPts {
				pts = append(pts[:worstIdx], pts[worstIdx+1:]...)
				n, c, ok = fitOrthogonalLine(pts)
				if !ok {
					return false
				}
				worstIdx, worstDist = worstPointDist(pts, n, c)
			}
			if worstDist > maxDist {
				return false
			}
			l.points = pts
		}
	}

	if l.dirInward.LenSq() > 0 && n.Dot(l.dirInward) < 0 {
		n = n.Neg()
		c = -c
	}
	l.normal = n
	l.c = c
	l.valid = true
	return true
}

// IntersectLines computes the intersection of two valid non-parallel RegressionLines.
func IntersectLines(l1, l2 RegressionLine) (PointF, bool) {
	if !l1.IsValid() || !l2.IsValid() {
		return PointF{}, false
	}
	det := l1.normal.X*l2.normal.Y - l1.normal.Y*l2.normal.X
	if math.Abs(det) < 1e-9 {
		return PointF{}, false
	}
	x := (l1.c*l2.normal.Y - l2.c*l1.normal.Y) / det
	y := (l1.normal.X*l2.c - l2.normal.X*l1.c) / det
	return PointF{X: x, Y: y}, true
}

func fitOrthogonalLine(pts []PointF) (PointF, float64, bool) {
	if len(pts) < 2 {
		return PointF{}, 0, false
	}
	var mean PointF
	for _, p := range pts {
		mean.X += p.X
		mean.Y += p.Y
	}
	invN := 1.0 / float64(len(pts))
	mean.X *= invN
	mean.Y *= invN

	var sumXX, sumYY, sumXY float64
	for _, p := range pts {
		dx := p.X - mean.X
		dy := p.Y - mean.Y
		sumXX += dx * dx
		sumYY += dy * dy
		sumXY += dx * dy
	}
	if sumXX+sumYY < 1e-18 {
		return PointF{}, 0, false
	}

	angle := 0.5 * math.Atan2(2*sumXY, sumXX-sumYY)
	normal := PointF{X: -math.Sin(angle), Y: math.Cos(angle)}
	c := normal.Dot(mean)
	return normal, c, true
}

func worstPointDist(pts []PointF, normal PointF, c float64) (int, float64) {
	worstIdx := 0
	worstDist := -1.0
	for i, p := range pts {
		d := math.Abs(normal.Dot(p) - c)
		if d > worstDist {
			worstDist = d
			worstIdx = i
		}
	}
	return worstIdx, worstDist
}
