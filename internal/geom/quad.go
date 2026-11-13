package geom

import "math"

// Quad represents a quadrilateral with 4 vertices ordered clockwise in image
// coordinates: TL=0, TR=1, BR=2, BL=3.
type Quad [4]PointF

// BoundingBox returns the axis-aligned bounding box (minX, minY, maxX, maxY).
func (q Quad) BoundingBox() (minX, minY, maxX, maxY float64) {
	minX, maxX = q[0].X, q[0].X
	minY, maxY = q[0].Y, q[0].Y
	for i := 1; i < 4; i++ {
		if q[i].X < minX {
			minX = q[i].X
		}
		if q[i].X > maxX {
			maxX = q[i].X
		}
		if q[i].Y < minY {
			minY = q[i].Y
		}
		if q[i].Y > maxY {
			maxY = q[i].Y
		}
	}
	return minX, minY, maxX, maxY
}

// Area returns the unsigned area of the quadrilateral using the shoelace formula.
func (q Quad) Area() float64 {
	return math.Abs(signedPolyArea(q[:]))
}

// Center returns the arithmetic centroid of the 4 vertices.
func (q Quad) Center() PointF {
	return PointF{
		X: (q[0].X + q[1].X + q[2].X + q[3].X) * 0.25,
		Y: (q[0].Y + q[1].Y + q[2].Y + q[3].Y) * 0.25,
	}
}

// ContainsPoint reports whether p lies inside the quadrilateral using ray-casting.
func (q Quad) ContainsPoint(p PointF) bool {
	inside := false
	j := 3
	for i := 0; i < 4; i++ {
		yi, yj := q[i].Y, q[j].Y
		if (yi > p.Y) != (yj > p.Y) {
			xi, xj := q[i].X, q[j].X
			xIntersect := (xj-xi)*(p.Y-yi)/(yj-yi) + xi
			if p.X < xIntersect {
				inside = !inside
			}
		}
		j = i
	}
	return inside
}

// Edges returns the 4 directed edges of the quadrilateral in order.
func (q Quad) Edges() [4][2]PointF {
	return [4][2]PointF{
		{q[0], q[1]},
		{q[1], q[2]},
		{q[2], q[3]},
		{q[3], q[0]},
	}
}

// OrderCornersClockwise orders 4 points clockwise in image coordinates (Y-down):
// TL=0, TR=1, BR=2, BL=3.
func OrderCornersClockwise(pts [4]PointF) Quad {
	c := Quad(pts).Center()
	type angled struct {
		pt    PointF
		angle float64
	}
	var items [4]angled
	for i, p := range pts {
		items[i] = angled{
			pt:    p,
			angle: math.Atan2(p.Y-c.Y, p.X-c.X),
		}
	}
	if items[1].angle < items[0].angle {
		items[0], items[1] = items[1], items[0]
	}
	if items[3].angle < items[2].angle {
		items[2], items[3] = items[3], items[2]
	}
	if items[2].angle < items[0].angle {
		items[0], items[2] = items[2], items[0]
	}
	if items[3].angle < items[1].angle {
		items[1], items[3] = items[3], items[1]
	}
	if items[2].angle < items[1].angle {
		items[1], items[2] = items[2], items[1]
	}

	// Rotate so the top-left corner (minimum X+Y, tie-breaking by Y then X) is at index 0.
	start := 0
	bestSum := items[0].pt.X + items[0].pt.Y
	for i := 1; i < 4; i++ {
		s := items[i].pt.X + items[i].pt.Y
		if s < bestSum-1e-9 || (math.Abs(s-bestSum) <= 1e-9 && (items[i].pt.Y < items[start].pt.Y ||
			(items[i].pt.Y == items[start].pt.Y && items[i].pt.X < items[start].pt.X))) {
			bestSum = s
			start = i
		}
	}

	var out Quad
	for i := 0; i < 4; i++ {
		out[i] = items[(start+i)%4].pt
	}
	return out
}

type quadParts struct {
	polys [2][4]PointF
	lens  [2]int
	count int
}

// PolygonIoU computes the exact Intersection-over-Union of two quadrilaterals
// using Sutherland-Hodgman polygon clipping and the shoelace area formula.
// Simple concave quads are decomposed into triangles; self-intersecting quads
// gracefully fall back to their convex hull.
func PolygonIoU(a, b Quad) float64 {
	// Quick AABB rejection.
	ax0, ay0, ax1, ay1 := a.BoundingBox()
	bx0, by0, bx1, by1 := b.BoundingBox()
	if ax1 <= bx0 || bx1 <= ax0 || ay1 <= by0 || by1 <= ay0 {
		return 0
	}

	partsA, areaA := decomposeQuad(a)
	partsB, areaB := decomposeQuad(b)
	if areaA <= 0 || areaB <= 0 {
		return 0
	}

	var bufA, bufB [8]PointF
	var interArea float64
	for i := 0; i < partsA.count; i++ {
		pa := partsA.polys[i][:partsA.lens[i]]
		for j := 0; j < partsB.count; j++ {
			pb := partsB.polys[j][:partsB.lens[j]]
			clipped := sutherlandHodgman(pa, pb, &bufA, &bufB)
			if len(clipped) >= 3 {
				interArea += math.Abs(signedPolyArea(clipped))
			}
		}
	}

	if interArea <= 0 {
		return 0
	}
	unionArea := areaA + areaB - interArea
	if unionArea <= 0 {
		return 0
	}
	iou := interArea / unionArea
	if iou > 1 {
		iou = 1
	}
	return iou
}

// decomposeQuad returns convex CCW polygons representing q and its non-negative area.
// If q is convex, it returns a single 4-vertex polygon.
// If q is simple and concave, it splits q into 2 triangles along the reflex diagonal.
// If q is self-intersecting, it falls back to the convex hull.
func decomposeQuad(q Quad) (quadParts, float64) {
	var out quadParts
	if isSelfIntersectingQuad(q) {
		var hullBuf [4]PointF
		copy(hullBuf[:], q[:])
		n := convexHullInPlace(hullBuf[:])
		if n < 3 {
			return out, 0
		}
		copy(out.polys[0][:n], hullBuf[:n])
		out.lens[0] = n
		out.count = 1
		area := math.Abs(signedPolyArea(out.polys[0][:n]))
		return out, area
	}

	pts := q
	sa := signedPolyArea(pts[:])
	if math.Abs(sa) < 1e-12 {
		return out, 0
	}
	if sa < 0 {
		// Reverse to ensure CCW orientation (positive signed area).
		pts[1], pts[3] = pts[3], pts[1]
		sa = -sa
	}

	// Check turn signs at each vertex to detect concavity.
	reflexIdx := -1
	for i := 0; i < 4; i++ {
		prev := pts[(i+3)%4]
		curr := pts[i]
		next := pts[(i+1)%4]
		cross := curr.Sub(prev).Cross(next.Sub(curr))
		if cross < -1e-12 {
			reflexIdx = i
			break
		}
	}

	if reflexIdx < 0 {
		out.polys[0] = pts
		out.lens[0] = 4
		out.count = 1
		return out, sa
	}

	// Simple concave quad: split into 2 CCW triangles along the reflex diagonal.
	k := reflexIdx
	out.polys[0] = [4]PointF{pts[k], pts[(k+1)%4], pts[(k+2)%4]}
	out.lens[0] = 3
	out.polys[1] = [4]PointF{pts[k], pts[(k+2)%4], pts[(k+3)%4]}
	out.lens[1] = 3
	out.count = 2
	return out, sa
}

// sutherlandHodgman clips subject against a convex CCW clip polygon using stack buffers.
func sutherlandHodgman(subject, clip []PointF, bufA, bufB *[8]PointF) []PointF {
	if len(subject) < 3 || len(clip) < 3 {
		return nil
	}
	nOut := len(subject)
	copy(bufA[:nOut], subject)
	cur := bufA
	nxt := bufB

	for i := 0; i < len(clip); i++ {
		if nOut == 0 {
			return nil
		}
		a := clip[i]
		b := clip[(i+1)%len(clip)]
		edge := b.Sub(a)

		nIn := nOut
		nOut = 0

		s := cur[nIn-1]
		ds := edge.Cross(s.Sub(a))
		sIn := ds >= -1e-12

		for k := 0; k < nIn; k++ {
			e := cur[k]
			de := edge.Cross(e.Sub(a))
			eIn := de >= -1e-12
			if eIn {
				if !sIn {
					nxt[nOut] = clipIntersect(s, e, ds, de)
					nOut++
				}
				nxt[nOut] = e
				nOut++
			} else if sIn {
				nxt[nOut] = clipIntersect(s, e, ds, de)
				nOut++
			}
			s = e
			ds = de
			sIn = eIn
		}
		cur, nxt = nxt, cur
	}
	return cur[:nOut]
}

func clipIntersect(s, e PointF, ds, de float64) PointF {
	denom := ds - de
	if math.Abs(denom) < 1e-15 {
		return s.Midpoint(e)
	}
	t := ds / denom
	return PointF{
		X: s.X + t*(e.X-s.X),
		Y: s.Y + t*(e.Y-s.Y),
	}
}

func signedPolyArea(pts []PointF) float64 {
	n := len(pts)
	if n < 3 {
		return 0
	}
	var sum float64
	j := n - 1
	for i := 0; i < n; i++ {
		sum += pts[j].X*pts[i].Y - pts[i].X*pts[j].Y
		j = i
	}
	return 0.5 * sum
}

func isSelfIntersectingQuad(q Quad) bool {
	return segmentsProperlyIntersect(q[0], q[1], q[2], q[3]) ||
		segmentsProperlyIntersect(q[1], q[2], q[3], q[0])
}

func segmentsProperlyIntersect(a, b, c, d PointF) bool {
	ab := b.Sub(a)
	cd := d.Sub(c)
	c1 := ab.Cross(c.Sub(a))
	c2 := ab.Cross(d.Sub(a))
	c3 := cd.Cross(a.Sub(c))
	c4 := cd.Cross(b.Sub(c))
	return ((c1 > 1e-12 && c2 < -1e-12) || (c1 < -1e-12 && c2 > 1e-12)) &&
		((c3 > 1e-12 && c4 < -1e-12) || (c3 < -1e-12 && c4 > 1e-12))
}

func convexHullInPlace(pts []PointF) int {
	n := len(pts)
	// Insertion sort by (X, Y) — n <= 4.
	for i := 1; i < n; i++ {
		key := pts[i]
		j := i - 1
		for j >= 0 && (pts[j].X > key.X || (pts[j].X == key.X && pts[j].Y > key.Y)) {
			pts[j+1] = pts[j]
			j--
		}
		pts[j+1] = key
	}

	// Deduplicate coincident points.
	uLen := 0
	for i := 0; i < n; i++ {
		if uLen == 0 || pts[uLen-1].DistSq(pts[i]) > 1e-18 {
			pts[uLen] = pts[i]
			uLen++
		}
	}
	if uLen < 3 {
		return uLen
	}

	var hull [8]PointF
	hLen := 0
	for i := 0; i < uLen; i++ {
		p := pts[i]
		for hLen >= 2 && hull[hLen-1].Sub(hull[hLen-2]).Cross(p.Sub(hull[hLen-2])) <= 0 {
			hLen--
		}
		hull[hLen] = p
		hLen++
	}
	lowerLen := hLen + 1
	for i := uLen - 2; i >= 0; i-- {
		p := pts[i]
		for hLen >= lowerLen && hull[hLen-1].Sub(hull[hLen-2]).Cross(p.Sub(hull[hLen-2])) <= 0 {
			hLen--
		}
		hull[hLen] = p
		hLen++
	}
	hLen--
	copy(pts[:hLen], hull[:hLen])
	return hLen
}
