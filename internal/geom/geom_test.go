package geom

import (
	"math"
	"testing"
)

func almostEqual(a, b, tol float64) bool {
	return math.Abs(a-b) <= tol
}

func pointAlmostEqual(a, b PointF, tol float64) bool {
	return almostEqual(a.X, b.X, tol) && almostEqual(a.Y, b.Y, tol)
}

func TestPointOperations(t *testing.T) {
	a := PointF{X: 3, Y: 4}
	b := PointF{X: 1, Y: -2}

	if got := a.Add(b); got != (PointF{4, 2}) {
		t.Fatalf("Add = %v, want {4, 2}", got)
	}
	if got := a.Sub(b); got != (PointF{2, 6}) {
		t.Fatalf("Sub = %v, want {2, 6}", got)
	}
	if got := a.Mul(2.5); got != (PointF{7.5, 10}) {
		t.Fatalf("Mul = %v, want {7.5, 10}", got)
	}
	if got := a.Div(2); got != (PointF{1.5, 2}) {
		t.Fatalf("Div = %v, want {1.5, 2}", got)
	}
	if got := a.Div(0); got != (PointF{0, 0}) {
		t.Fatalf("Div(0) = %v, want {0, 0}", got)
	}
	if got := a.Dot(b); !almostEqual(got, -5, 1e-12) {
		t.Fatalf("Dot = %v, want -5", got)
	}
	if got := a.Cross(b); !almostEqual(got, -10, 1e-12) {
		t.Fatalf("Cross = %v, want -10", got)
	}
	if got := a.Len(); !almostEqual(got, 5, 1e-12) {
		t.Fatalf("Len = %v, want 5", got)
	}
	if got := a.LenSq(); !almostEqual(got, 25, 1e-12) {
		t.Fatalf("LenSq = %v, want 25", got)
	}
	if got := a.Dist(PointF{0, 0}); !almostEqual(got, 5, 1e-12) {
		t.Fatalf("Dist = %v, want 5", got)
	}
	if got := a.DistSq(PointF{0, 0}); !almostEqual(got, 25, 1e-12) {
		t.Fatalf("DistSq = %v, want 25", got)
	}
	if got := a.Normalize(); !pointAlmostEqual(got, PointF{0.6, 0.8}, 1e-12) {
		t.Fatalf("Normalize = %v, want {0.6, 0.8}", got)
	}
	if got := (PointF{}).Normalize(); got != (PointF{0, 0}) {
		t.Fatalf("Normalize(0) = %v, want {0, 0}", got)
	}
	if got := a.Midpoint(PointF{5, 6}); got != (PointF{4, 5}) {
		t.Fatalf("Midpoint = %v, want {4, 5}", got)
	}
	if got := (PointF{2.6, 3.4}).ToPointI(); got != (PointI{3, 3}) {
		t.Fatalf("ToPointI = %v, want {3, 3}", got)
	}
	if got := a.Neg(); got != (PointF{-3, -4}) {
		t.Fatalf("Neg = %v, want {-3, -4}", got)
	}
	if got := Centered(PointI{7, 9}); got != (PointF{7.5, 9.5}) {
		t.Fatalf("Centered = %v, want {7.5, 9.5}", got)
	}
}

func TestQuadBasicsAndOrdering(t *testing.T) {
	q := RectangleQuad(100, 80, 10)
	if q != (Quad{{10, 10}, {90, 10}, {90, 70}, {10, 70}}) {
		t.Fatalf("RectangleQuad = %v", q)
	}
	x0, y0, x1, y1 := q.BoundingBox()
	if x0 != 10 || y0 != 10 || x1 != 90 || y1 != 70 {
		t.Fatalf("BoundingBox = (%v,%v,%v,%v)", x0, y0, x1, y1)
	}
	if !almostEqual(q.Area(), 80*60, 1e-12) {
		t.Fatalf("Area = %v, want 4800", q.Area())
	}
	if !pointAlmostEqual(q.Center(), PointF{50, 40}, 1e-12) {
		t.Fatalf("Center = %v, want {50, 40}", q.Center())
	}
	if !q.ContainsPoint(PointF{50, 40}) {
		t.Fatal("expected center to be inside quad")
	}
	if q.ContainsPoint(PointF{5, 40}) {
		t.Fatal("expected outside point to be outside quad")
	}
	edges := q.Edges()
	if edges[0] != [2]PointF{{10, 10}, {90, 10}} || edges[3] != [2]PointF{{10, 70}, {10, 10}} {
		t.Fatalf("unexpected Edges = %v", edges)
	}

	// Shuffled corners should recover TL, TR, BR, BL order.
	shuffled := [4]PointF{q[2], q[0], q[3], q[1]}
	ordered := OrderCornersClockwise(shuffled)
	if ordered != q {
		t.Fatalf("OrderCornersClockwise = %v, want %v", ordered, q)
	}

	// Slightly rotated quad (15 deg) should also preserve TL, TR, BR, BL.
	angle := 15.0 * math.Pi / 180.0
	cosA, sinA := math.Cos(angle), math.Sin(angle)
	rot := func(p PointF) PointF {
		return PointF{X: 100 + p.X*cosA - p.Y*sinA, Y: 100 + p.X*sinA + p.Y*cosA}
	}
	rotQuad := Quad{
		rot(PointF{-40, -30}),
		rot(PointF{40, -30}),
		rot(PointF{40, 30}),
		rot(PointF{-40, 30}),
	}
	rotOrdered := OrderCornersClockwise([4]PointF{rotQuad[3], rotQuad[1], rotQuad[0], rotQuad[2]})
	for i := 0; i < 4; i++ {
		if !pointAlmostEqual(rotOrdered[i], rotQuad[i], 1e-9) {
			t.Fatalf("rotated corner %d = %v, want %v", i, rotOrdered[i], rotQuad[i])
		}
	}
}

func TestPolygonIoU(t *testing.T) {
	a := Quad{{0, 0}, {10, 0}, {10, 10}, {0, 10}}
	if got := PolygonIoU(a, a); !almostEqual(got, 1.0, 1e-12) {
		t.Fatalf("identical IoU = %v, want 1", got)
	}

	disjoint := Quad{{20, 0}, {30, 0}, {30, 10}, {20, 10}}
	if got := PolygonIoU(a, disjoint); got != 0 {
		t.Fatalf("disjoint IoU = %v, want 0", got)
	}

	// Half-overlapping unit squares: intersection=50, union=150 => 1/3.
	halfShift := Quad{{5, 0}, {15, 0}, {15, 10}, {5, 10}}
	if got := PolygonIoU(a, halfShift); !almostEqual(got, 1.0/3.0, 1e-12) {
		t.Fatalf("halfShift IoU = %v, want 1/3", got)
	}

	// Inner square of side 4 inside square of side 10: IoU = 16/100 = 0.16.
	inner := Quad{{3, 3}, {7, 3}, {7, 7}, {3, 7}}
	if got := PolygonIoU(a, inner); !almostEqual(got, 0.16, 1e-12) {
		t.Fatalf("contained IoU = %v, want 0.16", got)
	}

	// Square [-1,1]^2 and 45-degree rotated square of same size (octagonal intersection):
	// Exact area of intersection of two unit-halfside squares rotated by 45 deg is
	// 8*(sqrt(2)-1) ~ 3.3137084989847603; each square has area 4.
	sq1 := Quad{{-1, -1}, {1, -1}, {1, 1}, {-1, 1}}
	s2 := math.Sqrt(2)
	sq2 := Quad{{0, -s2}, {s2, 0}, {0, s2}, {-s2, 0}}
	wantInter := 8 * (s2 - 1)
	wantIoU := wantInter / (4 + 4 - wantInter)
	if got := PolygonIoU(sq1, sq2); !almostEqual(got, wantIoU, 1e-11) {
		t.Fatalf("rotated squares IoU = %v, want %v", got, wantIoU)
	}

	// Concave dart quad vs overlapping square: exact decomposition into triangles.
	// Dart vertices: (0,0), (4,0), (1,1), (0,4). Area = 4.
	dart := Quad{{0, 0}, {4, 0}, {1, 1}, {0, 4}}
	box := Quad{{0, 0}, {2, 0}, {2, 2}, {0, 2}}
	// Intersection with [0,2]x[0,2]:
	// Triangle (0,0)-(4,0)-(1,1) clipped to [0,2]x[0,2] loses x in [2,4],
	// where y ranges from 0 to (4-x)/3, so lost area = 0.5*2*(2/3) = 2/3 => inter1 = 2 - 2/3 = 4/3.
	// Symmetrically inter2 = 4/3, total inter = 8/3.
	// Union = 4 + 4 - 8/3 = 16/3. IoU = (8/3)/(16/3) = 0.5.
	if got := PolygonIoU(dart, box); !almostEqual(got, 0.5, 1e-11) {
		t.Fatalf("concave dart IoU = %v, want 0.5", got)
	}

	// Self-intersecting bowtie quad should not panic and should return valid [0,1] IoU.
	bowtie := Quad{{0, 0}, {10, 10}, {10, 0}, {0, 10}}
	if got := PolygonIoU(bowtie, a); got < 0 || got > 1 || math.IsNaN(got) {
		t.Fatalf("bowtie IoU = %v, expected finite value in [0,1]", got)
	}

	// Degenerate zero-area quad.
	degen := Quad{{0, 0}, {1, 1}, {2, 2}, {3, 3}}
	if got := PolygonIoU(degen, a); got != 0 {
		t.Fatalf("degenerate IoU = %v, want 0", got)
	}
}

func TestPerspectiveTransform(t *testing.T) {
	dst := Quad{
		{100, 50},
		{320, 80},
		{290, 310},
		{80, 260},
	}

	s2q := SquareToQuadrilateral(dst[0], dst[1], dst[2], dst[3])
	if !s2q.IsValid() {
		t.Fatal("expected SquareToQuadrilateral to be valid")
	}
	unit := [4]PointF{{0, 0}, {1, 0}, {1, 1}, {0, 1}}
	for i, u := range unit {
		got := s2q.ProjectPoint(u)
		if !pointAlmostEqual(got, dst[i], 1e-9) {
			t.Fatalf("s2q.Project(%v) = %v, want %v", u, got, dst[i])
		}
	}

	q2s := QuadrilateralToSquare(dst[0], dst[1], dst[2], dst[3])
	if !q2s.IsValid() {
		t.Fatal("expected QuadrilateralToSquare to be valid")
	}
	for i, p := range dst {
		got := q2s.ProjectPoint(p)
		if !pointAlmostEqual(got, unit[i], 1e-9) {
			t.Fatalf("q2s.Project(%v) = %v, want %v", p, got, unit[i])
		}
	}

	// GeneralTransform between module quad and image quad.
	src := RectangleQuad(21, 21, 0)
	gt, ok := GeneralTransform(src, dst)
	if !ok || !gt.IsValid() {
		t.Fatal("expected GeneralTransform to succeed")
	}
	for i := 0; i < 4; i++ {
		if got := gt.ProjectPoint(src[i]); !pointAlmostEqual(got, dst[i], 1e-8) {
			t.Fatalf("GeneralTransform corner %d = %v, want %v", i, got, dst[i])
		}
	}

	// Degenerate collinear points must report invalid.
	collinear := Quad{{0, 0}, {10, 10}, {20, 20}, {30, 30}}
	if _, ok := GeneralTransform(src, collinear); ok {
		t.Fatal("expected GeneralTransform on collinear quad to fail")
	}
	if (PerspectiveTransform{}).IsValid() {
		t.Fatal("zero PerspectiveTransform should be invalid")
	}
}

func TestRegressionLineAndIntersection(t *testing.T) {
	// Horizontal line y = 10 with one gross outlier at (5, 50).
	var l1 RegressionLine
	for i := 0; i < 10; i++ {
		l1.Add(PointF{X: float64(i * 5), Y: 10})
	}
	l1.Add(PointF{X: 5, Y: 50})

	// Without trimming, maxDist=0.5 should fail.
	if l1.Evaluate(0.5, false) {
		t.Fatal("expected Evaluate(0.5, false) to fail with outlier present")
	}
	if l1.IsValid() {
		t.Fatal("expected invalid line after failed Evaluate")
	}

	// With trimming, outlier should be dropped.
	if !l1.Evaluate(0.5, true) {
		t.Fatal("expected Evaluate(0.5, true) to succeed after trimming outlier")
	}
	if !l1.IsValid() {
		t.Fatal("expected valid line after Evaluate")
	}
	if len(l1.Points()) != 10 {
		t.Fatalf("trimmed Points len = %d, want 10", len(l1.Points()))
	}
	if !l1.IsHighRes() {
		t.Fatal("expected 10-point 45px line to be high-res")
	}

	// SetDirectionInward toward y = 20 (positive Y).
	l1.SetDirectionInward(PointF{X: 0, Y: 1})
	if l1.Normal().Dot(PointF{0, 1}) <= 0 {
		t.Fatalf("expected normal %v to point in +Y direction", l1.Normal())
	}
	if !almostEqual(l1.Distance(PointF{20, 13}), 3.0, 1e-9) {
		t.Fatalf("Distance = %v, want 3.0", l1.Distance(PointF{20, 13}))
	}

	// Vertical line x = 25.
	var l2 RegressionLine
	for i := 0; i < 6; i++ {
		l2.Add(PointF{X: 25, Y: float64(i * 4)})
	}
	l2.SetDirectionInward(PointF{X: 1, Y: 0})
	if !l2.Evaluate(0.5, false) {
		t.Fatal("expected vertical line Evaluate to succeed")
	}
	if l2.Normal().Dot(PointF{1, 0}) <= 0 {
		t.Fatalf("expected normal %v to point in +X direction", l2.Normal())
	}

	inter, ok := IntersectLines(l1, l2)
	if !ok {
		t.Fatal("expected intersection to succeed")
	}
	if !pointAlmostEqual(inter, PointF{25, 10}, 1e-9) {
		t.Fatalf("IntersectLines = %v, want {25, 10}", inter)
	}

	// Parallel lines should fail to intersect.
	var l3 RegressionLine
	l3.Add(PointF{0, 20})
	l3.Add(PointF{50, 20})
	if !l3.Evaluate(0.1, false) {
		t.Fatal("expected l3 Evaluate to succeed")
	}
	if _, ok := IntersectLines(l1, l3); ok {
		t.Fatal("expected parallel lines not to intersect")
	}

	// Short/low-res line.
	var short RegressionLine
	short.Add(PointF{0, 0})
	short.Add(PointF{2, 0})
	short.Add(PointF{4, 0})
	if !short.Evaluate(0.1, false) {
		t.Fatal("expected short line to evaluate")
	}
	if short.IsHighRes() {
		t.Fatal("expected 3-point 4px line not to be high-res")
	}
}

func TestBilinearTileGrid(t *testing.T) {
	// 3x3 alignment pattern grid over module coordinates [6, 22, 38] under perspective.
	src := RectangleQuad(45, 45, 0)
	dst := Quad{{50, 40}, {500, 60}, {480, 490}, {30, 460}}
	h, ok := GeneralTransform(src, dst)
	if !ok {
		t.Fatal("failed to build reference homography")
	}

	apM := []int{6, 22, 38}
	apP := make([][]PointF, 3)
	valid := make([][]bool, 3)
	for r := 0; r < 3; r++ {
		apP[r] = make([]PointF, 3)
		valid[r] = make([]bool, 3)
		for c := 0; c < 3; c++ {
			apP[r][c] = h.Project(float64(apM[c]), float64(apM[r]))
			valid[r][c] = true
		}
	}

	grid := NewBilinearTileGrid(apM, apP, valid, h)

	// Unperturbed grid must match the global homography both inside tiles and
	// when extrapolating outside the alignment pattern ring.
	for _, pt := range []PointF{{0.5, 6.0}, {14.5, 18.5}, {30.5, 30.5}, {44.5, 44.5}} {
		if got, want := grid.Project(pt.X, pt.Y), h.Project(pt.X, pt.Y); !pointAlmostEqual(got, want, 1e-6) {
			t.Fatalf("unperturbed grid.Project(%v) = %v, want %v", pt, got, want)
		}
	}

	// Perturb center node and mark bottom-right node invalid to test local warping
	// and Fallback substitution.
	apP[1][1] = apP[1][1].Add(PointF{X: 3, Y: -2})
	valid[2][2] = false
	grid = NewBilinearTileGrid(apM, apP, valid, h)

	// Center node should hit perturbed position exactly.
	if got := grid.Project(22, 22); !pointAlmostEqual(got, apP[1][1], 1e-6) {
		t.Fatalf("grid.Project(22,22) = %v, want %v", got, apP[1][1])
	}

	// Missing node (38, 38) falls back to h.Project(38, 38).
	if got, want := grid.Project(38, 38), h.Project(38, 38); !pointAlmostEqual(got, want, 1e-6) {
		t.Fatalf("missing node fallback = %v, want %v", got, want)
	}
}

func TestCoonsPatchGrid(t *testing.T) {
	dim := 21
	n := 22

	// Curved barrel-like patch: edges bow outward quadratically.
	top := make([]PointF, n)
	bottom := make([]PointF, n)
	left := make([]PointF, n)
	right := make([]PointF, n)
	for i := 0; i < n; i++ {
		s := float64(i) / float64(n-1)
		bow := 20.0 * 4.0 * s * (1.0 - s) // 0 at ends, +20 at midpoint
		top[i] = PointF{X: 100 + 200*s, Y: 100 - bow}
		bottom[i] = PointF{X: 100 + 200*s, Y: 300 + bow}
		left[i] = PointF{X: 100 - bow, Y: 100 + 200*s}
		right[i] = PointF{X: 300 + bow, Y: 100 + 200*s}
	}

	patch := NewCoonsPatchGrid(dim, top, bottom, left, right)

	// Boundaries must match exact polyline samples.
	for i := 0; i < n; i++ {
		u := float64(i)
		if got := patch.Project(u, 0); !pointAlmostEqual(got, top[i], 1e-9) {
			t.Fatalf("top(%v) = %v, want %v", u, got, top[i])
		}
		if got := patch.Project(u, float64(dim)); !pointAlmostEqual(got, bottom[i], 1e-9) {
			t.Fatalf("bottom(%v) = %v, want %v", u, got, bottom[i])
		}
		if got := patch.Project(0, u); !pointAlmostEqual(got, left[i], 1e-9) {
			t.Fatalf("left(%v) = %v, want %v", u, got, left[i])
		}
		if got := patch.Project(float64(dim), u); !pointAlmostEqual(got, right[i], 1e-9) {
			t.Fatalf("right(%v) = %v, want %v", u, got, right[i])
		}
	}

	// Center of symmetric patch must be (200, 200).
	if got := patch.Project(float64(dim)/2, float64(dim)/2); !pointAlmostEqual(got, PointF{200, 200}, 1e-9) {
		t.Fatalf("center = %v, want {200, 200}", got)
	}
}

func TestRadialTransform(t *testing.T) {
	// Barrel distortion (k1 < 0) and pincushion distortion (k1 > 0).
	cases := []struct {
		name   string
		k1, k2 float64
	}{
		{name: "barrel", k1: -0.28, k2: 0.08},
		{name: "pincushion", k1: 0.25, k2: 0.05},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rt := NewRadialTransform(320, 240, 320, 240, tc.k1, tc.k2)

			// Verify exact OpenCV forward formula at a test point.
			p := PointF{X: 480, Y: 360} // xn=0.5, yn=0.5, r2=0.5
			r2 := 0.5
			factor := 1.0 + tc.k1*r2 + tc.k2*r2*r2
			wantFwd := PointF{X: 320 + 320*0.5*factor, Y: 240 + 240*0.5*factor}
			gotFwd := rt.Forward(p)
			if !pointAlmostEqual(gotFwd, wantFwd, 1e-12) {
				t.Fatalf("Forward(%v) = %v, want %v", p, gotFwd, wantFwd)
			}

			// Verify roundtrips across a grid of points.
			for x := 80.0; x <= 560.0; x += 60.0 {
				for y := 60.0; y <= 420.0; y += 60.0 {
					pt := PointF{X: x, Y: y}
					distorted := rt.Forward(pt)
					recovered := rt.Inverse(distorted)
					if !pointAlmostEqual(recovered, pt, 1e-8) {
						t.Fatalf("roundtrip(%v) -> %v -> %v", pt, distorted, recovered)
					}
				}
			}
		})
	}
}

func TestCylindricalTransform(t *testing.T) {
	ct := NewCylindricalTransform(300, 200, 150, 0, 0)

	// Verify exact forward formula at xd=375 (theta=0.5), yd=280.
	p := PointF{X: 375, Y: 280}
	theta := 0.5
	wantFwd := PointF{
		X: 300 + 150*math.Sin(theta),
		Y: 200 + 80*math.Cos(theta),
	}
	gotFwd := ct.Forward(p)
	if !pointAlmostEqual(gotFwd, wantFwd, 1e-12) {
		t.Fatalf("Forward = %v, want %v", gotFwd, wantFwd)
	}

	gotInv, ok := ct.Inverse(gotFwd)
	if !ok {
		t.Fatal("expected Inverse to succeed")
	}
	if !pointAlmostEqual(gotInv, p, 1e-9) {
		t.Fatalf("Inverse = %v, want %v", gotInv, p)
	}

	// Out-of-bounds |arg| >= 0.999 must return false.
	if _, ok := ct.Inverse(PointF{X: 300 + 150*0.9995, Y: 200}); ok {
		t.Fatal("expected Inverse at |arg|>=0.999 to return false")
	}

	// Test with non-zero tiltX and tiltY roundtrip.
	tilted := NewCylindricalTransform(300, 200, 180, 0.25, -0.15)
	for dx := -120.0; dx <= 120.0; dx += 40.0 {
		for dy := -90.0; dy <= 90.0; dy += 45.0 {
			pt := PointF{X: 300 + dx, Y: 200 + dy}
			fwd := tilted.Forward(pt)
			back, ok := tilted.Inverse(fwd)
			if !ok {
				t.Fatalf("tilted Inverse failed for %v (fwd=%v)", pt, fwd)
			}
			if !pointAlmostEqual(back, pt, 1e-9) {
				t.Fatalf("tilted roundtrip %v -> %v -> %v", pt, fwd, back)
			}
		}
	}
}

func TestWaveTransform(t *testing.T) {
	wt := NewWaveTransform(12.0, 0.02, 0.5, 8.0, 0.015, 1.1)

	for x := 50.0; x <= 450.0; x += 50.0 {
		for y := 50.0; y <= 450.0; y += 50.0 {
			p := PointF{X: x, Y: y}
			warped := wt.Forward(p)
			recovered := wt.Inverse(warped)
			if !pointAlmostEqual(recovered, p, 1e-8) {
				t.Fatalf("Wave roundtrip %v -> %v -> %v", p, warped, recovered)
			}
		}
	}
}

var benchSinkPointF PointF
var benchSinkBool bool
var benchSinkPT PerspectiveTransform

func BenchmarkPerspectiveTransformProject(b *testing.B) {
	src := RectangleQuad(177, 177, 0)
	dst := Quad{{50, 40}, {850, 60}, {820, 860}, {30, 820}}
	h, _ := GeneralTransform(src, dst)
	b.ResetTimer()
	var pt PointF
	for i := 0; i < b.N; i++ {
		u := float64(i%177) + 0.5
		v := float64((i/177)%177) + 0.5
		pt = h.Project(u, v)
	}
	benchSinkPointF = pt
}

func BenchmarkGeneralTransform(b *testing.B) {
	src := RectangleQuad(177, 177, 0)
	dst := Quad{{50, 40}, {850, 60}, {820, 860}, {30, 820}}
	b.ResetTimer()
	var pt PerspectiveTransform
	var ok bool
	for i := 0; i < b.N; i++ {
		pt, ok = GeneralTransform(src, dst)
	}
	benchSinkPT = pt
	benchSinkBool = ok
}

func BenchmarkBilinearTileGridProject(b *testing.B) {
	// Version 40 (dim=177) has 7x7 alignment pattern grid.
	apM := []int{6, 30, 58, 86, 114, 142, 170}
	src := RectangleQuad(177, 177, 0)
	dst := Quad{{50, 40}, {850, 60}, {820, 860}, {30, 820}}
	h, _ := GeneralTransform(src, dst)
	m := len(apM)
	apP := make([][]PointF, m)
	valid := make([][]bool, m)
	for r := 0; r < m; r++ {
		apP[r] = make([]PointF, m)
		valid[r] = make([]bool, m)
		for c := 0; c < m; c++ {
			apP[r][c] = h.Project(float64(apM[c]), float64(apM[r]))
			valid[r][c] = true
		}
	}
	grid := NewBilinearTileGrid(apM, apP, valid, h)
	b.ResetTimer()
	var pt PointF
	for i := 0; i < b.N; i++ {
		u := float64(i%177) + 0.5
		v := float64((i/177)%177) + 0.5
		pt = grid.Project(u, v)
	}
	benchSinkPointF = pt
}

func BenchmarkCoonsPatchGridProject(b *testing.B) {
	dim := 177
	n := 45
	top := make([]PointF, n)
	bottom := make([]PointF, n)
	left := make([]PointF, n)
	right := make([]PointF, n)
	for i := 0; i < n; i++ {
		s := float64(i) / float64(n-1)
		bow := 20.0 * 4.0 * s * (1.0 - s)
		top[i] = PointF{X: 100 + 600*s, Y: 100 - bow}
		bottom[i] = PointF{X: 100 + 600*s, Y: 700 + bow}
		left[i] = PointF{X: 100 - bow, Y: 100 + 600*s}
		right[i] = PointF{X: 700 + bow, Y: 100 + 600*s}
	}
	patch := NewCoonsPatchGrid(dim, top, bottom, left, right)
	b.ResetTimer()
	var pt PointF
	for i := 0; i < b.N; i++ {
		u := float64(i%dim) + 0.5
		v := float64((i/dim)%dim) + 0.5
		pt = patch.Project(u, v)
	}
	benchSinkPointF = pt
}

func BenchmarkRadialTransformForward(b *testing.B) {
	rt := NewRadialTransform(320, 240, 320, 240, -0.28, 0.08)
	b.ResetTimer()
	var pt PointF
	for i := 0; i < b.N; i++ {
		p := PointF{X: float64(100 + i%440), Y: float64(80 + (i/440)%320)}
		pt = rt.Forward(p)
	}
	benchSinkPointF = pt
}

func BenchmarkRadialTransformInverse(b *testing.B) {
	rt := NewRadialTransform(320, 240, 320, 240, -0.28, 0.08)
	b.ResetTimer()
	var pt PointF
	for i := 0; i < b.N; i++ {
		p := PointF{X: float64(100 + i%440), Y: float64(80 + (i/440)%320)}
		pt = rt.Inverse(p)
	}
	benchSinkPointF = pt
}

func BenchmarkCylindricalTransformForward(b *testing.B) {
	ct := NewCylindricalTransform(300, 200, 180, 0.25, -0.15)
	b.ResetTimer()
	var pt PointF
	for i := 0; i < b.N; i++ {
		p := PointF{X: float64(180 + i%240), Y: float64(100 + (i/240)%200)}
		pt = ct.Forward(p)
	}
	benchSinkPointF = pt
}

func BenchmarkCylindricalTransformInverse(b *testing.B) {
	ct := NewCylindricalTransform(300, 200, 180, 0.25, -0.15)
	b.ResetTimer()
	var pt PointF
	var ok bool
	for i := 0; i < b.N; i++ {
		p := PointF{X: float64(180 + i%240), Y: float64(100 + (i/240)%200)}
		pt, ok = ct.Inverse(p)
	}
	benchSinkPointF = pt
	benchSinkBool = ok
}
