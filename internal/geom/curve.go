package geom

import "math"

type tileEntry struct {
	hUV          PerspectiveTransform
	p00, p10     PointF
	p01, p11     PointF
	u0, v0       float64
	invDu, invDv float64
	useH         bool
	ok           bool
}

// BilinearTileGrid performs piecewise homography/bilinear interpolation over an
// MxM grid of alignment pattern points.
// ApM holds sorted module coordinates along both axes; ApP[row][col] holds the
// image point for module (ApM[col], ApM[row]); Valid[row][col] marks whether
// that grid node was reliably observed.
type BilinearTileGrid struct {
	ApM      []int
	ApP      [][]PointF
	Valid    [][]bool
	Fallback PerspectiveTransform

	tiles      []tileEntry
	tileLUT    []uint8
	gridDim    int
	fallbackOK bool
}

// NewBilinearTileGrid constructs a BilinearTileGrid from alignment pattern grid data
// and precomputes per-tile homographies and an O(1) module-to-tile lookup table.
func NewBilinearTileGrid(apM []int, apP [][]PointF, valid [][]bool, fallback PerspectiveTransform) *BilinearTileGrid {
	g := &BilinearTileGrid{
		ApM:        apM,
		ApP:        apP,
		Valid:      valid,
		Fallback:   fallback,
		fallbackOK: fallback.IsValid(),
	}
	g.buildCache()
	return g
}

func (g *BilinearTileGrid) buildCache() {
	m := len(g.ApM)
	if m < 2 || len(g.ApP) < m {
		return
	}
	gridDim := m - 1
	g.gridDim = gridDim
	g.tiles = make([]tileEntry, gridDim*gridDim)

	for iy := 0; iy < gridDim; iy++ {
		v0, v1 := float64(g.ApM[iy]), float64(g.ApM[iy+1])
		dv := v1 - v0
		for ix := 0; ix < gridDim; ix++ {
			u0, u1 := float64(g.ApM[ix]), float64(g.ApM[ix+1])
			du := u1 - u0
			idx := iy*gridDim + ix
			if math.Abs(du) < 1e-12 || math.Abs(dv) < 1e-12 {
				continue
			}
			p00, ok00 := g.nodePoint(iy, ix)
			p10, ok10 := g.nodePoint(iy, ix+1)
			p11, ok11 := g.nodePoint(iy+1, ix+1)
			p01, ok01 := g.nodePoint(iy+1, ix)
			if !ok00 || !ok10 || !ok11 || !ok01 {
				continue
			}

			invDu := 1.0 / du
			invDv := 1.0 / dv
			entry := tileEntry{
				p00:   p00,
				p10:   p10,
				p01:   p01,
				p11:   p11,
				u0:    u0,
				v0:    v0,
				invDu: invDu,
				invDv: invDv,
				ok:    true,
			}

			tileH := SquareToQuadrilateral(p00, p10, p11, p01)
			if tileH.IsValid() {
				aff := PerspectiveTransform{H: [9]float64{
					invDu, 0, -u0 * invDu,
					0, invDv, -v0 * invDv,
					0, 0, 1,
				}}
				entry.hUV = tileH.mul(aff)
				entry.useH = true
			}
			g.tiles[idx] = entry
		}
	}

	if g.ApM[0] >= 0 && gridDim <= 255 {
		maxCoord := g.ApM[m-2]
		if maxCoord >= 0 && maxCoord <= 4096 {
			g.tileLUT = make([]uint8, maxCoord+1)
			for c := 0; c <= maxCoord; c++ {
				g.tileLUT[c] = uint8(findTileIndex(g.ApM, float64(c)))
			}
		}
	}
}

// Project maps module coordinate (u, v) to image space using the local tile
// enclosing (u, v), falling back to Fallback for missing nodes or degenerate tiles.
func (g *BilinearTileGrid) Project(u, v float64) PointF {
	if len(g.tiles) == 0 {
		return g.projectUncached(u, v)
	}

	var ix, iy int
	if len(g.tileLUT) > 0 {
		iu := int(u)
		if u < 0 {
			ix = 0
		} else if iu >= len(g.tileLUT) {
			ix = g.gridDim - 1
		} else {
			ix = int(g.tileLUT[iu])
		}

		iv := int(v)
		if v < 0 {
			iy = 0
		} else if iv >= len(g.tileLUT) {
			iy = g.gridDim - 1
		} else {
			iy = int(g.tileLUT[iv])
		}
	} else {
		ix = findTileIndex(g.ApM, u)
		iy = findTileIndex(g.ApM, v)
	}

	t := &g.tiles[iy*g.gridDim+ix]
	if !t.ok {
		if g.fallbackOK {
			return g.Fallback.Project(u, v)
		}
		return PointF{}
	}

	if t.useH {
		h := t.hUV.H
		den := h[6]*u + h[7]*v + h[8]
		if math.Abs(den) > 1e-6 {
			invDen := 1.0 / den
			return PointF{
				X: (h[0]*u + h[1]*v + h[2]) * invDen,
				Y: (h[3]*u + h[4]*v + h[5]) * invDen,
			}
		}
	}

	s := (u - t.u0) * t.invDu
	tv := (v - t.v0) * t.invDv
	topX := t.p00.X + (t.p10.X-t.p00.X)*s
	topY := t.p00.Y + (t.p10.Y-t.p00.Y)*s
	botX := t.p01.X + (t.p11.X-t.p01.X)*s
	botY := t.p01.Y + (t.p11.Y-t.p01.Y)*s
	return PointF{
		X: topX + (botX-topX)*tv,
		Y: topY + (botY-topY)*tv,
	}
}

func (g *BilinearTileGrid) projectUncached(u, v float64) PointF {
	m := len(g.ApM)
	if m < 2 || len(g.ApP) < m {
		if g.Fallback.IsValid() {
			return g.Fallback.Project(u, v)
		}
		return PointF{}
	}

	ix := findTileIndex(g.ApM, u)
	iy := findTileIndex(g.ApM, v)

	p00, ok00 := g.nodePoint(iy, ix)
	p10, ok10 := g.nodePoint(iy, ix+1)
	p11, ok11 := g.nodePoint(iy+1, ix+1)
	p01, ok01 := g.nodePoint(iy+1, ix)
	if !ok00 || !ok10 || !ok11 || !ok01 {
		if g.Fallback.IsValid() {
			return g.Fallback.Project(u, v)
		}
		return PointF{}
	}

	u0, u1 := float64(g.ApM[ix]), float64(g.ApM[ix+1])
	v0, v1 := float64(g.ApM[iy]), float64(g.ApM[iy+1])
	du, dv := u1-u0, v1-v0
	if math.Abs(du) < 1e-12 || math.Abs(dv) < 1e-12 {
		if g.Fallback.IsValid() {
			return g.Fallback.Project(u, v)
		}
		return PointF{}
	}

	s := (u - u0) / du
	t := (v - v0) / dv

	tileH := SquareToQuadrilateral(p00, p10, p11, p01)
	if tileH.IsValid() && math.Abs(tileH.H[6]*s+tileH.H[7]*t+tileH.H[8]) > 1e-6 {
		return tileH.Project(s, t)
	}

	top := p00.Mul(1 - s).Add(p10.Mul(s))
	bot := p01.Mul(1 - s).Add(p11.Mul(s))
	return top.Mul(1 - t).Add(bot.Mul(t))
}

func (g *BilinearTileGrid) nodePoint(row, col int) (PointF, bool) {
	hasValid := g.Valid == nil || (row >= 0 && row < len(g.Valid) && col >= 0 && col < len(g.Valid[row]) && g.Valid[row][col])
	if hasValid && row >= 0 && row < len(g.ApP) && col >= 0 && col < len(g.ApP[row]) {
		return g.ApP[row][col], true
	}
	if g.Fallback.IsValid() && col >= 0 && col < len(g.ApM) && row >= 0 && row < len(g.ApM) {
		return g.Fallback.Project(float64(g.ApM[col]), float64(g.ApM[row])), true
	}
	return PointF{}, false
}

func findTileIndex(coords []int, val float64) int {
	m := len(coords)
	ix := 0
	for ix+1 < m-1 && val >= float64(coords[ix+1]) {
		ix++
	}
	return ix
}

// CoonsPatchGrid maps module coordinates (u, v) in [0, dim] x [0, dim] onto a
// curved quadrilateral bounded by 4 polylines via bilinearly blended Coons interpolation.
type CoonsPatchGrid struct {
	dim      int
	invDim   float64
	lutScale float64
	lutMax   int
	topLUT   []PointF
	botLUT   []PointF
	leftLUT  []PointF
	rightLUT []PointF
	c00      PointF
	c10      PointF
	c11      PointF
	c01      PointF
}

// NewCoonsPatchGrid constructs a CoonsPatchGrid from the grid dimension and 4
// boundary polylines (top: TL->TR, bottom: BL->BR, left: TL->BL, right: TR->BR),
// pre-resampling uniform-t lookup tables for O(1) Project calls.
func NewCoonsPatchGrid(dim int, top, bottom, left, right []PointF) *CoonsPatchGrid {
	g := &CoonsPatchGrid{
		dim: dim,
	}
	g.c00 = blendCorner(top, false, left, false)
	g.c10 = blendCorner(top, true, right, false)
	g.c11 = blendCorner(bottom, true, right, true)
	g.c01 = blendCorner(bottom, false, left, true)

	if dim <= 0 {
		return g
	}

	g.invDim = 1.0 / float64(dim)
	lutSteps := 2 * dim
	if lutSteps > 1024 {
		lutSteps = 1024
	}
	lutN := lutSteps + 1
	g.lutMax = lutSteps - 1
	g.lutScale = float64(lutSteps) * g.invDim

	buf := make([]PointF, 4*lutN)
	g.topLUT = buf[0:lutN]
	g.botLUT = buf[lutN : 2*lutN]
	g.leftLUT = buf[2*lutN : 3*lutN]
	g.rightLUT = buf[3*lutN : 4*lutN]

	invSteps := 1.0 / float64(lutSteps)
	for i := 0; i <= lutSteps; i++ {
		param := float64(i) * invSteps
		g.topLUT[i] = samplePolyline(top, param)
		g.botLUT[i] = samplePolyline(bottom, param)
		g.leftLUT[i] = samplePolyline(left, param)
		g.rightLUT[i] = samplePolyline(right, param)
	}
	return g
}

// Project computes S(s, t) = (1-t)*top(s) + t*bottom(s) + (1-s)*left(t) + s*right(t) - bilinearCorners(s, t)
// where s = u / float64(dim) and t = v / float64(dim).
func (g *CoonsPatchGrid) Project(u, v float64) PointF {
	if g.dim <= 0 {
		return PointF{}
	}
	s := u * g.invDim
	t := v * g.invDim

	fu := u * g.lutScale
	idxU := int(fu)
	if fu < 0 {
		idxU = 0
	} else if idxU > g.lutMax {
		idxU = g.lutMax
	}
	fracU := fu - float64(idxU)

	fv := v * g.lutScale
	idxV := int(fv)
	if fv < 0 {
		idxV = 0
	} else if idxV > g.lutMax {
		idxV = g.lutMax
	}
	fracV := fv - float64(idxV)

	t0 := g.topLUT[idxU]
	t1 := g.topLUT[idxU+1]
	topX := t0.X + (t1.X-t0.X)*fracU
	topY := t0.Y + (t1.Y-t0.Y)*fracU

	b0 := g.botLUT[idxU]
	b1 := g.botLUT[idxU+1]
	botX := b0.X + (b1.X-b0.X)*fracU
	botY := b0.Y + (b1.Y-b0.Y)*fracU

	l0 := g.leftLUT[idxV]
	l1 := g.leftLUT[idxV+1]
	leftX := l0.X + (l1.X-l0.X)*fracV
	leftY := l0.Y + (l1.Y-l0.Y)*fracV

	r0 := g.rightLUT[idxV]
	r1 := g.rightLUT[idxV+1]
	rightX := r0.X + (r1.X-r0.X)*fracV
	rightY := r0.Y + (r1.Y-r0.Y)*fracV

	omS := 1.0 - s
	omT := 1.0 - t
	w00 := omS * omT
	w10 := s * omT
	w01 := omS * t
	w11 := s * t

	bilX := g.c00.X*w00 + g.c10.X*w10 + g.c01.X*w01 + g.c11.X*w11
	bilY := g.c00.Y*w00 + g.c10.Y*w10 + g.c01.Y*w01 + g.c11.Y*w11

	return PointF{
		X: topX*omT + botX*t + leftX*omS + rightX*s - bilX,
		Y: topY*omT + botY*t + leftY*omS + rightY*s - bilY,
	}
}

func samplePolyline(poly []PointF, param float64) PointF {
	n := len(poly)
	if n == 0 {
		return PointF{}
	}
	if n == 1 {
		return poly[0]
	}
	f := param * float64(n-1)
	idx := int(math.Floor(f))
	if idx < 0 {
		idx = 0
	} else if idx > n-2 {
		idx = n - 2
	}
	frac := f - float64(idx)
	p0 := poly[idx]
	p1 := poly[idx+1]
	return PointF{
		X: p0.X + (p1.X-p0.X)*frac,
		Y: p0.Y + (p1.Y-p0.Y)*frac,
	}
}

func endpoint(poly []PointF, atEnd bool) (PointF, bool) {
	if len(poly) == 0 {
		return PointF{}, false
	}
	if atEnd {
		return poly[len(poly)-1], true
	}
	return poly[0], true
}

func blendCorner(polyA []PointF, endA bool, polyB []PointF, endB bool) PointF {
	a, okA := endpoint(polyA, endA)
	b, okB := endpoint(polyB, endB)
	switch {
	case okA && okB:
		return a.Midpoint(b)
	case okA:
		return a
	case okB:
		return b
	default:
		return PointF{}
	}
}

// RadialTransform models Brown-Conrady 2-term radial lens distortion (k1, k2)
// compatible with OpenCV undistort forward mapping and Newton-Raphson inversion.
type RadialTransform struct {
	cx, cy           float64
	normW, normH     float64
	invNormW         float64
	invNormH         float64
	k1, k2           float64
	k1Mul3, k2Mul5   float64
}

// NewRadialTransform constructs a RadialTransform centered at (cx, cy) with
// normalization scales (normW, normH) and radial coefficients (k1, k2).
func NewRadialTransform(cx, cy, normW, normH, k1, k2 float64) *RadialTransform {
	var invW, invH float64
	if normW != 0 && normH != 0 {
		invW = 1.0 / normW
		invH = 1.0 / normH
	}
	return &RadialTransform{
		cx:       cx,
		cy:       cy,
		normW:    normW,
		normH:    normH,
		invNormW: invW,
		invNormH: invH,
		k1:       k1,
		k2:       k2,
		k1Mul3:   3.0 * k1,
		k2Mul5:   5.0 * k2,
	}
}

// Forward maps an undistorted point p to the distorted source coordinate.
func (rt *RadialTransform) Forward(p PointF) PointF {
	if rt.invNormW == 0 || rt.invNormH == 0 {
		return p
	}
	xn := (p.X - rt.cx) * rt.invNormW
	yn := (p.Y - rt.cy) * rt.invNormH
	r2 := xn*xn + yn*yn
	factor := 1.0 + r2*(rt.k1+rt.k2*r2)
	return PointF{
		X: rt.cx + rt.normW*xn*factor,
		Y: rt.cy + rt.normH*yn*factor,
	}
}

// Inverse maps a distorted source coordinate src back to the undistorted point
// using Newton-Raphson iteration along the radial ray.
func (rt *RadialTransform) Inverse(src PointF) PointF {
	if rt.invNormW == 0 || rt.invNormH == 0 {
		return src
	}
	xd := (src.X - rt.cx) * rt.invNormW
	yd := (src.Y - rt.cy) * rt.invNormH
	rdSq := xd*xd + yd*yd
	if rdSq < 1e-24 {
		return src
	}
	rd := math.Sqrt(rdSq)

	r := rd
	for i := 0; i < 8; i++ {
		r2 := r * r
		f := r*(1.0+r2*(rt.k1+rt.k2*r2)) - rd
		df := 1.0 + r2*(rt.k1Mul3+rt.k2Mul5*r2)
		if math.Abs(df) < 1e-12 {
			break
		}
		delta := f / df
		r -= delta
		if r < 0 {
			r = 0
		}
		if math.Abs(delta) < 1e-12 {
			break
		}
	}

	scale := r / rd
	return PointF{
		X: rt.cx + rt.normW*xd*scale,
		Y: rt.cy + rt.normH*yd*scale,
	}
}

// CylindricalTransform models a cylindrical surface warp with center (cx, cy),
// cylinder radius, optional in-plane axis rotation tiltX (radians), and
// out-of-plane vertical pitch tiltY.
type CylindricalTransform struct {
	cx, cy               float64
	radius, invRadius    float64
	tiltX, tiltY         float64
	rTiltY               float64
	sinTiltX, cosTiltX   float64
	hasTiltX             bool
}

// NewCylindricalTransform constructs a CylindricalTransform.
func NewCylindricalTransform(cx, cy, radius, tiltX, tiltY float64) *CylindricalTransform {
	var invRadius float64
	if radius > 0 {
		invRadius = 1.0 / radius
	}
	sinA, cosA := 0.0, 1.0
	hasTiltX := tiltX != 0
	if hasTiltX {
		sinA, cosA = math.Sincos(tiltX)
	}
	return &CylindricalTransform{
		cx:        cx,
		cy:        cy,
		radius:    radius,
		invRadius: invRadius,
		tiltX:     tiltX,
		tiltY:     tiltY,
		rTiltY:    radius * tiltY,
		sinTiltX:  sinA,
		cosTiltX:  cosA,
		hasTiltX:  hasTiltX,
	}
}

// Forward maps a flat/warped destination coordinate (xd, yd) to the cylinder
// coordinate (xs, ys), matching cv2.remap cylindrical warp when tiltX == tiltY == 0.
func (ct *CylindricalTransform) Forward(p PointF) PointF {
	if ct.radius <= 0 {
		return p
	}
	theta := (p.X - ct.cx) * ct.invRadius
	sinT, cosT := math.Sincos(theta)
	ux := ct.radius * sinT
	uy := (p.Y-ct.cy)*cosT + ct.rTiltY*(1.0-cosT)

	if ct.hasTiltX {
		rx := ux*ct.cosTiltX - uy*ct.sinTiltX
		ry := ux*ct.sinTiltX + uy*ct.cosTiltX
		ux, uy = rx, ry
	}
	return PointF{X: ct.cx + ux, Y: ct.cy + uy}
}

// Inverse maps a cylinder coordinate (xs, ys) back to the flat coordinate (xd, yd).
// Returns false if |arg| >= 0.999 or radius <= 0.
func (ct *CylindricalTransform) Inverse(p PointF) (PointF, bool) {
	if ct.radius <= 0 {
		return PointF{}, false
	}
	ux := p.X - ct.cx
	uy := p.Y - ct.cy
	if ct.hasTiltX {
		rx := ux*ct.cosTiltX + uy*ct.sinTiltX
		ry := -ux*ct.sinTiltX + uy*ct.cosTiltX
		ux, uy = rx, ry
	}

	arg := ux * ct.invRadius
	if math.Abs(arg) >= 0.999 {
		return PointF{}, false
	}
	theta := math.Asin(arg)
	cosT := math.Sqrt(1.0 - arg*arg)
	xd := ct.cx + ct.radius*theta
	yd := ct.cy + (uy-ct.rTiltY*(1.0-cosT))/cosT
	return PointF{X: xd, Y: yd}, true
}

// WaveTransform models a 2D sinusoidal wave displacement:
// Forward(x, y) = (x + ampX*sin(freqX*y + phaseX), y + ampY*sin(freqY*x + phaseY)).
type WaveTransform struct {
	ampX, freqX, phaseX float64
	ampY, freqY, phaseY float64
}

// NewWaveTransform constructs a WaveTransform.
func NewWaveTransform(ampX, freqX, phaseX, ampY, freqY, phaseY float64) *WaveTransform {
	return &WaveTransform{
		ampX:   ampX,
		freqX:  freqX,
		phaseX: phaseX,
		ampY:   ampY,
		freqY:  freqY,
		phaseY: phaseY,
	}
}

// Forward applies the sinusoidal wave warp to p.
func (wt *WaveTransform) Forward(p PointF) PointF {
	return PointF{
		X: p.X + wt.ampX*math.Sin(wt.freqX*p.Y+wt.phaseX),
		Y: p.Y + wt.ampY*math.Sin(wt.freqY*p.X+wt.phaseY),
	}
}

// Inverse inverts the sinusoidal wave warp using fixed-point iteration.
func (wt *WaveTransform) Inverse(p PointF) PointF {
	if wt.ampX == 0 && wt.ampY == 0 {
		return p
	}
	if wt.ampY == 0 {
		return PointF{
			X: p.X - wt.ampX*math.Sin(wt.freqX*p.Y+wt.phaseX),
			Y: p.Y,
		}
	}
	if wt.ampX == 0 {
		return PointF{
			X: p.X,
			Y: p.Y - wt.ampY*math.Sin(wt.freqY*p.X+wt.phaseY),
		}
	}
	x, y := p.X, p.Y
	for i := 0; i < 32; i++ {
		nx := p.X - wt.ampX*math.Sin(wt.freqX*y+wt.phaseX)
		ny := p.Y - wt.ampY*math.Sin(wt.freqY*nx+wt.phaseY)
		if math.Abs(nx-x) < 1e-10 && math.Abs(ny-y) < 1e-10 {
			x, y = nx, ny
			break
		}
		x, y = nx, ny
	}
	return PointF{X: x, Y: y}
}
