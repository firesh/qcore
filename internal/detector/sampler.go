package detector

import (
	"math"
	"runtime"
	"sync"
	"sync/atomic"

	"github.com/firesh/qcore/internal/decoder"
	"github.com/firesh/qcore/internal/geom"
	"github.com/firesh/qcore/internal/imageproc"
)

func sampleModule(gray *imageproc.GrayImage, bm *imageproc.BitMatrix, integ *imageproc.IntegralImage, pt geom.PointF) (bool, uint8) {
	if bm != nil {
		if pt.X < 0 || pt.Y < 0 {
			return false, 0
		}
		ix := int(pt.X)
		iy := int(pt.Y)
		w, h := bm.Width, bm.Height
		if ix >= w || iy >= h {
			return false, 0
		}
		bits := bm.Bits
		row := iy * w
		centerOn := bits[row+ix] != 0

		// Fast path: if fractional coordinates are inside [0.25, 0.75)^2, all 5 ±0.25px
		// sub-pixel taps fall into the exact same pixel (ix, iy).
		fx := pt.X - float64(ix)
		fy := pt.Y - float64(iy)
		if fx >= 0.25 && fx < 0.75 && fy >= 0.25 && fy < 0.75 {
			return centerOn, 210
		}

		blackW := 0
		if centerOn {
			blackW = 2
		}
		// Left (-0.25, 0)
		if fx >= 0.25 {
			if centerOn {
				blackW++
			}
		} else if ix > 0 {
			if bits[row+ix-1] != 0 {
				blackW++
			}
		} else if centerOn {
			blackW++
		}
		// Right (+0.25, 0)
		if fx < 0.75 {
			if centerOn {
				blackW++
			}
		} else if ix+1 < w {
			if bits[row+ix+1] != 0 {
				blackW++
			}
		} else if centerOn {
			blackW++
		}
		// Up (0, -0.25)
		if fy >= 0.25 {
			if centerOn {
				blackW++
			}
		} else if iy > 0 {
			if bits[(iy-1)*w+ix] != 0 {
				blackW++
			}
		} else if centerOn {
			blackW++
		}
		// Down (0, +0.25)
		if fy < 0.75 {
			if centerOn {
				blackW++
			}
		} else if iy+1 < h {
			if bits[(iy+1)*w+ix] != 0 {
				blackW++
			}
		} else if centerOn {
			blackW++
		}

		bit := blackW >= 3
		if blackW == 0 || blackW == 6 {
			return bit, 210
		}

		agree := math.Abs(float64(2*blackW - 6)) // 0, 2, or 4
		c := 40.0 + agree*20.0
		if gray != nil && gray.Width > 0 && gray.Height > 0 && integ != nil {
			v := gray.SampleBilinear(pt.X, pt.Y)
			thresh := imageproc.LocalThresholdAt(gray, integ, pt.X, pt.Y, 31)
			contrast := math.Abs(thresh - v)
			c = agree*16.0 + contrast*4.0
		}
		if c > 255.0 {
			c = 255.0
		}
		if c < 1.0 {
			c = 1.0
		}
		return bit, uint8(c)
	}

	if gray != nil && gray.Width > 0 && gray.Height > 0 {
		if pt.X < -2 || pt.Y < -2 || pt.X > float64(gray.Width)+2 || pt.Y > float64(gray.Height)+2 {
			return false, 0
		}
		v := (2.0*gray.SampleBilinear(pt.X, pt.Y) +
			gray.SampleBilinear(pt.X-0.25, pt.Y) +
			gray.SampleBilinear(pt.X+0.25, pt.Y) +
			gray.SampleBilinear(pt.X, pt.Y-0.25) +
			gray.SampleBilinear(pt.X, pt.Y+0.25)) / 6.0

		thresh := imageproc.LocalThresholdAt(gray, integ, pt.X, pt.Y, 31)
		diff := thresh - v
		bit := diff > 0
		c := math.Abs(diff) * 8.0
		if c > 255.0 {
			c = 255.0
		}
		if c < 1.0 {
			c = 1.0
		}
		return bit, uint8(c)
	}

	return false, 0
}

func sampleBitFast(gray *imageproc.GrayImage, bm *imageproc.BitMatrix, integ *imageproc.IntegralImage, pt geom.PointF) (bit bool, inBounds bool) {
	if bm != nil {
		if pt.X < 0 || pt.Y < 0 {
			return false, false
		}
		ix := int(pt.X)
		iy := int(pt.Y)
		if ix >= bm.Width || iy >= bm.Height {
			return false, false
		}
		return bm.Bits[iy*bm.Width+ix] != 0, true
	}
	b, c := sampleModule(gray, nil, integ, pt)
	return b, c > 0
}

type gridValidateStats struct {
	valid      bool
	coreHits   int
	rowRatio   float64
	colRatio   float64
	fmtOK      bool
	fmtHD      int
	verMatched bool
}

// quickValidateGrid performs a microsecond-scale geometric & timing-pattern pre-check
// before full grid sampling and Reed-Solomon decoding.
func quickValidateGrid(gray *imageproc.GrayImage, bm *imageproc.BitMatrix, integ *imageproc.IntegralImage, dim int, proj func(u, v float64) geom.PointF) bool {
	return quickValidateGridWithStats(gray, bm, integ, dim, proj).valid
}

func quickValidateGridWithStats(gray *imageproc.GrayImage, bm *imageproc.BitMatrix, integ *imageproc.IntegralImage, dim int, proj func(u, v float64) geom.PointF) gridValidateStats {
	d := float64(dim)
	var st gridValidateStats

	// 1) At least 2 of the 3 finder pattern cores must land inside the image on black pixels.
	coreHits := 0
	for _, cp := range [3][2]float64{{3.5, 3.5}, {d - 3.5, 3.5}, {3.5, d - 3.5}} {
		if b, ok := sampleBitFast(gray, bm, integ, proj(cp[0], cp[1])); ok && b {
			coreHits++
		}
	}
	st.coreHits = coreHits
	if coreHits < 2 {
		return st
	}

	// 2) Horizontal (row 6) & vertical (col 6) alternating timing pattern checks.
	// Use an odd step (3) for large dimensions so black/white alternation is still tested.
	step := 1
	if dim > 57 {
		step = 3
	}
	rowMatch, rowTot := 0, 0
	colMatch, colTot := 0, 0
	for k := 8; k <= dim-9; k += step {
		exp := (k & 1) == 0
		if b, ok := sampleBitFast(gray, bm, integ, proj(float64(k)+0.5, 6.5)); ok {
			rowTot++
			if b == exp {
				rowMatch++
			}
		}
		if b, ok := sampleBitFast(gray, bm, integ, proj(6.5, float64(k)+0.5)); ok {
			colTot++
			if b == exp {
				colMatch++
			}
		}
	}
	if rowTot == 0 && colTot == 0 {
		return st
	}

	rowRatio := 0.0
	if rowTot > 0 {
		rowRatio = float64(rowMatch) / float64(rowTot)
	}
	colRatio := 0.0
	if colTot > 0 {
		colRatio = float64(colMatch) / float64(colTot)
	}
	st.rowRatio = rowRatio
	st.colRatio = colRatio

	if dim == 21 {
		if (rowMatch >= 3 && colMatch >= 3) || rowMatch == rowTot || colMatch == colTot {
			st.valid = true
			return st
		}
	} else {
		if (rowRatio >= 0.62 && colRatio >= 0.50) ||
			(colRatio >= 0.62 && rowRatio >= 0.50) ||
			rowRatio >= 0.72 || colRatio >= 0.72 {
			st.valid = true
			return st
		}
	}

	// 3) Fallback for damaged timing patterns: allow if format info has <= 1 bit error
	// (plus weak timing agreement), or for version >= 7 if version bits match dim.
	var f1, f2 uint16
	readBit := func(u, v int) uint16 {
		if b, ok := sampleBitFast(gray, bm, integ, proj(float64(u)+0.5, float64(v)+0.5)); ok && b {
			return 1
		}
		return 0
	}
	// Format copy 1 around TL in (u=col, v=row): row 8 cols 0..5,7,8, then col 8 rows 7,5..0.
	fCoords1 := [15][2]int{
		{0, 8}, {1, 8}, {2, 8}, {3, 8}, {4, 8}, {5, 8}, {7, 8}, {8, 8},
		{8, 7}, {8, 5}, {8, 4}, {8, 3}, {8, 2}, {8, 1}, {8, 0},
	}
	for _, p := range fCoords1 {
		f1 = (f1 << 1) | readBit(p[0], p[1])
	}
	// Format copy 2 around BL/TR: (dim-1..dim-7, 8)? Wait: rows dim-1..dim-7 col 8, then (8, dim-8..dim-1).
	for r := dim - 1; r >= dim-7; r-- {
		f2 = (f2 << 1) | readBit(8, r)
	}
	for c := dim - 8; c < dim; c++ {
		f2 = (f2 << 1) | readBit(c, 8)
	}
	if fi, ok := decoder.DecodeFormatInformation(f1, f2); ok {
		st.fmtOK = true
		st.fmtHD = fi.HammingDist
		if fi.HammingDist <= 1 && (rowRatio >= 0.45 || colRatio >= 0.45) {
			st.valid = true
			return st
		}
	}

	if dim >= 45 {
		var vb1, vb2 uint32
		for r := 0; r < 6; r++ {
			for c := 0; c < 3; c++ {
				if readBit(dim-11+c, r) != 0 {
					vb1 |= 1 << (r*3 + c)
				}
			}
		}
		for c := 0; c < 6; c++ {
			for r := 0; r < 3; r++ {
				if readBit(c, dim-11+r) != 0 {
					vb2 |= 1 << (c*3 + r)
				}
			}
		}
		if ver, okV := decoder.DecodeVersionInformation(vb1, vb2); okV && 17+4*ver == dim {
			st.verMatched = true
			st.valid = true
			return st
		}
	}

	return st
}

func sampleAndDecode(
	gray *imageproc.GrayImage,
	bm *imageproc.BitMatrix,
	integ *imageproc.IntegralImage,
	dim int,
	proj func(u, v float64) geom.PointF,
) (*decoder.DecodeResult, geom.Quad, bool) {
	res, quad, _, ok := sampleAndDecodeWithStats(gray, bm, integ, dim, proj)
	return res, quad, ok
}

func sampleAndDecodeWithStats(
	gray *imageproc.GrayImage,
	bm *imageproc.BitMatrix,
	integ *imageproc.IntegralImage,
	dim int,
	proj func(u, v float64) geom.PointF,
) (*decoder.DecodeResult, geom.Quad, gridValidateStats, bool) {
	if dim < 21 || dim > 177 || (dim-21)%4 != 0 {
		return nil, geom.Quad{}, gridValidateStats{}, false
	}

	st := quickValidateGridWithStats(gray, bm, integ, dim, proj)
	if !st.valid {
		return nil, geom.Quad{}, st, false
	}

	res, quad, ok := sampleAndDecodeValidated(gray, bm, integ, dim, proj)
	return res, quad, st, ok
}

func sampleAndDecodeValidated(
	gray *imageproc.GrayImage,
	bm *imageproc.BitMatrix,
	integ *imageproc.IntegralImage,
	dim int,
	proj func(u, v float64) geom.PointF,
) (*decoder.DecodeResult, geom.Quad, bool) {
	if dim < 21 || dim > 177 || (dim-21)%4 != 0 {
		return nil, geom.Quad{}, false
	}

	flatBits := make([]bool, dim*dim)
	flatConf := make([]uint8, dim*dim)
	bits := make([][]bool, dim)
	conf := make([][]uint8, dim)
	for r := 0; r < dim; r++ {
		off := r * dim
		bitsRow := flatBits[off : off+dim]
		confRow := flatConf[off : off+dim]
		v := float64(r) + 0.5
		for c := 0; c < dim; c++ {
			pt := proj(float64(c)+0.5, v)
			bitsRow[c], confRow[c] = sampleModule(gray, bm, integ, pt)
		}
		bits[r] = bitsRow
		conf[r] = confRow
	}

	res, err := decoder.DecodeMatrix(bits, conf)
	if err != nil || res == nil {
		return nil, geom.Quad{}, false
	}

	quad := geom.Quad{
		proj(0, 0),
		proj(float64(dim), 0),
		proj(float64(dim), float64(dim)),
		proj(0, float64(dim)),
	}
	return res, quad, true
}

// estimateModuleSizeAlongRay measures the 1:1:3:1:1 finder module size at a along
// the ray toward b using edge-to-edge run-lengths (accurate under arbitrary rotation/tilt).
func estimateModuleSizeAlongRay(bm *imageproc.BitMatrix, a, b geom.PointF, fallbackMS float64) float64 {
	if bm == nil {
		return fallbackMS
	}
	dir := b.Sub(a).Normalize()
	if dir.LenSq() == 0 {
		return fallbackMS
	}
	cur := imageproc.NewCursorF(bm, a.X, a.Y, dir.X, dir.Y)
	maxLen := int(math.Ceil(math.Max(14.0, fallbackMS*18.0)))
	counts, ok := cur.ReadSymmetricPattern5(maxLen)
	if !ok || !FindPattern(counts, true) {
		return fallbackMS
	}
	total := counts[0] + counts[1] + counts[2] + counts[3] + counts[4]
	// Edge-to-edge inner 5 modules: (2*total - counts[0] - counts[4]) / 12.0
	ms := (2.0*float64(total) - float64(counts[0]) - float64(counts[4])) / 12.0
	if ms <= 0.5 || (fallbackMS > 0 && (ms < fallbackMS*0.45 || ms > fallbackMS*2.1)) {
		return fallbackMS
	}
	return ms
}

// estimateCandidateDimensions computes plausible QR grid dimensions from directional
// ray-measured module sizes along the TL->TR and TL->BL finder legs.
func estimateCandidateDimensions(bm *imageproc.BitMatrix, set FinderPatternSet) []int {
	dTop := dist(set.TL.Center, set.TR.Center)
	dLeft := dist(set.TL.Center, set.BL.Center)

	msTLTop := estimateModuleSizeAlongRay(bm, set.TL.Center, set.TR.Center, set.TL.ModuleSize)
	msTRTop := estimateModuleSizeAlongRay(bm, set.TR.Center, set.TL.Center, set.TR.ModuleSize)
	msTop := 0.5 * (msTLTop + msTRTop)

	msTLLeft := estimateModuleSizeAlongRay(bm, set.TL.Center, set.BL.Center, set.TL.ModuleSize)
	msBLLeft := estimateModuleSizeAlongRay(bm, set.BL.Center, set.TL.Center, set.BL.ModuleSize)
	msLeft := 0.5 * (msTLLeft + msBLLeft)

	if msTop <= 0 || msLeft <= 0 {
		return nil
	}

	rawTop := dTop/msTop + 7.0
	rawLeft := dLeft/msLeft + 7.0
	rawAvg := (rawTop + rawLeft) * 0.5

	snapDim := func(raw float64) int {
		v := int(math.Round((raw - 17.0) / 4.0))
		if v < 1 {
			v = 1
		} else if v > 40 {
			v = 40
		}
		return 17 + 4*v
	}

	dimAvg := snapDim(rawAvg)
	dimTop := snapDim(rawTop)
	dimLeft := snapDim(rawLeft)

	seen := make(map[int]bool, 8)
	var dims []int
	addDim := func(d int) {
		if d >= 21 && d <= 177 && (d-21)%4 == 0 && !seen[d] {
			seen[d] = true
			dims = append(dims, d)
		}
	}

	addDim(dimAvg)
	if dimTop != dimLeft {
		addDim(dimTop)
		addDim(dimLeft)
		addDim(dimAvg - 4)
		addDim(dimAvg + 4)
		if math.Abs(float64(dimTop-dimLeft)) >= 4 {
			addDim(dimAvg - 8)
			addDim(dimAvg + 8)
		}
	} else {
		// Also include ±4 if rawAvg is close to a midpoint between versions.
		mod := math.Mod(rawAvg-21.0, 4.0)
		if mod > 1.0 && mod < 3.0 {
			addDim(dimAvg - 4)
			addDim(dimAvg + 4)
		}
	}
	return dims
}

func readVersionBits(gray *imageproc.GrayImage, bm *imageproc.BitMatrix, integ *imageproc.IntegralImage, dim int, h geom.PerspectiveTransform) (int, bool) {
	if dim < 45 {
		return 0, false
	}
	var bits1, bits2 uint32
	// Top-right 6x3 block: rows 0..5, cols dim-11..dim-9
	for r := 0; r < 6; r++ {
		for c := 0; c < 3; c++ {
			pt := h.Project(float64(dim-11+c)+0.5, float64(r)+0.5)
			bit, _ := sampleModule(gray, bm, integ, pt)
			if bit {
				bits1 |= 1 << (r*3 + c)
			}
		}
	}
	// Bottom-left 3x6 block: cols 0..5, rows dim-11..dim-9
	for c := 0; c < 6; c++ {
		for r := 0; r < 3; r++ {
			pt := h.Project(float64(c)+0.5, float64(dim-11+r)+0.5)
			bit, _ := sampleModule(gray, bm, integ, pt)
			if bit {
				bits2 |= 1 << (c*3 + r)
			}
		}
	}
	return decoder.DecodeVersionInformation(bits1, bits2)
}

// finderCornerTransform maps a finder pattern's 4 outer 7x7 corners into [0..7]x[0..7]
// local coordinates oriented along +uVec (TL->TR) and +vVec (TL->BL).
func finderCornerTransform(p ConcentricPattern, uVec, vVec geom.PointF) (geom.PerspectiveTransform, bool) {
	if !p.HasCorners {
		return geom.PerspectiveTransform{}, false
	}
	uDir := uVec.Normalize()
	vDir := vVec.Normalize()
	if uDir.LenSq() == 0 || vDir.LenSq() == 0 {
		return geom.PerspectiveTransform{}, false
	}
	var tlC, trC, brC, blC geom.PointF
	bestTL, bestTR, bestBR, bestBL := -1e18, -1e18, -1e18, -1e18
	for _, c := range p.Corners {
		d := c.Sub(p.Center)
		du := d.X*uDir.X + d.Y*uDir.Y
		dv := d.X*vDir.X + d.Y*vDir.Y
		if s := -du - dv; s > bestTL {
			bestTL, tlC = s, c
		}
		if s := du - dv; s > bestTR {
			bestTR, trC = s, c
		}
		if s := du + dv; s > bestBR {
			bestBR, brC = s, c
		}
		if s := -du + dv; s > bestBL {
			bestBL, blC = s, c
		}
	}
	if tlC == trC || trC == brC || brC == blC || blC == tlC {
		return geom.PerspectiveTransform{}, false
	}
	return geom.GeneralTransform(
		geom.Quad{{X: 0, Y: 0}, {X: 7, Y: 0}, {X: 7, Y: 7}, {X: 0, Y: 7}},
		geom.Quad{tlC, trC, brC, blC},
	)
}

// readVersionBitsFromFinders reads the 6x3 version block left of TR and 3x6 version
// block above BL using local finder geometry, independent of the global dimension estimate.
func readVersionBitsFromFinders(gray *imageproc.GrayImage, bm *imageproc.BitMatrix, integ *imageproc.IntegralImage, set FinderPatternSet) (int, bool) {
	uVec := set.TR.Center.Sub(set.TL.Center)
	vVec := set.BL.Center.Sub(set.TL.Center)
	uDir := uVec.Normalize()
	vDir := vVec.Normalize()
	if uDir.LenSq() == 0 || vDir.LenSq() == 0 {
		return 0, false
	}

	// Prefer perspective-exact 7x7 finder corner transforms when available.
	hTR, okTR := finderCornerTransform(set.TR, uVec, vVec)
	hBL, okBL := finderCornerTransform(set.BL, uVec, vVec)
	if okTR || okBL {
		var c1, c2 uint32
		if okTR {
			for r := 0; r < 6; r++ {
				for c := 0; c < 3; c++ {
					pt := hTR.Project(float64(c-4)+0.5, float64(r)+0.5)
					if b, ok := sampleBitFast(gray, bm, integ, pt); ok && b {
						c1 |= 1 << (r*3 + c)
					}
				}
			}
		}
		if okBL {
			for c := 0; c < 6; c++ {
				for r := 0; r < 3; r++ {
					pt := hBL.Project(float64(c)+0.5, float64(r-4)+0.5)
					if b, ok := sampleBitFast(gray, bm, integ, pt); ok && b {
						c2 |= 1 << (c*3 + r)
					}
				}
			}
		}
		if ver, ok := decoder.DecodeVersionInformation(c1, c2); ok && ver >= 7 {
			return ver, true
		}
	}

	msTRu := estimateModuleSizeAlongRay(bm, set.TR.Center, set.TL.Center, set.TR.ModuleSize)
	msTRv := estimateModuleSizeAlongRay(bm, set.TR.Center, set.TR.Center.Add(vDir.Mul(20)), msTRu)
	msBLu := estimateModuleSizeAlongRay(bm, set.BL.Center, set.BL.Center.Add(uDir.Mul(20)), set.BL.ModuleSize)
	msBLv := estimateModuleSizeAlongRay(bm, set.BL.Center, set.TL.Center, msBLu)

	var bits1, bits2 uint32

	sampleTR := func(duFromTRCenter, dvFromTRCenter float64) bool {
		pt := set.TR.Center.Add(uDir.Mul(duFromTRCenter * msTRu)).Add(vDir.Mul(dvFromTRCenter * msTRv))
		bit, _ := sampleBitFast(gray, bm, integ, pt)
		return bit
	}
	sampleBL := func(duFromBLCenter, dvFromBLCenter float64) bool {
		pt := set.BL.Center.Add(uDir.Mul(duFromBLCenter * msBLu)).Add(vDir.Mul(dvFromBLCenter * msBLv))
		bit, _ := sampleBitFast(gray, bm, integ, pt)
		return bit
	}

	for r := 0; r < 6; r++ {
		for c := 0; c < 3; c++ {
			du := float64(c) - 7.0
			dv := float64(r) - 3.0
			if sampleTR(du, dv) {
				bits1 |= 1 << (r*3 + c)
			}
		}
	}
	for c := 0; c < 6; c++ {
		for r := 0; r < 3; r++ {
			du := float64(c) - 3.0
			dv := float64(r) - 7.0
			if sampleBL(du, dv) {
				bits2 |= 1 << (c*3 + r)
			}
		}
	}

	return decoder.DecodeVersionInformation(bits1, bits2)
}

func makeFinderTransform(dim int, tl, tr, br, bl geom.PointF, brIsOuterCorner bool) (geom.PerspectiveTransform, bool) {
	d := float64(dim)
	brCoord := geom.PointF{X: d - 3.5, Y: d - 3.5}
	if brIsOuterCorner {
		brCoord = geom.PointF{X: d, Y: d}
	}
	src := geom.Quad{
		{X: 3.5, Y: 3.5},
		{X: d - 3.5, Y: 3.5},
		brCoord,
		{X: 3.5, Y: d - 3.5},
	}
	dst := geom.Quad{tl, tr, br, bl}
	return geom.GeneralTransform(src, dst)
}

func makeAlignTransform(dim int, tl, tr, brAlign, bl geom.PointF) (geom.PerspectiveTransform, bool) {
	d := float64(dim)
	src := geom.Quad{
		{X: 3.5, Y: 3.5},
		{X: d - 3.5, Y: 3.5},
		{X: d - 6.5, Y: d - 6.5},
		{X: 3.5, Y: d - 3.5},
	}
	dst := geom.Quad{tl, tr, brAlign, bl}
	return geom.GeneralTransform(src, dst)
}

type planarContext struct {
	dim          int
	baseH        geom.PerspectiveTransform
	brAlign      geom.PointF
	brAlignFound bool
	plausible    bool
}

// SamplePlanar implements Tier 1 planar homography sampling and decoding.
// It traces finder edges for BR intersection, locates the BR alignment pattern
// (for dim > 21), verifies Golay version bits (for dim >= 45), and falls back
// to extrapolated 4th corner TR - TL + BL.
func SamplePlanar(
	gray *imageproc.GrayImage,
	bm *imageproc.BitMatrix,
	integ *imageproc.IntegralImage,
	set FinderPatternSet,
) (*decoder.DecodeResult, geom.Quad, planarContext, bool) {
	dims := estimateCandidateDimensions(bm, set)
	if len(dims) == 0 {
		return nil, geom.Quad{}, planarContext{}, false
	}

	avgMS := (set.TL.ModuleSize + set.TR.ModuleSize + set.BL.ModuleSize) / 3.0
	estBR := set.TR.Center.Sub(set.TL.Center).Add(set.BL.Center)
	legLen := math.Max(dist(set.TL.Center, set.TR.Center), dist(set.TL.Center, set.BL.Center))

	uDir := set.TR.Center.Sub(set.TL.Center).Normalize()
	vDir := set.BL.Center.Sub(set.TL.Center).Normalize()

	// Trace outer QR edges (u = dim, v = dim) using multi-ray perpendicular edge fitting.
	lineRight := TraceFinderEdge(bm, set.TR.Center, vDir, uDir, set.TR.ModuleSize, 3)
	lineBottom := TraceFinderEdge(bm, set.BL.Center, uDir, vDir, set.BL.ModuleSize, 3)
	brInter, hasInter := geom.IntersectLines(lineRight, lineBottom)
	if hasInter {
		if brInter.Dist(estBR) > legLen*0.75 ||
			(brInter.X-set.TL.Center.X)*(estBR.X-set.TL.Center.X)+(brInter.Y-set.TL.Center.Y)*(estBR.Y-set.TL.Center.Y) <= 0 {
			hasInter = false
		}
	}

	// Trace inner finder edges (edges 2 and 3 toward TL) whose midpoint lines are
	// u = dim - 6.5 and v = dim - 6.5, intersecting directly at the BR alignment pattern.
	trIn2 := TraceFinderEdge(bm, set.TR.Center, vDir, uDir.Neg(), set.TR.ModuleSize, 2)
	trIn3 := TraceFinderEdge(bm, set.TR.Center, vDir, uDir.Neg(), set.TR.ModuleSize, 3)
	blIn2 := TraceFinderEdge(bm, set.BL.Center, uDir, vDir.Neg(), set.BL.ModuleSize, 2)
	blIn3 := TraceFinderEdge(bm, set.BL.Center, uDir, vDir.Neg(), set.BL.ModuleSize, 3)

	var brAlignInter geom.PointF
	hasAlignInter := false
	if p2, ok2 := geom.IntersectLines(trIn2, blIn2); ok2 {
		if p3, ok3 := geom.IntersectLines(trIn3, blIn3); ok3 {
			cand := p2.Midpoint(p3)
			if cand.Dist(estBR) <= legLen*0.75 &&
				(cand.X-set.TL.Center.X)*(estBR.X-set.TL.Center.X)+(cand.Y-set.TL.Center.Y)*(estBR.Y-set.TL.Center.Y) > 0 {
				brAlignInter = cand
				hasAlignInter = true
			}
		}
	}

	// Compute outer-corner intersection from local 7x7 finder corners (exact under planar perspective).
	var tlOut, trOut, brCornerOut, blOut geom.PointF
	hasCornerOut := false
	if set.TL.HasCorners && set.TR.HasCorners && set.BL.HasCorners {
		uVec := set.TR.Center.Sub(set.TL.Center)
		vVec := set.BL.Center.Sub(set.TL.Center)
		hTL, okTL := finderCornerTransform(set.TL, uVec, vVec)
		hTR, okTR := finderCornerTransform(set.TR, uVec, vVec)
		hBL, okBL := finderCornerTransform(set.BL, uVec, vVec)
		if okTL && okTR && okBL {
			trR0, trR1 := hTR.Project(7, 0), hTR.Project(7, 7)
			blB0, blB1 := hBL.Project(0, 7), hBL.Project(7, 7)
			a1, b1 := trR1.Y-trR0.Y, trR0.X-trR1.X
			c1 := a1*trR0.X + b1*trR0.Y
			a2, b2 := blB1.Y-blB0.Y, blB0.X-blB1.X
			c2 := a2*blB0.X + b2*blB0.Y
			det := a1*b2 - a2*b1
			if math.Abs(det) > 1e-6 {
				cand := geom.PointF{X: (b2*c1 - b1*c2) / det, Y: (a1*c2 - a2*c1) / det}
				toEst := estBR.Sub(set.TL.Center)
				toCand := cand.Sub(set.TL.Center)
				if cand.Dist(estBR) <= legLen*1.2 && (toCand.X*toEst.X+toCand.Y*toEst.Y) > 0 {
					tlOut = hTL.Project(0, 0)
					trOut = hTR.Project(7, 0)
					blOut = hBL.Project(0, 7)
					brCornerOut = cand
					hasCornerOut = true
				}
			}
		}
	}

	var bestCtx planarContext
	testedDims := make(map[int]bool, 10)
	fromFinderVersionBits := false
	dimLocked := false

	markPlausible := func(st gridValidateStats) {
		if st.valid || (st.coreHits >= 2 && (st.rowRatio >= 0.32 || st.colRatio >= 0.32 || (st.fmtOK && st.fmtHD <= 2))) {
			bestCtx.plausible = true
		}
	}
	shouldLockDim := func(hasAlign bool, st gridValidateStats) bool {
		_ = hasAlign
		return st.verMatched
	}

	// When estimated dimension is near or above Version 7 (dim >= 41), read the two
	// Golay(18,6) version info blocks directly relative to TR and BL finder patterns
	// so scale distortion at corners cannot cause a wrong dimension guess.
	if dims[0] >= 41 {
		if ver, okV := readVersionBitsFromFinders(gray, bm, integ, set); okV {
			verDim := 17 + 4*ver
			if verDim >= 45 && verDim <= 177 {
				fromFinderVersionBits = true
				if verDim != dims[0] {
					dims = []int{verDim, dims[0]}
				} else {
					dims = []int{verDim}
				}
			}
		}
	}

	tryDim := func(dim int) (*decoder.DecodeResult, geom.Quad, bool) {
		if testedDims[dim] || dim < 21 || dim > 177 || (dim-21)%4 != 0 {
			return nil, geom.Quad{}, false
		}
		testedDims[dim] = true

		hExtrap, okExtrap := makeFinderTransform(dim, set.TL.Center, set.TR.Center, estBR, set.BL.Center, false)
		hInter := hExtrap
		okInter := false
		if hasInter {
			hInter, okInter = makeFinderTransform(dim, set.TL.Center, set.TR.Center, brInter, set.BL.Center, true)
		}
		var hCornerOut geom.PerspectiveTransform
		okCornerH := false
		if hasCornerOut {
			d := float64(dim)
			hCornerOut, okCornerH = geom.GeneralTransform(
				geom.Quad{{0, 0}, {d, 0}, {d, d}, {0, d}},
				geom.Quad{tlOut, trOut, brCornerOut, blOut},
			)
		}

		initH := hExtrap
		okInit := okExtrap
		hasPerspectiveBR := false
		if okCornerH {
			initH = hCornerOut
			okInit = true
			hasPerspectiveBR = true
		} else if okInter {
			initH = hInter
			okInit = true
			hasPerspectiveBR = true
		} else if hasAlignInter && dim > 21 {
			if hA, okA := makeAlignTransform(dim, set.TL.Center, set.TR.Center, brAlignInter, set.BL.Center); okA {
				initH = hA
				okInit = true
				hasPerspectiveBR = true
			}
		}
		if !okInit {
			return nil, geom.Quad{}, false
		}

		// Record primary context for fallback tiers.
		if bestCtx.dim == 0 {
			bestCtx = planarContext{dim: dim, baseH: initH}
		}

		// Version info verification for dim >= 45 (Version >= 7).
		// Only override dim if we did not already read Golay version info from local finder corners,
		// and only trust global homography version bits when we have a perspective BR estimate.
		if dim >= 45 && !fromFinderVersionBits && hasPerspectiveBR {
			if ver, okV := readVersionBits(gray, bm, integ, dim, initH); okV {
				verDim := 17 + 4*ver
				if verDim != dim && !testedDims[verDim] {
					if okCornerH {
						dV := float64(verDim)
						if hV, ok := geom.GeneralTransform(geom.Quad{{0, 0}, {dV, 0}, {dV, dV}, {0, dV}}, geom.Quad{tlOut, trOut, brCornerOut, blOut}); ok {
							dim = verDim
							initH = hV
							hCornerOut = hV
							testedDims[dim] = true
							bestCtx.dim = dim
							bestCtx.baseH = initH
							bestCtx.plausible = true
							dimLocked = true
						}
					} else if okInter {
						if hV, ok := makeFinderTransform(verDim, set.TL.Center, set.TR.Center, brInter, set.BL.Center, true); ok {
							dim = verDim
							initH = hV
							testedDims[dim] = true
							bestCtx.dim = dim
							bestCtx.baseH = initH
							bestCtx.plausible = true
							dimLocked = true
						}
					}
				} else if verDim == dim {
					bestCtx.plausible = true
					dimLocked = true
				}
			}
		}

		var initStats gridValidateStats
		hasInitStats := false
		skipInit := false

		// Locate bottom-right alignment pattern for version >= 2 (dim > 21).
		if dim > 21 {
			initStats = quickValidateGridWithStats(gray, bm, integ, dim, initH.Project)
			hasInitStats = true
			markPlausible(initStats)
			if initStats.verMatched {
				dimLocked = true
			}
			if initStats.coreHits == 0 && !hasAlignInter {
				return nil, geom.Quad{}, false
			}
			if dim > 45 && !fromFinderVersionBits && !hasAlignInter && initStats.coreHits < 3 && initStats.rowRatio < 0.25 && initStats.colRatio < 0.25 {
				return nil, geom.Quad{}, false
			}

			maxMS := math.Max(set.TL.ModuleSize, math.Max(set.TR.ModuleSize, set.BL.ModuleSize))
			minMS := math.Min(set.TL.ModuleSize, math.Min(set.TR.ModuleSize, set.BL.ModuleSize))
			brMS := avgMS
			if maxMS > minMS*1.20 {
				estBRMS := set.TR.ModuleSize + set.BL.ModuleSize - set.TL.ModuleSize
				if estBRMS < minMS*0.7 {
					estBRMS = minMS * 0.7
				} else if estBRMS > maxMS*1.5 {
					estBRMS = maxMS * 1.5
				}
				brMS = estBRMS
			}
			searchR := math.Max(8.0, avgMS*5.5)
			if maxMS > minMS*1.25 {
				searchR = math.Max(searchR, maxMS*6.5)
			}
			estAlign := initH.Project(float64(dim)-6.5, float64(dim)-6.5)
			var brCandidates [3]geom.PointF
			nCand := 0
			addCand := func(pt geom.PointF) {
				for i := 0; i < nCand; i++ {
					if pt.Dist(brCandidates[i]) < searchR*0.4 {
						return
					}
				}
				if nCand < len(brCandidates) {
					brCandidates[nCand] = pt
					nCand++
				}
			}
			// Prefer exact 7x7 corner estimate first when available, then TraceFinderEdge intersections.
			if okCornerH {
				addCand(estAlign)
			}
			if hasAlignInter {
				addCand(brAlignInter)
			}
			if !okCornerH {
				addCand(estAlign)
			}
			if okInter && okCornerH {
				addCand(hInter.Project(float64(dim)-6.5, float64(dim)-6.5))
			}

			var brAlign geom.PointF
			okA := false
			for ci := 0; ci < nCand; ci++ {
				candAlign, okCand := LocateAlignmentPattern(bm, gray, brMS, brCandidates[ci], searchR)
				if !okCand && math.Abs(brMS-avgMS) > avgMS*0.12 {
					candAlign, okCand = LocateAlignmentPattern(bm, gray, avgMS, brCandidates[ci], searchR)
				}
				if !okCand {
					continue
				}
				if !okA {
					brAlign, okA = candAlign, true
				}
				if hAlign, okH := makeAlignTransform(dim, set.TL.Center, set.TR.Center, candAlign, set.BL.Center); okH {
					wasPlausible := bestCtx.plausible
					if ci == 0 || !bestCtx.brAlignFound {
						bestCtx = planarContext{dim: dim, baseH: hAlign, brAlign: candAlign, brAlignFound: true, plausible: wasPlausible}
					}
					res, quad, stAlign, okD := sampleAndDecodeWithStats(gray, bm, integ, dim, hAlign.Project)
					if okD {
						return res, quad, true
					}
					markPlausible(stAlign)
					if wasPlausible {
						bestCtx.plausible = true
					}
					if shouldLockDim(true, stAlign) {
						dimLocked = true
					}
					if ci == 0 && candAlign.Dist(estAlign) < 0.15 {
						skipInit = true
					}
				}
			}
			if hasAlignInter && (!okA || brAlign.Dist(brAlignInter) > 0.75) {
				if hAlignInt, okH := makeAlignTransform(dim, set.TL.Center, set.TR.Center, brAlignInter, set.BL.Center); okH {
					res, quad, stAI, okD := sampleAndDecodeWithStats(gray, bm, integ, dim, hAlignInt.Project)
					if okD {
						return res, quad, true
					}
					markPlausible(stAI)
					if shouldLockDim(true, stAI) {
						dimLocked = true
					}
				}
			}
		}

		if !skipInit {
			if hasInitStats {
				if initStats.valid {
					if res, quad, okD := sampleAndDecodeValidated(gray, bm, integ, dim, initH.Project); okD {
						return res, quad, true
					}
				}
			} else {
				res, quad, stInit, okD := sampleAndDecodeWithStats(gray, bm, integ, dim, initH.Project)
				if okD {
					return res, quad, true
				}
				markPlausible(stInit)
				if stInit.fmtOK && stInit.fmtHD <= 1 && stInit.rowRatio >= 0.85 && stInit.colRatio >= 0.85 {
					dimLocked = true
				}
			}
		}

		if okCornerH && okInter && brCornerOut.Dist(brInter) >= 0.35 {
			res, quad, stInt, okD := sampleAndDecodeWithStats(gray, bm, integ, dim, hInter.Project)
			if okD {
				return res, quad, true
			}
			markPlausible(stInt)
		}

		if (okCornerH || okInter) && okExtrap &&
			(!okCornerH || brCornerOut.Dist(estBR) >= 0.35) &&
			(!okInter || brInter.Dist(estBR) >= 0.35) {
			res, quad, stExt, okD := sampleAndDecodeWithStats(gray, bm, integ, dim, hExtrap.Project)
			if okD {
				return res, quad, true
			}
			markPlausible(stExt)
		}
		return nil, geom.Quad{}, false
	}

	for i, dim := range dims {
		if i >= 2 && !bestCtx.plausible && set.Score < 0.80 {
			break
		}
		if res, quad, ok := tryDim(dim); ok {
			return res, quad, bestCtx, true
		}
		if dimLocked {
			break
		}
	}

	return nil, geom.Quad{}, bestCtx, false
}

// SampleTiled implements Tier 2 piecewise-homography sampling via geom.BilinearTileGrid
// using the NxN alignment pattern centers for dim >= 25.
func SampleTiled(
	gray *imageproc.GrayImage,
	bm *imageproc.BitMatrix,
	integ *imageproc.IntegralImage,
	set FinderPatternSet,
	dim int,
	baseH geom.PerspectiveTransform,
) (*decoder.DecodeResult, geom.Quad, bool) {
	return sampleTiledWithAlign(gray, bm, integ, set, dim, baseH, geom.PointF{}, false)
}

func sampleTiledWithAlign(
	gray *imageproc.GrayImage,
	bm *imageproc.BitMatrix,
	integ *imageproc.IntegralImage,
	set FinderPatternSet,
	dim int,
	baseH geom.PerspectiveTransform,
	brAlign geom.PointF,
	brAlignFound bool,
) (*decoder.DecodeResult, geom.Quad, bool) {
	if dim < 25 || !baseH.IsValid() {
		return nil, geom.Quad{}, false
	}
	version := (dim - 17) / 4
	centers := decoder.AlignmentPatternCenters(version)
	m := len(centers)
	if m < 2 || (m == 2 && brAlignFound) {
		return nil, geom.Quad{}, false
	}

	avgMS := (set.TL.ModuleSize + set.TR.ModuleSize + set.BL.ModuleSize) / 3.0
	searchR := math.Max(4.5, avgMS*4.2)

	// Shift fallback homography so (u, v) aligns with module index coordinates centers[k]
	// where module center is (u+0.5, v+0.5).
	d := float64(dim)
	shiftedFallback, okShift := geom.GeneralTransform(
		geom.Quad{{X: 0, Y: 0}, {X: d - 1, Y: 0}, {X: d - 1, Y: d - 1}, {X: 0, Y: d - 1}},
		geom.Quad{baseH.Project(0.5, 0.5), baseH.Project(d-0.5, 0.5), baseH.Project(d-0.5, d-0.5), baseH.Project(0.5, d-0.5)},
	)
	if !okShift {
		shiftedFallback = baseH
	}

	apP := make([][]geom.PointF, m)
	valid := make([][]bool, m)
	for r := 0; r < m; r++ {
		apP[r] = make([]geom.PointF, m)
		valid[r] = make([]bool, m)
	}

	isFinderCorner := func(r, c int) bool {
		return (r == 0 && c == 0) || (r == 0 && c == m-1) || (r == m-1 && c == 0)
	}

	// Anchor finder-occupied corner alignment nodes using local 7x7 finder corners when valid.
	uVec := set.TR.Center.Sub(set.TL.Center)
	vVec := set.BL.Center.Sub(set.TL.Center)
	apP[0][0] = baseH.Project(float64(centers[0])+0.5, float64(centers[0])+0.5)
	if hTL, ok := finderCornerTransform(set.TL, uVec, vVec); ok {
		pt := hTL.Project(6.5, 6.5)
		if pt.Dist(apP[0][0]) <= avgMS*3.0 {
			apP[0][0] = pt
		}
	}
	apP[0][m-1] = baseH.Project(float64(centers[m-1])+0.5, float64(centers[0])+0.5)
	if hTR, ok := finderCornerTransform(set.TR, uVec, vVec); ok {
		pt := hTR.Project(0.5, 6.5)
		if pt.Dist(apP[0][m-1]) <= avgMS*3.0 {
			apP[0][m-1] = pt
		}
	}
	apP[m-1][0] = baseH.Project(float64(centers[0])+0.5, float64(centers[m-1])+0.5)
	if hBL, ok := finderCornerTransform(set.BL, uVec, vVec); ok {
		pt := hBL.Project(6.5, 0.5)
		if pt.Dist(apP[m-1][0]) <= avgMS*3.0 {
			apP[m-1][0] = pt
		}
	}
	valid[0][0] = true
	valid[0][m-1] = true
	valid[m-1][0] = true

	foundCount := 0
	searchHits := 0
	if brAlignFound {
		apP[m-1][m-1] = brAlign
		valid[m-1][m-1] = true
		foundCount++
	}
	for r := 0; r < m; r++ {
		for c := 0; c < m; c++ {
			if isFinderCorner(r, c) || valid[r][c] {
				continue
			}
			// Progressive estimate from already-found neighbors (row/col extrapolation + parallelogram).
			est := baseH.Project(float64(centers[c])+0.5, float64(centers[r])+0.5)
			localR := searchR
			if r > 0 && c > 0 && valid[r-1][c] && valid[r][c-1] && valid[r-1][c-1] {
				est = apP[r-1][c].Add(apP[r][c-1]).Sub(apP[r-1][c-1])
			} else if r == 0 && c >= 2 && valid[0][c-1] && valid[0][c-2] {
				ratio := float64(centers[c]-centers[c-1]) / float64(centers[c-1]-centers[c-2])
				est = apP[0][c-1].Add(apP[0][c-1].Sub(apP[0][c-2]).Mul(ratio))
			} else if c == 0 && r >= 2 && valid[r-1][0] && valid[r-2][0] {
				ratio := float64(centers[r]-centers[r-1]) / float64(centers[r-1]-centers[r-2])
				est = apP[r-1][0].Add(apP[r-1][0].Sub(apP[r-2][0]).Mul(ratio))
			} else if r == 0 || c == 0 {
				localR = math.Max(5.5, avgMS*5.5)
			}
			pt, ok := LocateAlignmentPattern(bm, gray, avgMS, est, localR)
			if !ok && r == 0 && centers[m-1] > centers[0] {
				t := float64(centers[c]-centers[0]) / float64(centers[m-1]-centers[0])
				lin := apP[0][0].Mul(1 - t).Add(apP[0][m-1].Mul(t))
				if lin.Dist(est) > localR*0.35 {
					pt, ok = LocateAlignmentPattern(bm, gray, avgMS, lin, localR)
				}
			} else if !ok && c == 0 && centers[m-1] > centers[0] {
				t := float64(centers[r]-centers[0]) / float64(centers[m-1]-centers[0])
				lin := apP[0][0].Mul(1 - t).Add(apP[m-1][0].Mul(t))
				if lin.Dist(est) > localR*0.35 {
					pt, ok = LocateAlignmentPattern(bm, gray, avgMS, lin, localR)
				}
			}
			if ok {
				apP[r][c] = pt
				valid[r][c] = true
				foundCount++
				searchHits++
			}
		}
		if m >= 3 && r == 1 && searchHits == 0 {
			return nil, geom.Quad{}, false
		}
	}

	if foundCount == 0 {
		return nil, geom.Quad{}, false
	}

	// Fill missing interior/corner alignment nodes from valid neighbor lines (2 passes).
	for pass := 0; pass < 2; pass++ {
		for r := 0; r < m; r++ {
			for c := 0; c < m; c++ {
				if valid[r][c] {
					continue
				}
				if pt, ok := interpolateMissingNode(centers, apP, valid, r, c); ok {
					apP[r][c] = pt
					valid[r][c] = true
				}
			}
		}
	}

	// Precompute (m-1)x(m-1) tile homographies once instead of calling SquareToQuadrilateral per pixel.
	grid := geom.NewBilinearTileGrid(centers, apP, valid, shiftedFallback)
	tileH := make([][]geom.PerspectiveTransform, m-1)
	tileOK := make([][]bool, m-1)
	for r := 0; r < m-1; r++ {
		tileH[r] = make([]geom.PerspectiveTransform, m-1)
		tileOK[r] = make([]bool, m-1)
		for c := 0; c < m-1; c++ {
			if valid[r][c] && valid[r][c+1] && valid[r+1][c+1] && valid[r+1][c] {
				h := geom.SquareToQuadrilateral(apP[r][c], apP[r][c+1], apP[r+1][c+1], apP[r+1][c])
				if h.IsValid() {
					tileH[r][c] = h
					tileOK[r][c] = true
				}
			}
		}
	}

	fastProj := func(u, v float64) geom.PointF {
		um, vm := u-0.5, v-0.5
		ix := 0
		for ix+1 < m-1 && um >= float64(centers[ix+1]) {
			ix++
		}
		iy := 0
		for iy+1 < m-1 && vm >= float64(centers[iy+1]) {
			iy++
		}
		if tileOK[iy][ix] {
			s := (um - float64(centers[ix])) / float64(centers[ix+1]-centers[ix])
			t := (vm - float64(centers[iy])) / float64(centers[iy+1]-centers[iy])
			h := &tileH[iy][ix]
			if math.Abs(h.H[6]*s+h.H[7]*t+h.H[8]) > 1e-6 {
				return h.Project(s, t)
			}
		}
		return grid.Project(um, vm)
	}

	return sampleAndDecode(gray, bm, integ, dim, fastProj)
}

func interpolateMissingNode(centers []int, apP [][]geom.PointF, valid [][]bool, r, c int) (geom.PointF, bool) {
	m := len(centers)
	// Try horizontal neighbor pair along row r.
	var prevC, nextC = -1, -1
	for cc := c - 1; cc >= 0; cc-- {
		if valid[r][cc] {
			prevC = cc
			break
		}
	}
	for cc := c + 1; cc < m; cc++ {
		if valid[r][cc] {
			nextC = cc
			break
		}
	}
	if prevC >= 0 && nextC >= 0 {
		t := float64(centers[c]-centers[prevC]) / float64(centers[nextC]-centers[prevC])
		return apP[r][prevC].Mul(1 - t).Add(apP[r][nextC].Mul(t)), true
	}

	// Try vertical neighbor pair along col c.
	var prevR, nextR = -1, -1
	for rr := r - 1; rr >= 0; rr-- {
		if valid[rr][c] {
			prevR = rr
			break
		}
	}
	for rr := r + 1; rr < m; rr++ {
		if valid[rr][c] {
			nextR = rr
			break
		}
	}
	if prevR >= 0 && nextR >= 0 {
		t := float64(centers[r]-centers[prevR]) / float64(centers[nextR]-centers[prevR])
		return apP[prevR][c].Mul(1 - t).Add(apP[nextR][c].Mul(t)), true
	}

	// Try parallelogram completion from any valid 3 corners of a neighbor cell.
	if r > 0 && c > 0 && valid[r-1][c] && valid[r][c-1] && valid[r-1][c-1] {
		return apP[r-1][c].Add(apP[r][c-1]).Sub(apP[r-1][c-1]), true
	}
	return geom.PointF{}, false
}

// SampleCurvedCoons implements Tier 3 Coons-patch curved boundary sampling.
// It traces the 4 outer edges at intermediate stations via perpendicular scanning
// to measure curvature sagitta and samples through geom.CoonsPatchGrid.
func SampleCurvedCoons(
	gray *imageproc.GrayImage,
	bm *imageproc.BitMatrix,
	integ *imageproc.IntegralImage,
	set FinderPatternSet,
	dim int,
	baseH geom.PerspectiveTransform,
) (*decoder.DecodeResult, geom.Quad, bool) {
	if dim < 21 || bm == nil {
		return nil, geom.Quad{}, false
	}

	estBR := set.TR.Center.Sub(set.TL.Center).Add(set.BL.Center)
	hFinder, okFinder := makeFinderTransform(dim, set.TL.Center, set.TR.Center, estBR, set.BL.Center, false)

	var refs []geom.PerspectiveTransform
	if okFinder {
		refs = append(refs, hFinder)
	}
	if baseH.IsValid() {
		if !okFinder || hFinder.Project(float64(dim), float64(dim)).Dist(baseH.Project(float64(dim), float64(dim))) >= 2.5 {
			refs = append(refs, baseH)
		}
	}

	var bestFallbackRes *decoder.DecodeResult
	var bestFallbackQuad geom.Quad
	for _, refH := range refs {
		if res, quad, ok := sampleCurvedCoonsWithRef(gray, bm, integ, set, dim, refH); ok {
			if !res.Mirrored || res.BitErrors <= 10 {
				return res, quad, true
			}
			if bestFallbackRes == nil || res.BitErrors < bestFallbackRes.BitErrors {
				bestFallbackRes = res
				bestFallbackQuad = quad
			}
		}
	}
	if bestFallbackRes != nil {
		return bestFallbackRes, bestFallbackQuad, true
	}
	return nil, geom.Quad{}, false
}

func isFinderEdgePoint(u, v, d float64) bool {
	const eps = 0.5
	if v <= eps || v >= d-eps {
		if v <= eps {
			return u <= 7.0 || u >= d-7.0
		}
		return u <= 7.0
	}
	if u <= eps || u >= d-eps {
		if u <= eps {
			return v <= 7.0 || v >= d-7.0
		}
		return v <= 7.0
	}
	return false
}

func sampleCurvedCoonsWithRef(
	gray *imageproc.GrayImage,
	bm *imageproc.BitMatrix,
	integ *imageproc.IntegralImage,
	set FinderPatternSet,
	dim int,
	baseH geom.PerspectiveTransform,
) (*decoder.DecodeResult, geom.Quad, bool) {
	if !baseH.IsValid() {
		return nil, geom.Quad{}, false
	}

	d := float64(dim)
	avgMS := (set.TL.ModuleSize + set.TR.ModuleSize + set.BL.ModuleSize) / 3.0
	maxSag := math.Max(6.0, d*avgMS*0.18)

	// Trace an outer edge from (u0, v0) to (u1, v1) with outward module normal (nu, nv).
	traceOuterBoundary := func(refH geom.PerspectiveTransform, u0, v0, u1, v1, nu, nv float64, stations int) ([]geom.PointF, []bool) {
		poly := make([]geom.PointF, stations)
		found := make([]bool, stations)
		cStart := refH.Project(u0, v0)
		cEnd := refH.Project(u1, v1)
		poly[0] = cStart
		poly[stations-1] = cEnd

		// Outward normal in image space.
		outVec := refH.Project((u0+u1)*0.5+nu, (v0+v1)*0.5+nv).Sub(refH.Project((u0+u1)*0.5, (v0+v1)*0.5))
		outDir := outVec.Normalize()
		if outDir.LenSq() == 0 {
			for i := 1; i < stations-1; i++ {
				t := float64(i) / float64(stations-1)
				poly[i] = cStart.Mul(1 - t).Add(cEnd.Mul(t))
			}
			return poly, found
		}
		tanDir := geom.PointF{X: -outDir.Y, Y: outDir.X}

		offs := make([]float64, stations)
		for i := 1; i < stations-1; i++ {
			t := float64(i) / float64(stations-1)
			u := u0 + t*(u1-u0)
			v := v0 + t*(v1-v0)
			pNom := refH.Project(u, v)

			if isFinderEdgePoint(u, v, d) {
				// Finder boundaries have guaranteed solid black outer modules; avoid tangent smear.
				for _, dt := range []float64{0, -0.4 * avgMS, 0.4 * avgMS} {
					probe := pNom.Add(tanDir.Mul(dt))
					if off, ok := findOuterEdgeOffset(bm, probe, outDir, maxSag, avgMS); ok {
						offs[i] = off
						found[i] = true
						break
					}
				}
				continue
			}

			// On data edges, probe a compact tangent band to hit at least one black edge module.
			bestOffset := 0.0
			foundEdge := false
			for _, dt := range []float64{0, -avgMS, avgMS, -2 * avgMS, 2 * avgMS} {
				probe := pNom.Add(tanDir.Mul(dt))
				if off, ok := findOuterEdgeOffset(bm, probe, outDir, maxSag, avgMS); ok {
					if !foundEdge || off > bestOffset {
						bestOffset = off
						foundEdge = true
					}
				}
			}
			if foundEdge {
				offs[i] = bestOffset
				found[i] = true
			}
		}

		// Suppress inward divots caused by consecutive white modules on data edges.
		for pass := 0; pass < 2; pass++ {
			for i := 1; i < stations-1; i++ {
				if !found[i] {
					continue
				}
				if found[i-1] && found[i+1] {
					avgNbr := 0.5 * (offs[i-1] + offs[i+1])
					if offs[i] < avgNbr-0.65*avgMS {
						offs[i] = avgNbr
					}
				}
			}
		}
		// Guard the non-finder end station (i = stations-2) against an inward white-module drop.
		if stations >= 5 && found[stations-2] && found[stations-3] && found[stations-4] {
			uEnd := u0 + float64(stations-2)/float64(stations-1)*(u1-u0)
			vEnd := v0 + float64(stations-2)/float64(stations-1)*(v1-v0)
			if !isFinderEdgePoint(uEnd, vEnd, d) {
				pred := 2*offs[stations-3] - offs[stations-4]
				if offs[stations-2] < pred-0.65*avgMS {
					offs[stations-2] = 0.5 * (offs[stations-3] + pred)
				}
			}
		}

		for i := 1; i < stations-1; i++ {
			t := float64(i) / float64(stations-1)
			pNom := refH.Project(u0+t*(u1-u0), v0+t*(v1-v0))
			if found[i] {
				poly[i] = pNom.Add(outDir.Mul(offs[i]))
			} else {
				poly[i] = pNom
			}
		}
		return poly, found
	}

	const stations = 9
	// Refine the 4 corners by intersecting the unsmoothed boundary curve ends when observed.
	intersectCorner := func(a1, a2, b1, b2, nom geom.PointF, okA1, okA2, okB1, okB2 bool) geom.PointF {
		if !okA1 || !okA2 || !okB1 || !okB2 {
			return nom
		}
		var l1, l2 geom.RegressionLine
		l1.Add(a1)
		l1.Add(a2)
		l2.Add(b1)
		l2.Add(b2)
		if !l1.Evaluate(0, false) || !l2.Evaluate(0, false) {
			return nom
		}
		inter, ok := geom.IntersectLines(l1, l2)
		if !ok || inter.Dist(nom) > maxSag*1.2 {
			return nom
		}
		return inter
	}

	smoothPoly := func(poly []geom.PointF) []geom.PointF {
		out := make([]geom.PointF, len(poly))
		copy(out, poly)
		for i := 1; i < len(poly)-1; i++ {
			out[i] = poly[i-1].Mul(0.25).Add(poly[i].Mul(0.5)).Add(poly[i+1].Mul(0.25))
		}
		return out
	}

	top, topOK := traceOuterBoundary(baseH, 0, 0, d, 0, 0, -1, stations)
	bottom, botOK := traceOuterBoundary(baseH, 0, d, d, d, 0, 1, stations)
	left, leftOK := traceOuterBoundary(baseH, 0, 0, 0, d, -1, 0, stations)
	right, rightOK := traceOuterBoundary(baseH, d, 0, d, d, 1, 0, stations)

	foundStations := 0
	for i := 1; i < stations-1; i++ {
		if topOK[i] {
			foundStations++
		}
		if botOK[i] {
			foundStations++
		}
		if leftOK[i] {
			foundStations++
		}
		if rightOK[i] {
			foundStations++
		}
	}
	if foundStations < 10 {
		return nil, geom.Quad{}, false
	}

	c00 := intersectCorner(top[1], top[2], left[1], left[2], top[0], topOK[1], topOK[2], leftOK[1], leftOK[2])
	c10 := intersectCorner(top[stations-3], top[stations-2], right[1], right[2], top[stations-1], topOK[stations-3], topOK[stations-2], rightOK[1], rightOK[2])
	c11 := intersectCorner(bottom[stations-3], bottom[stations-2], right[stations-3], right[stations-2], bottom[stations-1], botOK[stations-3], botOK[stations-2], rightOK[stations-3], rightOK[stations-2])
	c01 := intersectCorner(bottom[1], bottom[2], left[stations-3], left[stations-2], bottom[0], botOK[1], botOK[2], leftOK[stations-3], leftOK[stations-2])

	cornerShift := c00.Dist(top[0]) + c10.Dist(top[stations-1]) + c11.Dist(bottom[stations-1]) + c01.Dist(bottom[0])
	if cornerShift >= 2.0 {
		// Re-trace using cornerH anchored at the refined corners so tangential station spacing
		// aligns uniformly from corner to corner without endpoint jumps.
		srcQ := geom.Quad{{X: 0, Y: 0}, {X: d, Y: 0}, {X: d, Y: d}, {X: 0, Y: d}}
		dstQ := geom.Quad{c00, c10, c11, c01}
		if cornerH, okH := geom.GeneralTransform(srcQ, dstQ); okH && cornerH.IsValid() {
			top2, _ := traceOuterBoundary(cornerH, 0, 0, d, 0, 0, -1, stations)
			bottom2, _ := traceOuterBoundary(cornerH, 0, d, d, d, 0, 1, stations)
			left2, _ := traceOuterBoundary(cornerH, 0, 0, 0, d, -1, 0, stations)
			right2, _ := traceOuterBoundary(cornerH, d, 0, d, d, 1, 0, stations)
			grid2 := geom.NewCoonsPatchGrid(dim, smoothPoly(top2), smoothPoly(bottom2), smoothPoly(left2), smoothPoly(right2))
			if res, quad, ok := sampleAndDecode(gray, bm, integ, dim, grid2.Project); ok {
				return res, quad, true
			}
		}
	}

	top[0], left[0] = c00, c00
	top[stations-1], right[0] = c10, c10
	bottom[stations-1], right[stations-1] = c11, c11
	bottom[0], left[stations-1] = c01, c01

	grid := geom.NewCoonsPatchGrid(dim, smoothPoly(top), smoothPoly(bottom), smoothPoly(left), smoothPoly(right))
	return sampleAndDecode(gray, bm, integ, dim, grid.Project)
}

// findOuterEdgeOffset scans along outDir from -maxSag to +maxSag around pNom
// and returns the offset of the outermost black->white quiet-zone transition.
func findOuterEdgeOffset(bm *imageproc.BitMatrix, pNom, outDir geom.PointF, maxSag, ms float64) (float64, bool) {
	step := math.Max(0.5, math.Min(1.0, ms*0.3))
	quietNeed := math.Max(2.0, ms*1.2)

	lastBlackOff := 0.0
	foundBlack := false
	whiteRun := 0.0

	// Scan from inside (-maxSag*0.6) outward (+maxSag).
	startOff := -maxSag * 0.6
	for off := startOff; off <= maxSag; off += step {
		pt := pNom.Add(outDir.Mul(off))
		ix := int(math.Floor(pt.X))
		iy := int(math.Floor(pt.Y))
		if !bm.IsIn(ix, iy) {
			break
		}
		if bm.Get(ix, iy) {
			lastBlackOff = off
			foundBlack = true
			whiteRun = 0
		} else if foundBlack {
			whiteRun += step
			if whiteRun >= quietNeed {
				return lastBlackOff + step*0.5, true
			}
		}
	}
	return 0, false
}

// SampleParametricDeform implements Tier 4 parametric lens/surface unwarping
// (radial distortion, cylindrical curvature, sinusoidal wave distortion, and BR corner nudge).
func SampleParametricDeform(
	gray *imageproc.GrayImage,
	bm *imageproc.BitMatrix,
	integ *imageproc.IntegralImage,
	set FinderPatternSet,
	dim int,
	baseH geom.PerspectiveTransform,
	brAlignFound bool,
) (*decoder.DecodeResult, geom.Quad, bool) {
	if dim < 21 || bm == nil {
		return nil, geom.Quad{}, false
	}

	avgMS := (set.TL.ModuleSize + set.TR.ModuleSize + set.BL.ModuleSize) / 3.0
	estBR := set.TR.Center.Sub(set.TL.Center).Add(set.BL.Center)
	if baseH.IsValid() {
		estBR = baseH.Project(float64(dim)-3.5, float64(dim)-3.5)
	}

	// 1) BR Corner Nudge when alignment pattern was not confirmed.
	if !brAlignFound {
		type offset struct{ dx, dy float64 }
		var offsets []offset
		for r := 1; r <= 2; r++ {
			for dy := -2; dy <= 2; dy++ {
				for dx := -2; dx <= 2; dx++ {
					if dx == 0 && dy == 0 {
						continue
					}
					if int(math.Abs(float64(dx))+math.Abs(float64(dy))) == r {
						offsets = append(offsets, offset{float64(dx) * avgMS, float64(dy) * avgMS})
					}
				}
			}
		}
		for _, off := range offsets {
			nudgedBR := geom.PointF{X: estBR.X + off.dx, Y: estBR.Y + off.dy}
			if hNudge, ok := makeFinderTransform(dim, set.TL.Center, set.TR.Center, nudgedBR, set.BL.Center, false); ok {
				if res, quad, okD := sampleAndDecode(gray, bm, integ, dim, hNudge.Project); okD {
					return res, quad, true
				}
			}
		}
	}

	// 2) Radial unwarping (barrel / pincushion lens distortion).
	k1Values := []float64{-0.45, -0.3, -0.15, 0.15, 0.3, 0.45, -0.6}
	imgCenter := geom.PointF{X: float64(bm.Width) * 0.5, Y: float64(bm.Height) * 0.5}
	imgNorm := math.Max(float64(bm.Width), float64(bm.Height)) * 0.5
	qrCenter := set.TL.Center.Add(set.TR.Center).Add(set.BL.Center).Add(estBR).Mul(0.25)
	qrNorm := math.Max(dist(set.TL.Center, set.TR.Center), dist(set.TL.Center, set.BL.Center)) * 0.75

	type radialOrigin struct {
		c    geom.PointF
		norm float64
	}
	origins := []radialOrigin{{imgCenter, imgNorm}}
	if qrCenter.Dist(imgCenter) > imgNorm*0.15 && qrNorm > 0 {
		origins = append(origins, radialOrigin{qrCenter, qrNorm})
	}

	for _, orig := range origins {
		if orig.norm <= 0 {
			continue
		}
		for _, k1 := range k1Values {
			rt := geom.NewRadialTransform(orig.c.X, orig.c.Y, orig.norm, orig.norm, k1, 0)
			uTL := rt.Inverse(set.TL.Center)
			uTR := rt.Inverse(set.TR.Center)
			uBL := rt.Inverse(set.BL.Center)
			uBR := uTR.Sub(uTL).Add(uBL)

			radDims := estimateUnwarpedDims(rt.Inverse, set, dim)
			for _, rDim := range radDims {
				if hUndist, ok := makeFinderTransform(rDim, uTL, uTR, uBR, uBL, false); ok {
					projUndist := func(u, v float64) geom.PointF {
						return rt.Forward(hUndist.Project(u, v))
					}
					if !quickValidateGrid(gray, bm, integ, rDim, projUndist) {
						continue
					}
					if rDim > 21 {
						estA := rt.Forward(hUndist.Project(float64(rDim)-6.5, float64(rDim)-6.5))
						if brA, okA := LocateAlignmentPattern(bm, gray, avgMS, estA, avgMS*4.0); okA {
							uAlign := rt.Inverse(brA)
							if hA, okH := makeAlignTransform(rDim, uTL, uTR, uAlign, uBL); okH {
								if res, quad, okD := sampleAndDecode(gray, bm, integ, rDim, func(u, v float64) geom.PointF {
									return rt.Forward(hA.Project(u, v))
								}); okD {
									return res, quad, true
								}
							}
						}
					}
					if res, quad, okD := sampleAndDecode(gray, bm, integ, rDim, projUndist); okD {
						return res, quad, true
					}
				}
			}
		}
	}

	// 3) Cylindrical unwarping (vertical and horizontal cylinder axes).
	radii := []float64{250, 300, 350, 400, 500, 700}
	localCX := (set.TL.Center.X + set.TR.Center.X + set.BL.Center.X + estBR.X) * 0.25
	localCY := (set.TL.Center.Y + set.TR.Center.Y + set.BL.Center.Y + estBR.Y) * 0.25

	type cylAxis struct {
		cx, cy, tiltX float64
	}
	axes := []cylAxis{
		{imgCenter.X, imgCenter.Y, 0},
		{imgCenter.X, imgCenter.Y, math.Pi * 0.5},
	}
	if math.Hypot(localCX-imgCenter.X, localCY-imgCenter.Y) > 35.0 {
		axes = append(axes,
			cylAxis{localCX, localCY, 0},
			cylAxis{localCX, localCY, math.Pi * 0.5},
		)
	}

	type cylTask struct {
		ax  cylAxis
		rad float64
	}
	cylTasks := make([]cylTask, 0, len(axes)*len(radii))
	for _, ax := range axes {
		for _, rad := range radii {
			cylTasks = append(cylTasks, cylTask{ax: ax, rad: rad})
		}
	}

	evalCylTask := func(t cylTask) (*decoder.DecodeResult, geom.Quad, bool) {
		ct := geom.NewCylindricalTransform(t.ax.cx, t.ax.cy, t.rad, t.ax.tiltX, 0)
		uTL, ok1 := ct.Inverse(set.TL.Center)
		uTR, ok2 := ct.Inverse(set.TR.Center)
		uBL, ok3 := ct.Inverse(set.BL.Center)
		if !ok1 || !ok2 || !ok3 {
			return nil, geom.Quad{}, false
		}
		// Reject severely collapsed or degenerate unwarped finder triangles before full grid sampling.
		v1 := uTR.Sub(uTL)
		v2 := uBL.Sub(uTL)
		l1, l2 := v1.Len(), v2.Len()
		if l1 < 10 || l2 < 10 || l1 > l2*3.2 || l2 > l1*3.2 {
			return nil, geom.Quad{}, false
		}
		cosU := (v1.X*v2.X + v1.Y*v2.Y) / (l1 * l2)
		if cosU < -0.75 || cosU > 0.75 {
			return nil, geom.Quad{}, false
		}
		uBR := uTR.Sub(uTL).Add(uBL)
		fBR := ct.Forward(uBR)
		margin := avgMS * 6.0
		if fBR.X < -margin || fBR.Y < -margin || fBR.X > float64(bm.Width)+margin || fBR.Y > float64(bm.Height)+margin {
			return nil, geom.Quad{}, false
		}
		cylDims := estimateUnwarpedDims(func(p geom.PointF) geom.PointF {
			q, ok := ct.Inverse(p)
			if !ok {
				return p
			}
			return q
		}, set, dim)
		for _, cDim := range cylDims {
			if hFlat, ok := makeFinderTransform(cDim, uTL, uTR, uBR, uBL, false); ok {
				projCyl := func(u, v float64) geom.PointF {
					return ct.Forward(hFlat.Project(u, v))
				}
				if res, quad, okD := sampleAndDecode(gray, bm, integ, cDim, projCyl); okD {
					return res, quad, true
				}
			}
			if uBR2, ok4 := ct.Inverse(estBR); ok4 && uBR2.Dist(uBR) >= 2.5 {
				if hFlat2, ok := makeFinderTransform(cDim, uTL, uTR, uBR2, uBL, false); ok {
					projCyl2 := func(u, v float64) geom.PointF {
						return ct.Forward(hFlat2.Project(u, v))
					}
					if res, quad, okD := sampleAndDecode(gray, bm, integ, cDim, projCyl2); okD {
						return res, quad, true
					}
				}
			}
		}
		return nil, geom.Quad{}, false
	}

	workers := runtime.GOMAXPROCS(0)
	if workers > 6 {
		workers = 6
	}
	if workers > 1 && len(cylTasks) > 1 {
		type cylOut struct {
			res  *decoder.DecodeResult
			quad geom.Quad
			ok   bool
		}
		outs := make([]cylOut, len(cylTasks))
		var foundAny atomic.Bool
		var nextIdx atomic.Int64
		var wg sync.WaitGroup
		numW := workers
		if numW > len(cylTasks) {
			numW = len(cylTasks)
		}
		wg.Add(numW)
		for w := 0; w < numW; w++ {
			go func() {
				defer wg.Done()
				for !foundAny.Load() {
					idx := int(nextIdx.Add(1) - 1)
					if idx >= len(cylTasks) {
						return
					}
					if res, quad, okD := evalCylTask(cylTasks[idx]); okD {
						outs[idx] = cylOut{res: res, quad: quad, ok: true}
						foundAny.Store(true)
						return
					}
				}
			}()
		}
		wg.Wait()
		for _, o := range outs {
			if o.ok {
				return o.res, o.quad, true
			}
		}
	} else {
		for _, t := range cylTasks {
			if res, quad, okD := evalCylTask(t); okD {
				return res, quad, true
			}
		}
	}

	// 4) Wave unwarping from timing/edge lateral deviations.
	if baseH.IsValid() {
		if wt, okW := estimateWaveTransform(bm, dim, baseH, avgMS); okW {
			uTL := wt.Inverse(set.TL.Center)
			uTR := wt.Inverse(set.TR.Center)
			uBL := wt.Inverse(set.BL.Center)
			uBR := wt.Inverse(estBR)
			if hWave, ok := makeFinderTransform(dim, uTL, uTR, uBR, uBL, false); ok {
				if res, quad, okD := sampleAndDecode(gray, bm, integ, dim, func(u, v float64) geom.PointF {
					return wt.Forward(hWave.Project(u, v))
				}); okD {
					return res, quad, true
				}
			}
		}
	}

	return nil, geom.Quad{}, false
}

// estimateWaveTransform measures lateral deviations along the top and left QR borders
// and fits a sinusoidal WaveTransform if significant periodic distortion exists.
func estimateWaveTransform(bm *imageproc.BitMatrix, dim int, baseH geom.PerspectiveTransform, ms float64) (*geom.WaveTransform, bool) {
	const numSamples = 13
	d := float64(dim)
	maxSag := math.Max(5.0, d*ms*0.15)

	// Collect vertical Y-deviations along top edge (x varies) and horizontal X-deviations along left edge (y varies).
	collectEdgeDevs := func(horizontal bool) (coords, devs []float64) {
		for i := 1; i < numSamples-1; i++ {
			t := float64(i) / float64(numSamples-1)
			var u, v, nu, nv float64
			if horizontal {
				u, v = t*d, 0
				nu, nv = 0, -1
			} else {
				u, v = 0, t*d
				nu, nv = -1, 0
			}
			pNom := baseH.Project(u, v)
			outDir := baseH.Project(u+nu, v+nv).Sub(pNom).Normalize()
			if outDir.LenSq() == 0 {
				continue
			}
			tanDir := geom.PointF{X: -outDir.Y, Y: outDir.X}
			bestOff := 0.0
			foundEdge := false
			for _, dt := range []float64{0, -ms, ms, -2 * ms, 2 * ms} {
				probe := pNom.Add(tanDir.Mul(dt))
				if off, ok := findOuterEdgeOffset(bm, probe, outDir, maxSag, ms); ok {
					if !foundEdge || off > bestOff {
						bestOff = off
						foundEdge = true
					}
				}
			}
			if foundEdge {
				if horizontal {
					coords = append(coords, pNom.X)
					devs = append(devs, bestOff*outDir.Y)
				} else {
					coords = append(coords, pNom.Y)
					devs = append(devs, bestOff*outDir.X)
				}
			}
		}
		return coords, devs
	}

	fitSinusoid := func(coords, devs []float64, span float64) (amp, freq, phase float64) {
		if len(coords) < 5 || span <= 0 {
			return 0, 0, 0
		}
		// Linearly detrend devs(x) so chord tilt from baseH does not mask the wave.
		n := float64(len(coords))
		var sx, sy, sxx, sxy float64
		for idx, x := range coords {
			y := devs[idx]
			sx += x
			sy += y
			sxx += x * x
			sxy += x * y
		}
		mx, my := sx/n, sy/n
		denom := sxx - n*mx*mx
		slope := 0.0
		if math.Abs(denom) > 1e-9 {
			slope = (sxy - n*mx*my) / denom
		}
		detrended := make([]float64, len(devs))
		for idx, x := range coords {
			detrended[idx] = devs[idx] - (my + slope*(x-mx))
		}

		bestSSE := math.MaxFloat64
		for cycles := 0.5; cycles <= 3.0; cycles += 0.1 {
			f := 2.0 * math.Pi * cycles / span
			// Fit detrended ≈ a*sin(f*x) + b*cos(f*x) + c via normal equations.
			var sSS, sCC, sSC, sS, sC, sYS, sYC, sY float64
			for idx, x := range coords {
				y := detrended[idx]
				sv, cv := math.Sin(f*x), math.Cos(f*x)
				sSS += sv * sv
				sCC += cv * cv
				sSC += sv * cv
				sS += sv
				sC += cv
				sYS += y * sv
				sYC += y * cv
				sY += y
			}
			// Center around mean to decouple constant offset c.
			sSS -= sS * sS / n
			sCC -= sC * sC / n
			sSC -= sS * sC / n
			sYS -= sY * sS / n
			sYC -= sY * sC / n
			det := sSS*sCC - sSC*sSC
			if math.Abs(det) < 1e-9 {
				continue
			}
			a := (sYS*sCC - sYC*sSC) / det
			b := (sSS*sYC - sSC*sYS) / det
			c := (sY - a*sS - b*sC) / n

			var sse float64
			for idx, x := range coords {
				pred := a*math.Sin(f*x) + b*math.Cos(f*x) + c
				err := detrended[idx] - pred
				sse += err * err
			}
			if sse < bestSSE {
				bestSSE = sse
				amp = math.Hypot(a, b)
				freq = f
				phase = math.Atan2(b, a)
			}
		}
		return amp, freq, phase
	}

	xCoords, yDevs := collectEdgeDevs(true)
	yCoords, xDevs := collectEdgeDevs(false)

	spanX := dist(baseH.Project(0, 0), baseH.Project(d, 0))
	spanY := dist(baseH.Project(0, 0), baseH.Project(0, d))

	ampY, freqY, phaseY := fitSinusoid(xCoords, yDevs, spanX)
	ampX, freqX, phaseX := fitSinusoid(yCoords, xDevs, spanY)

	if ampX < 0.8 && ampY < 0.8 {
		return nil, false
	}
	if ampX < 0.8 {
		ampX, freqX, phaseX = 0, 0, 0
	}
	if ampY < 0.8 {
		ampY, freqY, phaseY = 0, 0, 0
	}
	return geom.NewWaveTransform(ampX, freqX, phaseX, ampY, freqY, phaseY), true
}

// estimateUnwarpedDims estimates candidate QR dimensions in unwarped space by
// transforming each finder pattern's local scale alongside its center.
func estimateUnwarpedDims(unwarp func(geom.PointF) geom.PointF, set FinderPatternSet, fallbackDim int) []int {
	unwarpMS := func(p ConcentricPattern) float64 {
		if p.HasCorners {
			var uc [4]geom.PointF
			for i := 0; i < 4; i++ {
				uc[i] = unwarp(p.Corners[i])
			}
			perim := dist(uc[0], uc[1]) + dist(uc[1], uc[2]) + dist(uc[2], uc[3]) + dist(uc[3], uc[0])
			if perim > 0 {
				return (perim * 0.25) / 7.0
			}
		}
		h := 3.5 * p.ModuleSize
		c := unwarp(p.Center)
		dx := dist(unwarp(geom.PointF{X: p.Center.X - h, Y: p.Center.Y}), unwarp(geom.PointF{X: p.Center.X + h, Y: p.Center.Y}))
		dy := dist(unwarp(geom.PointF{X: p.Center.X, Y: p.Center.Y - h}), unwarp(geom.PointF{X: p.Center.X, Y: p.Center.Y + h}))
		_ = c
		return ((dx + dy) * 0.5) / 7.0
	}

	uTL := unwarp(set.TL.Center)
	uTR := unwarp(set.TR.Center)
	uBL := unwarp(set.BL.Center)
	msTL := unwarpMS(set.TL)
	msTR := unwarpMS(set.TR)
	msBL := unwarpMS(set.BL)

	var dims []int
	seen := make(map[int]bool, 4)
	addDim := func(d int) {
		if d >= 21 && d <= 177 && (d-21)%4 == 0 && !seen[d] {
			seen[d] = true
			dims = append(dims, d)
		}
	}

	if msTL > 0 && msTR > 0 && msBL > 0 {
		rawTop := dist(uTL, uTR)/math.Sqrt(msTL*msTR) + 7.0
		rawLeft := dist(uTL, uBL)/math.Sqrt(msTL*msBL) + 7.0
		avg := (rawTop + rawLeft) * 0.5
		v := int(math.Round((avg - 17.0) / 4.0))
		if v < 1 {
			v = 1
		} else if v > 40 {
			v = 40
		}
		est := 17 + 4*v
		addDim(est)
		if math.Abs(avg-float64(est)) > 1.0 {
			if avg < float64(est) {
				addDim(est - 4)
			} else {
				addDim(est + 4)
			}
		}
	}
	addDim(fallbackDim)
	return dims
}
