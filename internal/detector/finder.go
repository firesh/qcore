package detector

import (
	"math"
	"runtime"
	"sort"
	"sync"

	"github.com/firesh/qcore/internal/geom"
	"github.com/firesh/qcore/internal/imageproc"
)

// ConcentricPattern represents a located 1:1:3:1:1 QR finder pattern.
type ConcentricPattern struct {
	Center     geom.PointF
	Size       float64
	ModuleSize float64
	Corners    [4]geom.PointF // Clockwise outer 7x7 corners: TL, TR, BR, BL
	HasCorners bool
	Score      float64
}

// FindFinderPatterns scans bm for 1:1:3:1:1 finder patterns.
// It uses an adaptive row skip unless tryHarder is true or bm.Height < 400,
// and performs an additional column scan when tryHarder is enabled.
func FindFinderPatterns(bm *imageproc.BitMatrix, tryHarder bool) []ConcentricPattern {
	return findFinderPatterns(bm, tryHarder, tryHarder)
}

func findFinderPatterns(bm *imageproc.BitMatrix, tryHarder, allowTiny bool) []ConcentricPattern {
	if bm == nil || bm.Width < 7 || bm.Height < 7 {
		return nil
	}

	skip := 1
	if !tryHarder && bm.Height >= 400 {
		skip = clampInt(bm.Height/450, 2, 3)
	} else if tryHarder && bm.Height >= 2000 {
		skip = 2
	}

	var patterns []ConcentricPattern

	// Horizontal row scan (parallelized across stripes on large bitmatrices).
	numRows := (bm.Height + skip - 1) / skip
	workers := runtime.GOMAXPROCS(0)
	if workers > 4 {
		workers = 4
	}
	if workers > 1 && numRows >= 240 {
		stripes := make([][]ConcentricPattern, workers)
		var wg sync.WaitGroup
		wg.Add(workers)
		for w := 0; w < workers; w++ {
			startIdx := (numRows * w) / workers
			endIdx := (numRows * (w + 1)) / workers
			go func(workerID, sIdx, eIdx int) {
				defer wg.Done()
				var local []ConcentricPattern
				for i := sIdx; i < eIdx; i++ {
					scanLineForPatterns(bm, i*skip, true, allowTiny, &local)
				}
				stripes[workerID] = local
			}(w, startIdx, endIdx)
		}
		wg.Wait()
		for _, local := range stripes {
			for _, pat := range local {
				addPatternDeduplicated(&patterns, pat)
			}
		}
	} else {
		for y := 0; y < bm.Height; y += skip {
			scanLineForPatterns(bm, y, true, allowTiny, &patterns)
		}
	}

	// Optional vertical column scan in tryHarder mode (catches patterns missed by row quantization/blur).
	if tryHarder {
		colSkip := 1
		if bm.Width >= 2000 {
			colSkip = 3
		} else if bm.Width >= 400 {
			colSkip = 2
		}
		numCols := (bm.Width + colSkip - 1) / colSkip
		if workers > 1 && numCols >= 240 {
			stripes := make([][]ConcentricPattern, workers)
			var wg sync.WaitGroup
			wg.Add(workers)
			for w := 0; w < workers; w++ {
				startIdx := (numCols * w) / workers
				endIdx := (numCols * (w + 1)) / workers
				go func(workerID, sIdx, eIdx int) {
					defer wg.Done()
					var local []ConcentricPattern
					for i := sIdx; i < eIdx; i++ {
						scanLineForPatterns(bm, i*colSkip, false, allowTiny, &local)
					}
					stripes[workerID] = local
				}(w, startIdx, endIdx)
			}
			wg.Wait()
			for _, local := range stripes {
				for _, pat := range local {
					addPatternDeduplicated(&patterns, pat)
				}
			}
		} else {
			for x := 0; x < bm.Width; x += colSkip {
				scanLineForPatterns(bm, x, false, allowTiny, &patterns)
			}
		}
	}

	sort.Slice(patterns, func(i, j int) bool {
		if patterns[i].Score != patterns[j].Score {
			return patterns[i].Score > patterns[j].Score
		}
		return patterns[i].Size > patterns[j].Size
	})

	// Compute outer 7x7 corners once per deduplicated pattern using the averaged center/size.
	cornerLimit := len(patterns)
	if cornerLimit > 256 {
		cornerLimit = 256
	}
	for i := 0; i < cornerLimit; i++ {
		patterns[i].Corners, patterns[i].HasCorners = FindConcentricPatternCorners(bm, patterns[i].Center, patterns[i].ModuleSize)
	}

	return patterns
}

func scanLineForPatterns(bm *imageproc.BitMatrix, fixedCoord int, horizontal bool, tryHarder bool, out *[]ConcentricPattern) {
	length := bm.Width
	if !horizontal {
		length = bm.Height
	}
	if length <= 0 || len(bm.Bits) == 0 {
		return
	}

	var counts [5]int
	state := 0 // 0:B, 1:W, 2:B(3), 3:W, 4:B
	minCenter := 3
	if tryHarder {
		minCenter = 2
	}

	checkAndAdd := func(endPos int) {
		if counts[2] < minCenter {
			return
		}
		if score, ok := matchFinderRatio(counts, tryHarder); ok {
			total := counts[0] + counts[1] + counts[2] + counts[3] + counts[4]
			centerPos := float64(endPos) - float64(counts[4]+counts[3]) - float64(counts[2])*0.5
			estMS := float64(total) / 7.0
			cx, cy := centerPos, float64(fixedCoord)+0.5
			if !horizontal {
				cx, cy = float64(fixedCoord)+0.5, centerPos
			}
			if pat, ok := locateConcentricCore(bm, cx, cy, estMS, tryHarder); ok {
				if score > pat.Score {
					pat.Score = (pat.Score + score) * 0.5
				}
				addPatternDeduplicated(out, pat)
			}
		}
	}

	if horizontal {
		row := bm.Bits[fixedCoord*bm.Width : fixedCoord*bm.Width+length]
		for pos, b := range row {
			if b != 0 {
				if (state & 1) == 1 {
					state++
				}
				counts[state]++
			} else {
				if (state & 1) == 0 {
					if state == 4 {
						checkAndAdd(pos)
						counts[0] = counts[2]
						counts[1] = counts[3]
						counts[2] = counts[4]
						counts[3] = 1
						counts[4] = 0
						state = 3
					} else {
						state++
						counts[state]++
					}
				} else {
					counts[state]++
				}
			}
		}
	} else {
		stride := bm.Width
		idx := fixedCoord
		bits := bm.Bits
		for pos := 0; pos < length; pos++ {
			if bits[idx] != 0 {
				if (state & 1) == 1 {
					state++
				}
				counts[state]++
			} else {
				if (state & 1) == 0 {
					if state == 4 {
						checkAndAdd(pos)
						counts[0] = counts[2]
						counts[1] = counts[3]
						counts[2] = counts[4]
						counts[3] = 1
						counts[4] = 0
						state = 3
					} else {
						state++
						counts[state]++
					}
				} else {
					counts[state]++
				}
			}
			idx += stride
		}
	}
	if state == 4 {
		checkAndAdd(length)
	}
}

// FindPattern checks whether counts approximates the 1:1:3:1:1 finder ratio.
// When tryHarder is true, tiny patterns with center >= 2 and total >= 6 are accepted.
func FindPattern(counts [5]int, tryHarder bool) bool {
	_, ok := matchFinderRatio(counts, tryHarder)
	return ok
}

func matchFinderRatio(counts [5]int, tryHarder bool) (float64, bool) {
	if tryHarder {
		if counts[2] < 2 {
			return 0, false
		}
	} else {
		if counts[2] < 3 {
			return 0, false
		}
	}

	total := counts[0] + counts[1] + counts[2] + counts[3] + counts[4]
	if tryHarder {
		if total < 6 {
			return 0, false
		}
	} else {
		if total < 7 {
			return 0, false
		}
	}

	for i := 0; i < 5; i++ {
		if counts[i] <= 0 {
			return 0, false
		}
	}

	ms := float64(total) / 7.0
	var maxVar float64
	switch {
	case total < 12:
		maxVar = ms * 0.75
	case tryHarder:
		maxVar = ms * 0.65
	default:
		maxVar = ms * 0.55
	}

	d0 := math.Abs(ms - float64(counts[0]))
	d1 := math.Abs(ms - float64(counts[1]))
	d2 := math.Abs(3.0*ms - float64(counts[2]))
	d3 := math.Abs(ms - float64(counts[3]))
	d4 := math.Abs(ms - float64(counts[4]))

	if d0 >= maxVar || d1 >= maxVar || d2 >= 3.0*maxVar || d3 >= maxVar || d4 >= maxVar {
		return 0, false
	}

	errSum := (d0 + d1 + d2/3.0 + d3 + d4) / (5.0 * ms)
	score := 1.0 - errSum
	if score < 0.05 {
		score = 0.05
	}
	return score, true
}

// LocateConcentricPattern cross-checks horizontal, vertical, and diagonal 1:1:3:1:1
// ratios around (cx, cy), refines the sub-pixel center via concentric ring centering,
// and extracts the outer 7x7 square corners when possible.
func LocateConcentricPattern(bm *imageproc.BitMatrix, cx, cy, estimatedModuleSize float64, tryHarder bool) (ConcentricPattern, bool) {
	pat, ok := locateConcentricCore(bm, cx, cy, estimatedModuleSize, tryHarder)
	if !ok {
		return ConcentricPattern{}, false
	}
	pat.Corners, pat.HasCorners = FindConcentricPatternCorners(bm, pat.Center, pat.ModuleSize)
	return pat, true
}

func locateConcentricCore(bm *imageproc.BitMatrix, cx, cy, estimatedModuleSize float64, tryHarder bool) (ConcentricPattern, bool) {
	ix := clampInt(int(math.Floor(cx)), 0, bm.Width-1)
	iy := clampInt(int(math.Floor(cy)), 0, bm.Height-1)
	if !bm.Get(ix, iy) {
		return ConcentricPattern{}, false
	}

	maxScan := int(math.Ceil(estimatedModuleSize * 6.5))
	if maxScan < 10 {
		maxScan = 10
	}

	// Vertical cross-check.
	vCenter, vTotal, vScore, ok := crossCheckAxis(bm, ix, iy, 0, 1, maxScan, estimatedModuleSize, tryHarder)
	if !ok {
		return ConcentricPattern{}, false
	}
	iyRef := clampInt(int(math.Floor(vCenter)), 0, bm.Height-1)

	// Horizontal cross-check at refined Y.
	hCenter, hTotal, hScore, ok := crossCheckAxis(bm, ix, iyRef, 1, 0, maxScan, estimatedModuleSize, tryHarder)
	if !ok {
		return ConcentricPattern{}, false
	}
	ixRef := clampInt(int(math.Floor(hCenter)), 0, bm.Width-1)

	// Re-verify vertical at refined X to handle off-center initial hits.
	if ixRef != ix {
		if vc2, vt2, vs2, ok2 := crossCheckAxis(bm, ixRef, iyRef, 0, 1, maxScan, estimatedModuleSize, tryHarder); ok2 {
			vCenter, vTotal, vScore = vc2, vt2, vs2
		}
	}

	// Consistency between horizontal and vertical sizes.
	if math.Abs(hTotal-vTotal) > math.Max(hTotal, vTotal)*0.55 {
		return ConcentricPattern{}, false
	}

	// Diagonal cross-checks (+1,+1) and (+1,-1). Pass 0 for estTotalOrMS so rotated
	// 45-degree finder squares (where diagonal runs are shorter than HV runs) are not rejected.
	diagMaxScan := int(math.Ceil(float64(maxScan) * 1.45))
	_, d1Total, d1Score, ok1 := crossCheckAxis(bm, ixRef, iyRef, 1, 1, diagMaxScan, 0, tryHarder)
	_, d2Total, d2Score, ok2 := crossCheckAxis(bm, ixRef, iyRef, 1, -1, diagMaxScan, 0, tryHarder)

	hvAvg := (hTotal + vTotal) * 0.5
	if ok1 && (d1Total < 0.35*hvAvg || d1Total > 1.95*hvAvg) {
		ok1 = false
	}
	if ok2 && (d2Total < 0.35*hvAvg || d2Total > 1.95*hvAvg) {
		ok2 = false
	}

	if !ok1 && !ok2 {
		return ConcentricPattern{}, false
	}
	// If only one diagonal passed, ensure HV quality is solid unless tryHarder.
	if (!ok1 || !ok2) && !tryHarder {
		if hScore < 0.45 || vScore < 0.45 {
			return ConcentricPattern{}, false
		}
	}

	size := (hTotal + vTotal) * 0.5
	diagCount := 0.0
	diagScore := 0.0
	if ok1 {
		diagCount++
		diagScore += d1Score
		size = (size*2.0 + d1Total/math.Sqrt2) / 3.0
	}
	if ok2 {
		diagCount++
		diagScore += d2Score
		if ok1 {
			size = (hTotal + vTotal + (d1Total+d2Total)/math.Sqrt2) / 4.0
		} else {
			size = (size*2.0 + d2Total/math.Sqrt2) / 3.0
		}
	}

	ms := size / 7.0
	center := geom.PointF{X: hCenter, Y: vCenter}

	// Refine sub-pixel center via concentric ring centering.
	if refined, okR := CenterOfRing(bm, center, ms); okR {
		if dist(refined, center) <= ms*1.2 {
			center = refined
		}
	}

	score := (hScore + vScore + diagScore) / (2.0 + diagCount)
	if ok1 && ok2 {
		score += 0.1
	}

	return ConcentricPattern{
		Center:     center,
		Size:       size,
		ModuleSize: ms,
		Score:      score,
	}, true
}

func crossCheckAxis(bm *imageproc.BitMatrix, startX, startY, dx, dy, maxCount int, estTotalOrMS float64, tryHarder bool) (centerCoord float64, totalLen float64, score float64, ok bool) {
	if !bm.Get(startX, startY) {
		return 0, 0, 0, false
	}

	var counts [5]int

	// Walk backward (-dx, -dy) from start.
	x, y := startX, startY
	steps := 0
	for inBounds(bm, x, y) && bm.Get(x, y) && steps < maxCount {
		counts[2]++
		x -= dx
		y -= dy
		steps++
	}
	if !inBounds(bm, x, y) || counts[2] == 0 {
		return 0, 0, 0, false
	}
	for inBounds(bm, x, y) && !bm.Get(x, y) && steps < maxCount {
		counts[1]++
		x -= dx
		y -= dy
		steps++
	}
	if !inBounds(bm, x, y) || counts[1] == 0 {
		return 0, 0, 0, false
	}
	for inBounds(bm, x, y) && bm.Get(x, y) && steps < maxCount {
		counts[0]++
		x -= dx
		y -= dy
		steps++
	}
	if counts[0] == 0 {
		return 0, 0, 0, false
	}

	// Walk forward (+dx, +dy) from start + step.
	x, y = startX+dx, startY+dy
	steps = 0
	for inBounds(bm, x, y) && bm.Get(x, y) && steps < maxCount {
		counts[2]++
		x += dx
		y += dy
		steps++
	}
	if !inBounds(bm, x, y) {
		return 0, 0, 0, false
	}
	for inBounds(bm, x, y) && !bm.Get(x, y) && steps < maxCount {
		counts[3]++
		x += dx
		y += dy
		steps++
	}
	if !inBounds(bm, x, y) || counts[3] == 0 {
		return 0, 0, 0, false
	}
	for inBounds(bm, x, y) && bm.Get(x, y) && steps < maxCount {
		counts[4]++
		x += dx
		y += dy
		steps++
	}
	if counts[4] == 0 {
		return 0, 0, 0, false
	}

	score, ok = matchFinderRatio(counts, tryHarder)
	if !ok {
		return 0, 0, 0, false
	}

	totalSteps := float64(counts[0] + counts[1] + counts[2] + counts[3] + counts[4])
	stepLen := math.Hypot(float64(dx), float64(dy))
	totalLen = totalSteps * stepLen

	estTotal := estTotalOrMS * 7.0
	if estTotal > 0 && math.Abs(totalLen-estTotal) > 0.65*estTotal {
		return 0, 0, 0, false
	}

	// Compute center coordinate along the primary axis (x if dx != 0, else y).
	endX, endY := x, y
	centerStepOffset := float64(counts[4]+counts[3]) + float64(counts[2])*0.5
	if dx != 0 {
		centerCoord = float64(endX) - float64(dx)*centerStepOffset
		if dx > 0 {
			centerCoord += 0.0 // endX is 1 past last black pixel, so subtracting offset gives pixel center
		}
	} else {
		centerCoord = float64(endY) - float64(dy)*centerStepOffset
	}
	return centerCoord, totalLen, score, true
}

// CenterOfRing refines the sub-pixel center of a concentric black/white ring pattern
// around center with estimated moduleSize by casting opposite ray pairs.
func CenterOfRing(bm *imageproc.BitMatrix, center geom.PointF, moduleSize float64) (geom.PointF, bool) {
	if moduleSize < 0.5 {
		moduleSize = 0.5
	}
	const numDirs = 8 // 8 opposite pairs = 16 rays
	step := math.Max(0.2, moduleSize*0.18)
	maxDist := moduleSize * 5.2

	var sumX, sumY float64
	var weightSum float64

	for i := 0; i < numDirs; i++ {
		angle := float64(i) * math.Pi / float64(numDirs)
		dx, dy := math.Cos(angle), math.Sin(angle)

		pR1, pR2, pR3, okPos := traceConcentricRay(bm, center, dx, dy, step, maxDist)
		nR1, nR2, nR3, okNeg := traceConcentricRay(bm, center, -dx, -dy, step, maxDist)

		if okPos && okNeg {
			// Inner 3x3 black boundary midpoint (highest reliability).
			sumX += (pR1.X + nR1.X) * 0.5 * 2.0
			sumY += (pR1.Y + nR1.Y) * 0.5 * 2.0
			weightSum += 2.0

			// White ring outer boundary midpoint.
			if pR2.X >= 0 && nR2.X >= 0 {
				sumX += (pR2.X + nR2.X) * 0.5
				sumY += (pR2.Y + nR2.Y) * 0.5
				weightSum += 1.0
			}
			// Outer 7x7 black boundary midpoint.
			if pR3.X >= 0 && nR3.X >= 0 {
				sumX += (pR3.X + nR3.X) * 0.5
				sumY += (pR3.Y + nR3.Y) * 0.5
				weightSum += 1.0
			}
		}
	}

	if weightSum < 4.0 {
		return center, false
	}
	return geom.PointF{X: sumX / weightSum, Y: sumY / weightSum}, true
}

// traceConcentricRay walks from center along (dx, dy) and records transitions:
// r1: inner black -> white
// r2: white -> outer black
// r3: outer black -> white
func traceConcentricRay(bm *imageproc.BitMatrix, center geom.PointF, dx, dy, step, maxDist float64) (r1, r2, r3 geom.PointF, ok bool) {
	r1 = geom.PointF{X: -1, Y: -1}
	r2 = geom.PointF{X: -1, Y: -1}
	r3 = geom.PointF{X: -1, Y: -1}

	ix := int(math.Floor(center.X))
	iy := int(math.Floor(center.Y))
	if !inBounds(bm, ix, iy) || !bm.Get(ix, iy) {
		return r1, r2, r3, false
	}

	state := 0 // 0: inner black, 1: white ring, 2: outer black
	prevPt := center
	for d := step; d <= maxDist; d += step {
		px := center.X + dx*d
		py := center.Y + dy*d
		ix = int(math.Floor(px))
		iy = int(math.Floor(py))
		if !inBounds(bm, ix, iy) {
			break
		}
		black := bm.Get(ix, iy)
		mid := geom.PointF{X: (prevPt.X + px) * 0.5, Y: (prevPt.Y + py) * 0.5}
		switch state {
		case 0:
			if !black {
				r1 = mid
				state = 1
			}
		case 1:
			if black {
				r2 = mid
				state = 2
			}
		case 2:
			if !black {
				r3 = mid
				return r1, r2, r3, true
			}
		}
		prevPt = geom.PointF{X: px, Y: py}
	}

	return r1, r2, r3, r1.X >= 0
}

// FindConcentricPatternCorners extracts the 4 outer corners of the 7x7 finder square
// ordered clockwise around center.
func FindConcentricPatternCorners(bm *imageproc.BitMatrix, center geom.PointF, moduleSize float64) ([4]geom.PointF, bool) {
	var zero [4]geom.PointF
	if moduleSize < 1.0 {
		return zero, false
	}

	const numRays = 48
	step := math.Max(0.25, moduleSize*0.2)
	maxDist := moduleSize * 6.2

	outerPts := make([]geom.PointF, 0, numRays)
	for i := 0; i < numRays; i++ {
		angle := float64(i) * (2.0 * math.Pi / float64(numRays))
		dx, dy := math.Cos(angle), math.Sin(angle)
		_, _, r3, ok := traceConcentricRay(bm, center, dx, dy, step, maxDist)
		if !ok || r3.X < 0 {
			continue
		}
		d := dist(center, r3)
		// Expected outer boundary of 7x7 square lies between 2.8*ms (edge midpoint ~3.5) and 5.4*ms (corner ~4.95).
		if d >= 2.3*moduleSize && d <= 5.8*moduleSize {
			outerPts = append(outerPts, r3)
		}
	}

	if len(outerPts) < 16 {
		return zero, false
	}

	// Find extremal corner candidates on the convex hull / polar boundary.
	// 1) p0: farthest from center.
	idx0 := 0
	maxD0 := -1.0
	for i, p := range outerPts {
		if d := distSq(center, p); d > maxD0 {
			maxD0 = d
			idx0 = i
		}
	}
	p0 := outerPts[idx0]

	// 2) p2: farthest from p0 (opposite corner).
	idx2 := 0
	maxD2 := -1.0
	for i, p := range outerPts {
		if d := distSq(p0, p); d > maxD2 {
			maxD2 = d
			idx2 = i
		}
	}
	p2 := outerPts[idx2]

	// 3) p1 and p3: maximum perpendicular signed distance on each side of diagonal p0->p2.
	diagX := p2.X - p0.X
	diagY := p2.Y - p0.Y
	diagLen := math.Hypot(diagX, diagY)
	if diagLen < 4.0*moduleSize {
		return zero, false
	}

	idx1, idx3 := -1, -1
	maxPos, maxNeg := 0.0, 0.0
	for i, p := range outerPts {
		cross := diagX*(p.Y-p0.Y) - diagY*(p.X-p0.X)
		if cross > maxPos {
			maxPos = cross
			idx1 = i
		} else if cross < maxNeg {
			maxNeg = cross
			idx3 = i
		}
	}
	if idx1 < 0 || idx3 < 0 {
		return zero, false
	}

	// Partition outerPts into the 4 sides around polar order and fit a RegressionLine per side.
	cornerIdxs := [4]int{idx0, idx1, idx2, idx3}
	// Sort corner indices in order along outerPts.
	sort.Ints(cornerIdxs[:])

	var lines [4]geom.RegressionLine
	n := len(outerPts)
	for side := 0; side < 4; side++ {
		a := cornerIdxs[side]
		b := cornerIdxs[(side+1)%4]
		var sidePts []geom.PointF
		for k := 1; k < n; k++ {
			idx := (a + k) % n
			if idx == b {
				break
			}
			sidePts = append(sidePts, outerPts[idx])
		}
		// Trim 15% near corners to avoid rounded corner bias.
		if len(sidePts) >= 5 {
			trim := len(sidePts) / 6
			if trim >= 1 {
				sidePts = sidePts[trim : len(sidePts)-trim]
			}
		}
		if len(sidePts) < 2 {
			return zero, false
		}
		for _, sp := range sidePts {
			lines[side].Add(sp)
		}
		if !lines[side].Evaluate(math.Max(1.0, moduleSize*0.6), true) {
			return zero, false
		}
	}

	var rawCorners [4]geom.PointF
	for side := 0; side < 4; side++ {
		inter, ok := geom.IntersectLines(lines[side], lines[(side+1)%4])
		if !ok {
			return zero, false
		}
		d := dist(center, inter)
		if d < 2.8*moduleSize || d > 7.0*moduleSize {
			return zero, false
		}
		rawCorners[side] = inter
	}

	ordered := geom.OrderCornersClockwise(rawCorners)
	// Verify reasonable side lengths around ~7*moduleSize.
	for i := 0; i < 4; i++ {
		sideLen := dist(ordered[i], ordered[(i+1)%4])
		if sideLen < 4.0*moduleSize || sideLen > 10.5*moduleSize {
			return zero, false
		}
	}

	return [4]geom.PointF(ordered), true
}

func addPatternDeduplicated(patterns *[]ConcentricPattern, pat ConcentricPattern) {
	for i := range *patterns {
		old := &(*patterns)[i]
		mergeDist := math.Max(4.0, old.Size*0.5)
		if dist(pat.Center, old.Center) <= mergeDist &&
			math.Abs(pat.Size-old.Size) <= math.Max(pat.Size, old.Size)*0.55 {
			totalScore := old.Score + pat.Score
			if totalScore > 0 {
				old.Center.X = (old.Center.X*old.Score + pat.Center.X*pat.Score) / totalScore
				old.Center.Y = (old.Center.Y*old.Score + pat.Center.Y*pat.Score) / totalScore
				old.Size = (old.Size*old.Score + pat.Size*pat.Score) / totalScore
				old.ModuleSize = old.Size / 7.0
			}
			if pat.Score > old.Score {
				old.Score = pat.Score
				if pat.HasCorners {
					old.Corners = pat.Corners
					old.HasCorners = true
				}
			} else if !old.HasCorners && pat.HasCorners {
				old.Corners = pat.Corners
				old.HasCorners = true
			}
			return
		}
	}
	*patterns = append(*patterns, pat)
}

func inBounds(bm *imageproc.BitMatrix, x, y int) bool {
	return x >= 0 && y >= 0 && x < bm.Width && y < bm.Height
}

func clampInt(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func distSq(a, b geom.PointF) float64 {
	dx := a.X - b.X
	dy := a.Y - b.Y
	return dx*dx + dy*dy
}

func dist(a, b geom.PointF) float64 {
	return math.Hypot(a.X-b.X, a.Y-b.Y)
}
