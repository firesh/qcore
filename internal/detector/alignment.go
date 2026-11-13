package detector

import (
	"math"

	"github.com/firesh/qcore/internal/geom"
	"github.com/firesh/qcore/internal/imageproc"
)

// LocateAlignmentPattern searches in a spiral around estimate within searchRadius
// for a 5x5 QR alignment pattern (1:1:1:1:1 B:W:B:W:B core) and refines its sub-pixel center.
func LocateAlignmentPattern(bm *imageproc.BitMatrix, gray *imageproc.GrayImage, moduleSize float64, estimate geom.PointF, searchRadius float64) (geom.PointF, bool) {
	if bm == nil || moduleSize <= 0 || searchRadius <= 0 {
		return geom.PointF{}, false
	}

	cx := int(math.Round(estimate.X))
	cy := int(math.Round(estimate.Y))
	maxR := int(math.Ceil(searchRadius))
	if maxR < 2 {
		maxR = 2
	}

	step := 1
	if maxR > 24 && moduleSize > 4.0 {
		step = int(math.Max(2.0, math.Floor(moduleSize*0.35)))
	}

	var bestPt geom.PointF
	bestDistSq := math.MaxFloat64
	bestScore := -1.0
	bmW, bmH := bm.Width, bm.Height
	bmBits := bm.Bits

	// Expanding square spiral around (cx, cy).
	checkCandidate := func(x, y int) {
		if x < 0 || y < 0 || x >= bmW || y >= bmH || bmBits[y*bmW+x] == 0 {
			return
		}
		dx := float64(x) - estimate.X
		dy := float64(y) - estimate.Y
		if dx*dx+dy*dy > searchRadius*searchRadius {
			return
		}
		hCenter, hScore, okH := checkAlignment1D(bm, x, y, 1, 0, moduleSize)
		if !okH {
			return
		}
		vCenter, vScore, okV := checkAlignment1D(bm, int(math.Round(hCenter)), y, 0, 1, moduleSize)
		if !okV {
			return
		}
		// Re-check horizontal at refined Y if needed.
		if int(math.Round(vCenter)) != y {
			if hc2, hs2, okH2 := checkAlignment1D(bm, int(math.Round(hCenter)), int(math.Round(vCenter)), 1, 0, moduleSize); okH2 {
				hCenter, hScore = hc2, hs2
			}
		}

		cand := geom.PointF{X: hCenter, Y: vCenter}
		if refined, okR := centerOfAlignmentRing(bm, gray, cand, moduleSize); okR {
			if refined.Dist(cand) <= moduleSize*1.0 {
				cand = refined
			}
		}
		if cand.Dist(estimate) > searchRadius+moduleSize {
			return
		}

		score := (hScore + vScore) * 0.5
		dSq := cand.DistSq(estimate)
		if score > bestScore+0.15 || (math.Abs(score-bestScore) <= 0.15 && dSq < bestDistSq) {
			bestScore = score
			bestDistSq = dSq
			bestPt = cand
		}
	}

	checkCandidate(cx, cy)
	if bestScore >= 0.75 {
		return bestPt, true
	}

	for r := step; r <= maxR; r += step {
		for dx := -r; dx <= r; dx += step {
			checkCandidate(cx+dx, cy-r)
			checkCandidate(cx+dx, cy+r)
		}
		for dy := -r + step; dy <= r-step; dy += step {
			checkCandidate(cx-r, cy+dy)
			checkCandidate(cx+r, cy+dy)
		}
		if bestScore > 0 {
			return bestPt, true
		}
	}

	return geom.PointF{}, false
}

// checkAlignment1D verifies the 5-bar B:W:B:W:B cross-section of a 5x5 alignment pattern
// where outer black bars have width >= 0.35*ms (since adjacent data modules may merge with
// the outer border) and the inner W:B:W core matches 1:1:1.
func checkAlignment1D(bm *imageproc.BitMatrix, startX, startY, dx, dy int, ms float64) (centerCoord float64, score float64, ok bool) {
	w, h := bm.Width, bm.Height
	if startX < 0 || startY < 0 || startX >= w || startY >= h {
		return 0, 0, false
	}
	bits := bm.Bits
	if bits[startY*w+startX] == 0 {
		return 0, 0, false
	}

	maxHalf := int(math.Ceil(ms * 4.5))
	if maxHalf < 5 {
		maxHalf = 5
	}

	var bCenter, wLeft, wRight, bLeft, bRight int
	stride := dy*w + dx

	// Walk backward.
	x, y, idx := startX, startY, startY*w+startX
	steps := 0
	for x >= 0 && y >= 0 && x < w && y < h && bits[idx] != 0 && steps < maxHalf {
		bCenter++
		x -= dx
		y -= dy
		idx -= stride
		steps++
	}
	if x < 0 || y < 0 || x >= w || y >= h || bCenter == 0 {
		return 0, 0, false
	}
	for x >= 0 && y >= 0 && x < w && y < h && bits[idx] == 0 && steps < maxHalf {
		wLeft++
		x -= dx
		y -= dy
		idx -= stride
		steps++
	}
	if x < 0 || y < 0 || x >= w || y >= h || wLeft == 0 {
		return 0, 0, false
	}
	for x >= 0 && y >= 0 && x < w && y < h && bits[idx] != 0 && steps < maxHalf {
		bLeft++
		x -= dx
		y -= dy
		idx -= stride
		steps++
	}
	if bLeft == 0 {
		return 0, 0, false
	}

	// Walk forward.
	x, y = startX+dx, startY+dy
	idx = y*w + x
	steps = 0
	for x >= 0 && y >= 0 && x < w && y < h && bits[idx] != 0 && steps < maxHalf {
		bCenter++
		x += dx
		y += dy
		idx += stride
		steps++
	}
	if x < 0 || y < 0 || x >= w || y >= h {
		return 0, 0, false
	}
	for x >= 0 && y >= 0 && x < w && y < h && bits[idx] == 0 && steps < maxHalf {
		wRight++
		x += dx
		y += dy
		idx += stride
		steps++
	}
	if x < 0 || y < 0 || x >= w || y >= h || wRight == 0 {
		return 0, 0, false
	}
	for x >= 0 && y >= 0 && x < w && y < h && bits[idx] != 0 && steps < maxHalf {
		bRight++
		x += dx
		y += dy
		idx += stride
		steps++
	}
	if bRight == 0 {
		return 0, 0, false
	}

	minW := math.Max(0.5, ms*0.35)
	maxW := math.Max(2.5, ms*2.3)
	if float64(bCenter) < minW || float64(bCenter) > maxW ||
		float64(wLeft) < minW || float64(wLeft) > maxW ||
		float64(wRight) < minW || float64(wRight) > maxW ||
		float64(bLeft) < minW*0.8 || float64(bRight) < minW*0.8 {
		return 0, 0, false
	}

	coreTotal := float64(wLeft + bCenter + wRight)
	estMS := coreTotal / 3.0
	if math.Abs(estMS-ms) > ms*0.75 {
		return 0, 0, false
	}

	d0 := math.Abs(float64(wLeft) - estMS)
	d1 := math.Abs(float64(bCenter) - estMS)
	d2 := math.Abs(float64(wRight) - estMS)
	errFrac := (d0 + d1 + d2) / (3.0 * estMS)
	if errFrac > 0.65 {
		return 0, 0, false
	}

	// Compute center coordinate of the central black module along the scan axis.
	// At the end of forward bRight scan, (x, y) is 1 step past bRight.
	offset := float64(bRight) + float64(wRight) + float64(bCenter)*0.5
	if dx != 0 {
		centerCoord = float64(x) - float64(dx)*offset
	} else {
		centerCoord = float64(y) - float64(dy)*offset
	}
	return centerCoord, 1.0 - errFrac, true
}

func centerOfAlignmentRing(bm *imageproc.BitMatrix, gray *imageproc.GrayImage, center geom.PointF, ms float64) (geom.PointF, bool) {
	const numDirs = 8
	step := math.Max(0.2, ms*0.18)
	maxDist := ms * 2.8

	var sumX, sumY, wSum float64
	for i := 0; i < numDirs; i++ {
		angle := float64(i) * math.Pi / float64(numDirs)
		dx, dy := math.Cos(angle), math.Sin(angle)

		pR1, pR2, _, okP := traceConcentricRay(bm, center, dx, dy, step, maxDist)
		nR1, nR2, _, okN := traceConcentricRay(bm, center, -dx, -dy, step, maxDist)
		if okP && okN {
			sumX += (pR1.X + nR1.X) * 0.5 * 2.0
			sumY += (pR1.Y + nR1.Y) * 0.5 * 2.0
			wSum += 2.0
			if pR2.X >= 0 && nR2.X >= 0 {
				sumX += (pR2.X + nR2.X) * 0.5
				sumY += (pR2.Y + nR2.Y) * 0.5
				wSum += 1.0
			}
		}
	}

	if wSum < 4.0 {
		return center, false
	}
	refined := geom.PointF{X: sumX / wSum, Y: sumY / wSum}

	// Sub-pixel grayscale dark-centroid refinement inside the 1x1 center dot when gray is provided.
	if gray != nil && ms >= 2.0 {
		rad := int(math.Max(1.0, math.Floor(ms*0.65)))
		cx := int(math.Round(refined.X))
		cy := int(math.Round(refined.Y))
		var maxVal float64
		for y := cy - rad - 1; y <= cy+rad+1; y++ {
			for x := cx - rad - 1; x <= cx+rad+1; x++ {
				v := float64(gray.At(x, y))
				if v > maxVal {
					maxVal = v
				}
			}
		}
		var gx, gy, gSum float64
		for y := cy - rad; y <= cy+rad; y++ {
			for x := cx - rad; x <= cx+rad; x++ {
				w := maxVal - float64(gray.At(x, y))
				if w > 0 {
					gx += (float64(x) + 0.5) * w
					gy += (float64(y) + 0.5) * w
					gSum += w
				}
			}
		}
		if gSum > 0 {
			gCenter := geom.PointF{X: gx / gSum, Y: gy / gSum}
			if gCenter.Dist(refined) <= ms*0.5 {
				refined = refined.Midpoint(gCenter)
			}
		}
	}

	return refined, true
}

// TraceLine traces along the 2nd or 3rd concentric edge of a finder pattern centered at p
// in the direction toward target using imageproc.CursorI, returning a fitted RegressionLine.
// edge > 0 searches along the left-hand perpendicular normal (dir.Y, -dir.X) in screen coords
// (e.g. outer right edge when tracing TR -> BR); edge < 0 searches along (-dir.Y, dir.X)
// (e.g. outer bottom edge when tracing BL -> BR). |edge| specifies the transition index (2 or 3).
func TraceLine(bm *imageproc.BitMatrix, p, target geom.PointF, edge int) geom.RegressionLine {
	var line geom.RegressionLine
	if bm == nil || edge == 0 {
		return line
	}

	dirVec := target.Sub(p)
	if dirVec.LenSq() < 1.0 {
		return line
	}
	dir := dirVec.Normalize()

	var perp geom.PointF
	edgeNum := edge
	if edge > 0 {
		perp = geom.PointF{X: dir.Y, Y: -dir.X}
	} else {
		perp = geom.PointF{X: -dir.Y, Y: dir.X}
		edgeNum = -edge
	}

	// Step outward from finder pattern center along perp to find the requested edge (2nd or 3rd).
	maxWalk := int(math.Min(float64(bm.Width+bm.Height), math.Max(30.0, dirVec.Len()*0.5)))
	ray := imageproc.NewCursorF(bm, p.X, p.Y, perp.X, perp.Y)
	steps := ray.StepToEdge(edgeNum, maxWalk, true)
	if steps == 0 {
		return line
	}

	// Choose the cardinal Direction closest to dir for integer CursorI edge following.
	cardDir := nearestCardinalDir(dir)
	cdx, cdy := cardDir.Delta()
	// Determine which side of CursorI faces outward along perp:
	// CursorI.EdgeAtLeft() checks (DY, -DX), EdgeAtRight() checks (-DY, DX).
	sideDir := 1
	if float64(cdy)*perp.X+float64(-cdx)*perp.Y > 0 {
		sideDir = -1
	}

	startX := int(math.Floor(ray.X))
	startY := int(math.Floor(ray.Y))
	if !bm.IsIn(startX, startY) {
		return line
	}

	maxEdgeSteps := int(math.Min(float64(bm.Width+bm.Height), math.Max(16.0, float64(steps)*2.8)))
	offsetToBoundary := perp.Mul(0.5)

	// Trace forward along dir.
	fwd := imageproc.NewCursorI(bm, startX, startY, cardDir)
	line.Add(geom.Centered(geom.PointI{X: fwd.X, Y: fwd.Y}).Add(offsetToBoundary))
	for i := 0; i < maxEdgeSteps; i++ {
		if !fwd.StepAlongEdge(sideDir, true) {
			break
		}
		// Stop if edge turns away sharply from target direction.
		if float64(fwd.DX)*dir.X+float64(fwd.DY)*dir.Y < 0.25 {
			break
		}
		line.Add(geom.Centered(geom.PointI{X: fwd.X, Y: fwd.Y}).Add(offsetToBoundary))
	}

	// Trace backward along -dir from the starting edge pixel to cover the full 7x7 finder edge.
	bwd := imageproc.NewCursorI(bm, startX, startY, cardDir.TurnBack())
	bwdSide := -sideDir
	for i := 0; i < maxEdgeSteps; i++ {
		if !bwd.StepAlongEdge(bwdSide, true) {
			break
		}
		if float64(bwd.DX)*(-dir.X)+float64(bwd.DY)*(-dir.Y) < 0.25 {
			break
		}
		line.Add(geom.Centered(geom.PointI{X: bwd.X, Y: bwd.Y}).Add(offsetToBoundary))
	}

	line.SetDirectionInward(perp.Neg())
	line.Evaluate(1.5, true)
	return line
}

// TraceFinderEdge samples 9 parallel rays perpendicular to a finder pattern edge
// (along normalDir) at stations along tangentDir in [-2.2*ms, +2.2*ms] around center,
// locates transition edgeNum (2 = white->outer black, 3 = outer black->white),
// and fits a robust RegressionLine with outlier trimming.
func TraceFinderEdge(bm *imageproc.BitMatrix, center, tangentDir, normalDir geom.PointF, ms float64, edgeNum int) geom.RegressionLine {
	var line geom.RegressionLine
	if bm == nil || ms <= 0 || edgeNum <= 0 {
		return line
	}
	tDir := tangentDir.Normalize()
	nDir := normalDir.Normalize()
	if tDir.LenSq() == 0 || nDir.LenSq() == 0 {
		return line
	}

	step := 0.25
	maxDist := math.Max(8.0, ms*6.0)
	const numStations = 9
	for i := 0; i < numStations; i++ {
		frac := -1.3 + 2.6*float64(i)/float64(numStations-1)
		origin := center.Add(tDir.Mul(frac * ms))
		ox := int(math.Floor(origin.X))
		oy := int(math.Floor(origin.Y))
		if !bm.IsIn(ox, oy) || !bm.Get(ox, oy) {
			continue
		}
		prevBlack := true
		transitions := 0
		for d := step; d <= maxDist; d += step {
			px := origin.X + nDir.X*d
			py := origin.Y + nDir.Y*d
			ix := int(math.Floor(px))
			iy := int(math.Floor(py))
			if !bm.IsIn(ix, iy) {
				break
			}
			curBlack := bm.Get(ix, iy)
			if curBlack != prevBlack {
				transitions++
				if transitions == edgeNum {
					// Half-step midpoint for sub-pixel edge accuracy.
					edgePt := geom.PointF{
						X: px - nDir.X*(step*0.5),
						Y: py - nDir.Y*(step*0.5),
					}
					line.Add(edgePt)
					break
				}
				prevBlack = curBlack
			}
		}
	}

	if len(line.Points()) >= 3 {
		line.SetDirectionInward(nDir.Neg())
		line.Evaluate(math.Max(1.0, ms*0.5), true)
	}
	return line
}

func nearestCardinalDir(d geom.PointF) imageproc.Direction {
	if math.Abs(d.X) >= math.Abs(d.Y) {
		if d.X >= 0 {
			return imageproc.DirRight
		}
		return imageproc.DirLeft
	}
	if d.Y >= 0 {
		return imageproc.DirDown
	}
	return imageproc.DirUp
}
