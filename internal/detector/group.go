package detector

import (
	"math"
	"sort"

	"github.com/firesh/qcore/internal/geom"
	"github.com/firesh/qcore/internal/imageproc"
)

// FinderPatternSet represents a candidate trio of QR finder patterns (BL, TL, TR).
type FinderPatternSet struct {
	BL    ConcentricPattern
	TL    ConcentricPattern
	TR    ConcentricPattern
	Score float64
}

// GenerateFinderPatternSets groups detected concentric patterns into valid QR
// finder pattern trios (BL, TL, TR), sorted descending by quality score.
// For large pattern counts (dense multi-QR scenes), spatial binning keeps
// candidate pairing near O(N).
func GenerateFinderPatternSets(patterns []ConcentricPattern, maxSets int, relaxed bool) []FinderPatternSet {
	n := len(patterns)
	if n < 3 {
		return nil
	}

	// If a noisy binarizer yields hundreds of low-score speckles, keep only the top 260
	// strongest patterns so pairing stays bounded without truncating dense grids like lots (~180-230).
	if n > 260 {
		scored := make([]ConcentricPattern, n)
		copy(scored, patterns)
		sort.Slice(scored, func(i, j int) bool {
			si, sj := scored[i].Score, scored[j].Score
			if scored[i].HasCorners {
				si += 0.15
			}
			if scored[j].HasCorners {
				sj += 0.15
			}
			if si != sj {
				return si > sj
			}
			return scored[i].Size > scored[j].Size
		})
		patterns = scored[:260]
		n = 260
	}

	sorted := make([]ConcentricPattern, n)
	copy(sorted, patterns)
	sort.Slice(sorted, func(i, j int) bool {
		if sorted[i].Size != sorted[j].Size {
			return sorted[i].Size > sorted[j].Size
		}
		return sorted[i].Score > sorted[j].Score
	})

	sizeRatioMax := 2.2
	legRatioMax := 2.0
	cosMax := 0.50
	if relaxed {
		sizeRatioMax = 3.5
		legRatioMax = 3.5
		cosMax = 0.70
	}

	// Build spatial hash grid when pattern count is large ("lots" dense multi-QR images).
	useGrid := n > 32
	var cellSize float64
	var grid map[[2]int][]int
	if useGrid {
		medianSize := sorted[n/2].Size
		cellSize = math.Max(40.0, medianSize*16.0)
		grid = make(map[[2]int][]int, n)
		for idx, p := range sorted {
			cx := int(math.Floor(p.Center.X / cellSize))
			cy := int(math.Floor(p.Center.Y / cellSize))
			grid[[2]int{cx, cy}] = append(grid[[2]int{cx, cy}], idx)
		}
	}

	type indexedSet struct {
		set        FinderPatternSet
		i0, i1, i2 int
	}
	var rawSets []indexedSet

	for i := 0; i < n-2; i++ {
		pi := sorted[i]
		if pi.ModuleSize <= 0 {
			continue
		}
		maxDist := 270.0 * pi.ModuleSize
		minSize := pi.Size / sizeRatioMax

		// Collect candidate neighbor indices > i within maxDist and compatible Size.
		var cands []int
		if useGrid {
			gx0 := int(math.Floor((pi.Center.X - maxDist) / cellSize))
			gx1 := int(math.Floor((pi.Center.X + maxDist) / cellSize))
			gy0 := int(math.Floor((pi.Center.Y - maxDist) / cellSize))
			gy1 := int(math.Floor((pi.Center.Y + maxDist) / cellSize))
			numCells := (gx1 - gx0 + 1) * (gy1 - gy0 + 1)
			if numCells < n {
				for gy := gy0; gy <= gy1; gy++ {
					for gx := gx0; gx <= gx1; gx++ {
						for _, idx := range grid[[2]int{gx, gy}] {
							if idx <= i {
								continue
							}
							if sorted[idx].Size < minSize {
								continue
							}
							if dist(pi.Center, sorted[idx].Center) <= maxDist {
								cands = append(cands, idx)
							}
						}
					}
				}
			}
		}
		if cands == nil {
			for j := i + 1; j < n; j++ {
				if sorted[j].Size < minSize {
					// sorted is descending by Size; later entries are even smaller.
					break
				}
				if dist(pi.Center, sorted[j].Center) <= maxDist {
					cands = append(cands, j)
				}
			}
		}
		if len(cands) > 24 {
			if n >= 60 {
				cands = selectDiverseCandidates(pi, sorted, cands, 4, 20)
			} else {
				cands = selectDiverseCandidates(pi, sorted, cands, 12, 16)
			}
		}

		// Precompute distances from pi to candidates for cheap pair filtering.
		dFromI := make([]float64, len(cands))
		for k, idx := range cands {
			dFromI[k] = dist(pi.Center, sorted[idx].Center)
		}
		denseScene := n >= 60
		maxLegFudge := legRatioMax * sizeRatioMax * 1.15

		for a := 0; a < len(cands)-1; a++ {
			pa := sorted[cands[a]]
			da := dFromI[a]
			for b := a + 1; b < len(cands); b++ {
				pb := sorted[cands[b]]
				if math.Max(pa.Size, pb.Size)/math.Min(pa.Size, pb.Size) > sizeRatioMax {
					continue
				}
				db := dFromI[b]
				// Quick triangle leg-ratio sanity check when pi->pa and pi->pb differ wildly
				// and pa->pb could not compensate.
				hi, lo := da, db
				if db > da {
					hi, lo = db, da
				}
				if hi > lo*maxLegFudge {
					dabSq := distSq(pa.Center, pb.Center)
					if dabSq < (hi/maxLegFudge)*(hi/maxLegFudge) {
						continue
					}
				}
				if set, ok := evaluateTriple(pi, pa, pb, legRatioMax, cosMax, relaxed, denseScene); ok {
					rawSets = append(rawSets, indexedSet{set: set, i0: i, i1: cands[a], i2: cands[b]})
				}
			}
		}
	}

	sort.Slice(rawSets, func(i, j int) bool {
		return rawSets[i].set.Score > rawSets[j].set.Score
	})

	effectiveMax := maxSets
	if effectiveMax > 0 && n*2 > effectiveMax {
		effectiveMax = n * 2
	}
	if effectiveMax > 400 {
		effectiveMax = 400
	}

	if effectiveMax > 0 && len(rawSets) > effectiveMax && n >= 60 {
		// In dense multi-QR grids, cross-QR triples can flood the top-N by score.
		// Prioritize triples covering previously unclaimed patterns so every real QR stays near the front.
		sets := make([]FinderPatternSet, 0, effectiveMax)
		picked := make([]bool, len(rawSets))
		used := make([]int, n)

		// Pass 0a: high-score disjoint triples (>= 1.02) so dense grids get their clean QRs
		// without being flooded by lower-score (0.55..1.01) disjoint background speckles.
		for k, item := range rawSets {
			if item.set.Score < 1.02 {
				break
			}
			if used[item.i0] == 0 && used[item.i1] == 0 && used[item.i2] == 0 {
				picked[k] = true
				used[item.i0]++
				used[item.i1]++
				used[item.i2]++
				sets = append(sets, item.set)
				if len(sets) >= effectiveMax {
					return sets
				}
			}
		}

		// Pass 1a: high-score triples (>= 1.02) with at most 1 already-claimed pattern.
		for k, item := range rawSets {
			if item.set.Score < 1.02 {
				break
			}
			if picked[k] {
				continue
			}
			uc := 0
			if used[item.i0] > 0 {
				uc++
			}
			if used[item.i1] > 0 {
				uc++
			}
			if used[item.i2] > 0 {
				uc++
			}
			if uc <= 1 {
				picked[k] = true
				used[item.i0]++
				used[item.i1]++
				used[item.i2]++
				sets = append(sets, item.set)
				if len(sets) >= effectiveMax {
					return sets
				}
			}
		}

		// Pass 0b: remaining disjoint triples so tilted/lower-score QRs in dense grids are still covered.
		for k, item := range rawSets {
			if picked[k] {
				continue
			}
			if used[item.i0] == 0 && used[item.i1] == 0 && used[item.i2] == 0 {
				picked[k] = true
				used[item.i0]++
				used[item.i1]++
				used[item.i2]++
				sets = append(sets, item.set)
				if len(sets) >= effectiveMax {
					return sets
				}
			}
		}

		// Pass 1: triples with at most 1 already-claimed pattern.
		for k, item := range rawSets {
			if picked[k] {
				continue
			}
			uc := 0
			if used[item.i0] > 0 {
				uc++
			}
			if used[item.i1] > 0 {
				uc++
			}
			if used[item.i2] > 0 {
				uc++
			}
			if uc <= 1 {
				picked[k] = true
				used[item.i0]++
				used[item.i1]++
				used[item.i2]++
				sets = append(sets, item.set)
				if len(sets) >= effectiveMax {
					return sets
				}
			}
		}

		// Pass 2: triples with at least 1 fresh pattern.
		for k, item := range rawSets {
			if picked[k] {
				continue
			}
			if used[item.i0] == 0 || used[item.i1] == 0 || used[item.i2] == 0 {
				picked[k] = true
				used[item.i0]++
				used[item.i1]++
				used[item.i2]++
				sets = append(sets, item.set)
				if len(sets) >= effectiveMax {
					return sets
				}
			}
		}

		// Pass 3: fill remaining budget in score order.
		for k, item := range rawSets {
			if picked[k] {
				continue
			}
			sets = append(sets, item.set)
			if len(sets) >= effectiveMax {
				break
			}
		}
		return sets
	}

	limit := len(rawSets)
	if effectiveMax > 0 && limit > effectiveMax {
		limit = effectiveMax
	}
	sets := make([]FinderPatternSet, limit)
	for i := 0; i < limit; i++ {
		sets[i] = rawSets[i].set
	}
	return sets
}

func selectDiverseCandidates(pi ConcentricPattern, sorted []ConcentricPattern, cands []int, topScoreN, nearestN int) []int {
	m := len(cands)
	out := make([]int, 0, topScoreN+nearestN)

	// Partial top-K by pattern quality (O(M * topScoreN), zero map allocations).
	if topScoreN > m {
		topScoreN = m
	}
	picked := make([]bool, m)
	for k := 0; k < topScoreN; k++ {
		bestPos := -1
		bestScore := -1.0
		for j, idx := range cands {
			if picked[j] {
				continue
			}
			p := sorted[idx]
			s := p.Score
			if p.HasCorners {
				s += 0.12
			}
			if s > bestScore {
				bestScore = s
				bestPos = j
			}
		}
		if bestPos < 0 {
			break
		}
		picked[bestPos] = true
		out = append(out, cands[bestPos])
	}

	// Partial nearest-K by distance from pi (O(M * nearestN)).
	dists := make([]float64, m)
	for j, idx := range cands {
		dists[j] = distSq(pi.Center, sorted[idx].Center)
	}
	for k := 0; k < nearestN; k++ {
		bestPos := -1
		bestDist := math.MaxFloat64
		for j := range cands {
			if picked[j] {
				continue
			}
			if dists[j] < bestDist {
				bestDist = dists[j]
				bestPos = j
			}
		}
		if bestPos < 0 {
			break
		}
		picked[bestPos] = true
		out = append(out, cands[bestPos])
	}
	return out
}

func evaluateTriple(p0, p1, p2 ConcentricPattern, legRatioMax, cosMax float64, relaxed bool, denseScene bool) (FinderPatternSet, bool) {
	pts := [3]ConcentricPattern{p0, p1, p2}
	var best FinderPatternSet
	found := false

	// Test each vertex as candidate TL corner.
	for tlIdx := 0; tlIdx < 3; tlIdx++ {
		tl := pts[tlIdx]
		c1 := pts[(tlIdx+1)%3]
		c2 := pts[(tlIdx+2)%3]

		// Perspective-compensated squared distances from TL.
		d1Sq := distSq(tl.Center, c1.Center)
		d2Sq := distSq(tl.Center, c2.Center)
		if d1Sq < 25.0 || d2Sq < 25.0 {
			continue
		}
		d1 := math.Sqrt(d1Sq)
		d2 := math.Sqrt(d2Sq)

		// Perspective-compensated distances: DistSq(a, b) * b.Size / a.Size
		comp1 := math.Sqrt(d1Sq * c1.Size / tl.Size)
		comp2 := math.Sqrt(d2Sq * c2.Size / tl.Size)
		legRatio := math.Max(comp1, comp2) / math.Min(comp1, comp2)
		if legRatio > legRatioMax {
			// Also allow raw distance ratio when sizes are noisy under relaxed mode.
			rawRatio := math.Max(d1, d2) / math.Min(d1, d2)
			if !relaxed || rawRatio > legRatioMax {
				continue
			}
			legRatio = rawRatio
		}

		// Angle cosine check at TL.
		v1x, v1y := c1.Center.X-tl.Center.X, c1.Center.Y-tl.Center.Y
		v2x, v2y := c2.Center.X-tl.Center.X, c2.Center.Y-tl.Center.Y
		cosTheta := (v1x*v2x + v1y*v2y) / (d1 * d2)
		if cosTheta < -cosMax || cosTheta > cosMax {
			continue
		}

		// Module count check: estimated dimension and hypotenuse in [18, 265].
		ms1 := math.Sqrt(tl.ModuleSize * c1.ModuleSize)
		ms2 := math.Sqrt(tl.ModuleSize * c2.ModuleSize)
		estDim1 := d1/ms1 + 7.0
		estDim2 := d2/ms2 + 7.0
		hyp := dist(c1.Center, c2.Center)
		msHyp := math.Sqrt(c1.ModuleSize * c2.ModuleSize)
		hypMods := hyp / msHyp

		if estDim1 < 18.0 || estDim1 > 265.0 || estDim2 < 18.0 || estDim2 > 265.0 {
			continue
		}
		if hypMods < 18.0 || hypMods > 265.0 {
			continue
		}

		// Canonical orientation: Cross(TR - TL, BL - TL) > 0 in screen coords (+X right, +Y down).
		tr, bl := c1, c2
		cross := (tr.Center.X-tl.Center.X)*(bl.Center.Y-tl.Center.Y) - (tr.Center.Y-tl.Center.Y)*(bl.Center.X-tl.Center.X)
		if cross < 0 {
			tr, bl = bl, tr
		} else if math.Abs(cross) < 1e-3 {
			continue
		}

		// Score candidate orientation.
		patScore := (tl.Score + tr.Score + bl.Score) / 3.0
		angleScore := 1.0 - math.Abs(cosTheta)
		ratioScore := 1.0 / legRatio
		dimDiff := math.Abs(estDim1-estDim2) / math.Max(estDim1, estDim2)
		dimScore := math.Max(0.0, 1.0-dimDiff*2.0)

		cornerAlign := 0.0
		if tl.HasCorners {
			cornerAlign = cornerAxisAlignment(tl.Corners, tr.Center, bl.Center, tl.Center)
		}
		cornersBonus := 0.0
		if tl.HasCorners {
			cornersBonus += 0.07
		}
		if tr.HasCorners {
			cornersBonus += 0.07
		}
		if bl.HasCorners {
			cornersBonus += 0.07
		}

		compactBonus := 0.0
		if denseScene && tl.ModuleSize >= 1.5 && tr.ModuleSize >= 1.5 && bl.ModuleSize >= 1.5 {
			avgDim := (estDim1 + estDim2) * 0.5
			compactBonus = 0.14 / (1.0 + avgDim/25.0)
		}
		noisePenalty := 0.0
		if denseScene && (tl.ModuleSize < 1.5 || tr.ModuleSize < 1.5 || bl.ModuleSize < 1.5) {
			noisePenalty = 0.10
		}

		totalScore := patScore*0.40 + angleScore*0.25 + ratioScore*0.20 + dimScore*0.15 + cornerAlign*0.15 + cornersBonus + compactBonus - noisePenalty
		if !found || totalScore > best.Score {
			best = FinderPatternSet{
				BL:    bl,
				TL:    tl,
				TR:    tr,
				Score: totalScore,
			}
			found = true
		}
	}

	return best, found
}

// cornerAxisAlignment measures how closely a finder pattern's 7x7 square edges align
// with the vectors toward TR and BL. Returns a bonus in [0, 1].
func cornerAxisAlignment(corners [4]geom.PointF, tr, bl, tl geom.PointF) float64 {
	u1 := normalize(geom.PointF{X: tr.X - tl.X, Y: tr.Y - tl.Y})
	u2 := normalize(geom.PointF{X: bl.X - tl.X, Y: bl.Y - tl.Y})
	var bestSum float64
	for i := 0; i < 4; i++ {
		e := normalize(geom.PointF{
			X: corners[(i+1)%4].X - corners[i].X,
			Y: corners[(i+1)%4].Y - corners[i].Y,
		})
		a1 := math.Abs(e.X*u1.X + e.Y*u1.Y)
		a2 := math.Abs(e.X*u2.X + e.Y*u2.Y)
		if a1 > bestSum {
			bestSum += a1 * 0.25
		}
		if a2 > bestSum {
			bestSum += a2 * 0.25
		}
	}
	if bestSum > 1.0 {
		bestSum = 1.0
	}
	return bestSum
}

// SynthesizeMissingFinderSets predicts a missing 3rd finder pattern from pairs of
// compatible detected finder patterns that both have extracted 7x7 outer corners.
// This recovers QR codes where one finder pattern is obscured by glare, shadows, or damage.
func SynthesizeMissingFinderSets(bm *imageproc.BitMatrix, patterns []ConcentricPattern) []FinderPatternSet {
	if bm == nil || len(patterns) < 2 {
		return nil
	}

	var withCorners []ConcentricPattern
	for _, p := range patterns {
		if p.HasCorners && p.ModuleSize > 0 {
			withCorners = append(withCorners, p)
		}
	}
	if len(withCorners) < 2 {
		return nil
	}

	var sets []FinderPatternSet
	maxPairs := 24
	if len(withCorners) > maxPairs {
		withCorners = withCorners[:maxPairs]
	}

	for i := 0; i < len(withCorners)-1; i++ {
		p1 := withCorners[i]
		for j := i + 1; j < len(withCorners); j++ {
			p2 := withCorners[j]

			sizeRatio := math.Max(p1.Size, p2.Size) / math.Min(p1.Size, p2.Size)
			if sizeRatio > 2.2 {
				continue
			}

			ms := math.Sqrt(p1.ModuleSize * p2.ModuleSize)
			d := dist(p1.Center, p2.Center)
			mods := d / ms
			if mods < 12.0 || mods > 245.0 {
				continue
			}

			// Map p2.Center into p1's local 7x7 coordinate frame ([0,7] x [0,7], center at (3.5, 3.5)).
			localToImg1 := geom.SquareToQuadrilateral(p1.Corners[0], p1.Corners[1], p1.Corners[2], p1.Corners[3])
			inv1 := geom.QuadrilateralToSquare(p1.Corners[0], p1.Corners[1], p1.Corners[2], p1.Corners[3])
			if !localToImg1.IsValid() || !inv1.IsValid() {
				continue
			}

			// In [0,1]^2 units of p1's 7x7 square, center is (0.5, 0.5) and 1 module is 1/7.
			q2 := inv1.ProjectPoint(p2.Center)
			du := (q2.X - 0.5) * 7.0
			dv := (q2.Y - 0.5) * 7.0
			absU, absV := math.Abs(du), math.Abs(dv)

			// Verify p2 also aligns reasonably with p1's local frame.
			inv2 := geom.QuadrilateralToSquare(p2.Corners[0], p2.Corners[1], p2.Corners[2], p2.Corners[3])
			if !inv2.IsValid() {
				continue
			}
			q1 := inv2.ProjectPoint(p1.Center)
			du2 := (q1.X - 0.5) * 7.0
			dv2 := (q1.Y - 0.5) * 7.0

			// Case 1: p1 and p2 are adjacent (one axis dominant in p1's frame).
			if math.Max(absU, absV) > 2.5*math.Min(absU, absV)+1.5 {
				var pred1, pred2 geom.PointF
				if absU >= absV {
					// p2 lies along p1's local U axis -> 3rd pattern lies along V axis (+absU or -absU).
					pred1 = localToImg1.Project(0.5, 0.5+absU/7.0)
					pred2 = localToImg1.Project(0.5, 0.5-absU/7.0)
				} else {
					// p2 lies along p1's local V axis -> 3rd pattern lies along U axis (+absV or -absV).
					pred1 = localToImg1.Project(0.5+absV/7.0, 0.5)
					pred2 = localToImg1.Project(0.5-absV/7.0, 0.5)
				}

				for _, cand := range [2]geom.PointF{pred1, pred2} {
					if !pointNearImage(bm, cand, ms*4.0) {
						continue
					}
					// Also ensure the candidate doesn't duplicate an existing detected pattern closely.
					if hasNearbyPattern(patterns, cand, ms*3.5) {
						continue
					}
					synth := ConcentricPattern{
						Center:     cand,
						Size:       p1.Size,
						ModuleSize: p1.ModuleSize,
						Score:      (p1.Score + p2.Score) * 0.35,
					}
					if set, ok := evaluateTriple(p1, p2, synth, 2.2, 0.55, false, false); ok {
						// Score candidate by timing/pattern dark pixel presence around cand.
						set.Score += scoreCandidateRegion(bm, cand, ms) * 0.25
						sets = append(sets, set)
					}
				}
			} else if math.Abs(absU-absV) < 0.45*math.Max(absU, absV) &&
				math.Abs(math.Abs(du2)-math.Abs(dv2)) < 0.45*math.Max(math.Abs(du2), math.Abs(dv2)) {
				// Case 2: p1 and p2 are diagonal (TR & BL), so TL is at (0.5 + du/7, 0.5) or (0.5, 0.5 + dv/7).
				candA := localToImg1.Project(q2.X, 0.5)
				candB := localToImg1.Project(0.5, q2.Y)
				for _, cand := range [2]geom.PointF{candA, candB} {
					if !pointNearImage(bm, cand, ms*4.0) || hasNearbyPattern(patterns, cand, ms*3.5) {
						continue
					}
					avgSize := (p1.Size + p2.Size) * 0.5
					synth := ConcentricPattern{
						Center:     cand,
						Size:       avgSize,
						ModuleSize: avgSize / 7.0,
						Score:      (p1.Score + p2.Score) * 0.35,
					}
					if set, ok := evaluateTriple(p1, p2, synth, 2.2, 0.55, false, false); ok {
						set.Score += scoreCandidateRegion(bm, cand, ms) * 0.25
						sets = append(sets, set)
					}
				}
			}
		}
	}

	sort.Slice(sets, func(i, j int) bool {
		return sets[i].Score > sets[j].Score
	})
	if len(sets) > 8 {
		sets = sets[:8]
	}
	return sets
}

func pointNearImage(bm *imageproc.BitMatrix, p geom.PointF, margin float64) bool {
	return p.X >= -margin && p.Y >= -margin && p.X <= float64(bm.Width)+margin && p.Y <= float64(bm.Height)+margin
}

func hasNearbyPattern(patterns []ConcentricPattern, pt geom.PointF, radius float64) bool {
	rSq := radius * radius
	for _, p := range patterns {
		if distSq(p.Center, pt) <= rSq {
			return true
		}
	}
	return false
}

// scoreCandidateRegion estimates how plausible a missing finder pattern center is
// by sampling black pixel fraction in a 7x7 module window (expected ~24/49 ≈ 0.49).
func scoreCandidateRegion(bm *imageproc.BitMatrix, c geom.PointF, ms float64) float64 {
	half := 3.5 * ms
	step := math.Max(1.0, ms*0.75)
	total, black := 0, 0
	for dy := -half; dy <= half; dy += step {
		for dx := -half; dx <= half; dx += step {
			ix := int(math.Round(c.X + dx))
			iy := int(math.Round(c.Y + dy))
			if inBounds(bm, ix, iy) {
				total++
				if bm.Get(ix, iy) {
					black++
				}
			}
		}
	}
	if total == 0 {
		return 0
	}
	frac := float64(black) / float64(total)
	// Finder pattern has 33 black modules out of 49 (~0.673) or 25+8=33.
	diff := math.Abs(frac - 0.65)
	return math.Max(0, 1.0-diff*2.0)
}

func normalize(v geom.PointF) geom.PointF {
	l := math.Hypot(v.X, v.Y)
	if l < 1e-9 {
		return geom.PointF{}
	}
	return geom.PointF{X: v.X / l, Y: v.Y / l}
}
