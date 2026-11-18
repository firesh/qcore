package detector

import (
	"image"
	"math"
	"runtime"
	"sync"
	"sync/atomic"

	"github.com/firesh/qcore/internal/decoder"
	"github.com/firesh/qcore/internal/geom"
	"github.com/firesh/qcore/internal/imageproc"
)

// Result holds a detected and optionally decoded QR code symbol.
type Result struct {
	Text        string
	RawBytes    []byte
	Quad        geom.Quad
	Version     int
	ECLevel     string
	Orientation float64 // Degrees in (-180, 180] along TL -> TR edge
	Decoded     bool
}

// Options configures detection and decoding behavior.
type Options struct {
	Multi                      bool
	PureBarcode                bool
	MaxResults                 int
	ReturnUnreadableDetections bool
}

// DetectAndDecode runs a fast-exit multi-tier QR detection and decoding cascade
// on a single-channel grayscale image.
func DetectAndDecode(gray *imageproc.GrayImage, opts Options) []Result {
	if gray == nil || gray.Width <= 0 || gray.Height <= 0 {
		return nil
	}

	if opts.PureBarcode {
		// Pad with a quiet zone so edge-touching pure barcodes still trigger finder detection.
		pad := max(12, min(gray.Width, gray.Height)/8)
		padded := padWhiteBorder(gray, pad)
		res := runCascade(padded, opts, float64(pad))
		if len(res) > 0 {
			return res
		}
	}

	results := runCascade(gray, opts, 0)
	if hasDecoded(results) {
		return results
	}
	// A uniform frame cannot contain a finder.
	varied := false
	for _, v := range gray.Pix {
		if v != gray.Pix[0] {
			varied = true
			break
		}
	}
	if !varied {
		return results
	}
	bm := imageproc.GlobalOtsuBinarizer(gray)
	for _, inverted := range []bool{false, true} {
		if inverted {
			bm.Invert()
		}
		pats := scanFinderPatterns(bm, false, false, true)
		if len(pats) < 3 {
			continue
		}
		g := gray
		if inverted {
			g = gray.Invert()
		}
		dec, _, _, _, _, _, _ := processPatternsOnBitMatrixDetail(g, bm, nil, pats, 1, 1, 0, opts, nil, false, false, nil, true)
		if hasDecoded(dec) {
			return trimResults(dec, opts)
		}
	}
	return results
}

// DetectAndDecodeImage converts img to luminance before detection.
func DetectAndDecodeImage(img image.Image, opts Options) []Result {
	if img == nil {
		return nil
	}
	return trimResults(DetectAndDecode(imageproc.ExtractChannel(img, imageproc.ChannelLuma), opts), opts)
}

func runCascade(gray *imageproc.GrayImage, opts Options, padOffset float64) []Result {
	var results []Result
	var unreadable []Result

	w, h := gray.Width, gray.Height
	integ := imageproc.NewIntegralImage(gray)
	allDone := false

	done := func() bool {
		if allDone {
			return true
		}
		if !opts.Multi && hasDecoded(results) {
			return true
		}
		if opts.MaxResults > 0 && countDecoded(results) >= opts.MaxResults {
			return true
		}
		return false
	}

	maxPatFound := 0
	minMSFound := 1e9
	maxMSFound := 0.0
	decodedMinMS := 1e9
	recordDecoded := func(dec []Result) {
		for _, r := range dec {
			if !r.Decoded || r.Version < 1 {
				continue
			}
			dim := float64(17 + 4*r.Version)
			perim := r.Quad[0].Dist(r.Quad[1]) + r.Quad[1].Dist(r.Quad[2]) + r.Quad[2].Dist(r.Quad[3]) + r.Quad[3].Dist(r.Quad[0])
			ms := (perim * 0.25) / dim
			if ms > 0 && ms < decodedMinMS {
				decodedMinMS = ms
			}
		}
	}
	tryPass := func(g *imageproc.GrayImage, ii *imageproc.IntegralImage, bm *imageproc.BitMatrix, sx, sy float64, allowRelaxed, allowTier4, tryHarderScan bool) int {
		if bm == nil || done() {
			return 0
		}
		if ii == nil && bm == nil && g != nil {
			ii = imageproc.NewIntegralImage(g)
		}
		allowTiny := tryHarderScan && sx >= 0.99
		pats := findFinderPatterns(bm, tryHarderScan, allowTiny)
		dec, un, rem, patCount, minMS, maxMS, _ := processPatternsOnBitMatrixDetail(g, bm, ii, pats, sx, sy, padOffset, opts, results, allowRelaxed, allowTier4, nil, true)
		if patCount > maxPatFound {
			maxPatFound = patCount
		}
		if minMS > 0 && minMS < minMSFound {
			minMSFound = minMS
		}
		if maxMS > maxMSFound {
			maxMSFound = maxMS
		}
		recordDecoded(dec)
		mergeResults(&results, dec, opts)
		unreadable = append(unreadable, un...)
		return rem
	}

	// Pass 1: Fast path — HybridBinarizer(24) at 1x.
	bm1 := imageproc.HybridBinarizer(gray, 24)
	var bm1Pats []ConcentricPattern
	pats1 := FindFinderPatterns(bm1, false)
	bm1Pats = pats1
	rem := 0
	{
		// Try deformation recovery on the initial, bounded set of full-resolution
		// candidates before spending work on additional scans.
		dec, un, r, patC, minMS, maxMS, _ := processPatternsOnBitMatrixDetail(gray, bm1, integ, pats1, 1.0, 1.0, padOffset, opts, results, false, true, nil, true)
		if patC > maxPatFound {
			maxPatFound = patC
		}
		if minMS > 0 && minMS < minMSFound {
			minMSFound = minMS
		}
		if maxMS > maxMSFound {
			maxMSFound = maxMS
		}
		recordDecoded(dec)
		mergeResults(&results, dec, opts)
		unreadable = append(unreadable, un...)
		rem = r
	}
	if done() {
		return trimResults(results, opts)
	}
	if countDecoded(results) >= 1 && rem <= 1 {
		// In Multi mode, if only 1 QR decoded so far and skip > 1, verify with a quick skip=1
		// scan on bm1 so small secondary QRs (e.g. F03_multi_sizes) are never missed.
		if opts.Multi && countDecoded(results) == 1 && bm1.Height >= 400 {
			pats1H := findFinderPatterns(bm1, true, max(w, h) < 2400)
			bm1Pats = pats1H
			dec, un, r, patC, minMS, maxMS, _ := processPatternsOnBitMatrixDetail(gray, bm1, integ, pats1H, 1.0, 1.0, padOffset, opts, results, false, false, nil, true)
			if patC > maxPatFound {
				maxPatFound = patC
			}
			if minMS > 0 && minMS < minMSFound {
				minMSFound = minMS
			}
			if maxMS > maxMSFound {
				maxMSFound = maxMS
			}
			recordDecoded(dec)
			mergeResults(&results, dec, opts)
			unreadable = append(unreadable, un...)
			if r >= 2 && !hasPlausibleOutsidePair(pats1H, 1.0, 1.0, padOffset, results) {
				r = 0
			}
			rem = r
		}
		if rem <= 1 {
			return trimResults(results, opts)
		}
	}
	bm1NeedsFullRes := func() bool {
		if !opts.Multi || countDecoded(results) >= 8 || minMSFound >= 6.0 {
			return false
		}
		return hasPlausibleOutsidePair(bm1Pats, 1.0, 1.0, padOffset, results)
	}

	// Fast Downscale Pyramid for large images (BoofCV close, monitor moire, high_version, blurred).
	var down50 *imageproc.GrayImage
	var integ50 *imageproc.IntegralImage
	var sx50, sy50 float64
	if max(w, h) >= 1400 {
		down50 = imageproc.DownscaleArea(gray, 0.5)
		sx50 = float64(down50.Width) / float64(w)
		sy50 = float64(down50.Height) / float64(h)
		integ50 = imageproc.NewIntegralImage(down50)
		bmDown50 := imageproc.HybridBinarizer(down50, 20)
		rem50 := tryPass(down50, integ50, bmDown50, sx50, sy50, true, false, false)
		if done() || (countDecoded(results) >= 1 && rem50 <= 1 && !bm1NeedsFullRes()) {
			return trimResults(results, opts)
		}

		if max(w, h) >= 2200 && (minMSFound > 1e8 || maxMSFound >= 7.0) {
			down25 := imageproc.DownscaleArea(gray, 0.25)
			sx25 := float64(down25.Width) / float64(w)
			sy25 := float64(down25.Height) / float64(h)
			integ25 := imageproc.NewIntegralImage(down25)
			bmDown25 := imageproc.HybridBinarizer(down25, 20)
			rem25 := tryPass(down25, integ25, bmDown25, sx25, sy25, true, false, false)
			if done() || (countDecoded(results) >= 1 && rem25 <= 1 && !bm1NeedsFullRes()) {
				return trimResults(results, opts)
			}
		}
	}

	// Pass 2: Low-contrast & strong shadow recovery (AdaptiveMean + HybridBinarizer(6) + Inverted),
	// executed concurrently across workers against a read-only snapshot of results.
	type cascadeTaskOut struct {
		dec      []Result
		un       []Result
		rem      int
		patCount int
		minMS    float64
		maxMS    float64
		remPts   []geom.PointF
	}
	runTask := func(g *imageproc.GrayImage, bm *imageproc.BitMatrix, ii *imageproc.IntegralImage, sx, sy float64, allowRelaxed, allowTier4, tryHarderScan bool, snap []Result, abort *atomic.Bool, parallelSets bool) cascadeTaskOut {
		d, u, r, pc, mms, mxms, rpts := processBitMatrixWithAbortDetail(g, bm, ii, sx, sy, padOffset, opts, snap, allowRelaxed, allowTier4, tryHarderScan, tryHarderScan, abort, parallelSets)
		return cascadeTaskOut{d, u, r, pc, mms, mxms, rpts}
	}
	runWavefront := func(tasks []func(snap []Result, abort *atomic.Bool) cascadeTaskOut) int {
		snap := append([]Result(nil), results...)
		outs := make([]cascadeTaskOut, len(tasks))
		var stopFlag *atomic.Bool
		if !opts.Multi || opts.MaxResults == 1 {
			stopFlag = &atomic.Bool{}
		}
		if len(tasks) > 1 && runtime.GOMAXPROCS(0) > 1 {
			var wg sync.WaitGroup
			wg.Add(len(tasks))
			for i, fn := range tasks {
				go func(idx int, f func([]Result, *atomic.Bool) cascadeTaskOut) {
					defer wg.Done()
					if stopFlag != nil && stopFlag.Load() {
						outs[idx] = cascadeTaskOut{rem: math.MaxInt32}
						return
					}
					out := f(snap, stopFlag)
					if stopFlag != nil && countDecoded(out.dec) > 0 {
						stopFlag.Store(true)
					}
					outs[idx] = out
				}(i, fn)
			}
			wg.Wait()
		} else {
			for i, fn := range tasks {
				out := fn(snap, stopFlag)
				outs[i] = out
				if stopFlag != nil && countDecoded(out.dec) > 0 {
					break
				}
			}
		}
		for _, o := range outs {
			if o.patCount > maxPatFound {
				maxPatFound = o.patCount
			}
			if o.minMS > 0 && o.minMS < minMSFound {
				minMSFound = o.minMS
			}
			if o.maxMS > maxMSFound {
				maxMSFound = o.maxMS
			}
			recordDecoded(o.dec)
			mergeResults(&results, o.dec, opts)
			unreadable = append(unreadable, o.un...)
		}
		minRem := math.MaxInt32
		for _, o := range outs {
			if o.rem == math.MaxInt32 {
				continue
			}
			rOut := o.rem
			if countDecoded(o.dec) > 0 {
				actualRem := 0
				for _, pt := range o.remPts {
					if !pointInsideAnyDecoded(pt, results) {
						actualRem++
					}
				}
				rOut = actualRem
			}
			if rOut < minRem {
				minRem = rOut
			}
		}
		if minRem == math.MaxInt32 {
			return rem
		}
		return minRem
	}

	msForHelperDown50 := func() float64 {
		if decodedMinMS <= 1e8 {
			return decodedMinMS
		}
		return minMSFound
	}
	win := clampInt(min(w, h)/10, 31, 181) | 1
	p2Tasks := []func([]Result, *atomic.Bool) cascadeTaskOut{
		func(snap []Result, abort *atomic.Bool) cascadeTaskOut {
			bm := imageproc.AdaptiveMeanBinarizer(gray, integ, win, 0.06)
			d, u, r, pc, mms, mxms, rpts := processBitMatrixWithAbortDetail(gray, bm, integ, 1.0, 1.0, padOffset, opts, snap, true, false, false, false, abort, true)
			return cascadeTaskOut{d, u, r, pc, mms, mxms, rpts}
		},
		func(snap []Result, abort *atomic.Bool) cascadeTaskOut {
			aGray, aSX, aSY, aInteg, aWin := gray, 1.0, 1.0, integ, win
			if max(w, h) >= 2400 && down50 != nil && (msForHelperDown50() > 1e8 || msForHelperDown50() >= 8.0) {
				aGray, aSX, aSY, aInteg = down50, sx50, sy50, integ50
				aWin = clampInt(min(down50.Width, down50.Height)/10, 31, 151) | 1
			}
			bm := imageproc.AdaptiveMeanBinarizer(aGray, aInteg, aWin, 0.02)
			return runTask(aGray, bm, aInteg, aSX, aSY, true, false, false, snap, abort, true)
		},
		func(snap []Result, abort *atomic.Bool) cascadeTaskOut {
			hGray, hSX, hSY, hInteg := gray, 1.0, 1.0, integ
			if max(w, h) >= 2400 && down50 != nil && (msForHelperDown50() > 1e8 || msForHelperDown50() >= 8.0) {
				hGray, hSX, hSY, hInteg = down50, sx50, sy50, integ50
			}
			bm := imageproc.HybridBinarizer(hGray, 6)
			return runTask(hGray, bm, hInteg, hSX, hSY, true, false, false, snap, abort, true)
		},
	}
	if opts.Multi {
		p2Tasks = append(p2Tasks, func(snap []Result, abort *atomic.Bool) cascadeTaskOut {
			invGray := gray.Invert()
			bmInv := imageproc.HybridBinarizer(invGray, 16)
			return runTask(invGray, bmInv, nil, 1.0, 1.0, true, false, false, snap, abort, true)
		})
	}
	rem = runWavefront(p2Tasks)
	if done() || (countDecoded(results) >= 1 && rem <= 1) {
		return trimResults(results, opts)
	}

	if !opts.Multi {
		if opts.ReturnUnreadableDetections && len(results) == 0 && len(unreadable) > 0 {
			return dedupUnreadable(unreadable, opts)
		}
		return trimResults(results, opts)
	}

	// Check whether the remaining finder patterns warrant upscaling.
	needUpscale := min(w, h) <= 350 || (max(w, h) <= 1800 && !(maxPatFound >= 100 && minMSFound <= 1.25 && countDecoded(results) == 0) && (maxPatFound < 3 || minMSFound < 3.2 || (opts.Multi && countDecoded(results) > 0 && rem >= 2 && minMSFound < 4.5)))

	// Pass 3: Upscale pyramid for tiny / downscaled images (D10_downscale_15pct, A06_tiny_qr).
	if (opts.Multi) && needUpscale {
		var upscaleFactors []float64
		if min(w, h) <= 350 {
			upscaleFactors = []float64{3.0, 4.0, 2.0}
		} else {
			upscaleFactors = []float64{2.0}
		}
		for _, scale := range upscaleFactors {
			if max(w, h)*int(scale) > 3600 {
				continue
			}
			up := imageproc.UpscaleBicubic(gray, scale)
			upSharp := imageproc.UnsharpMask(up, 1.5)
			sx := float64(upSharp.Width) / float64(w)
			sy := float64(upSharp.Height) / float64(h)
			upInteg := imageproc.NewIntegralImage(upSharp)
			bmUp := imageproc.HybridBinarizer(upSharp, 10)
			rem = tryPass(upSharp, upInteg, bmUp, sx, sy, true, false, true)
			if done() || (countDecoded(results) >= 1 && rem <= 1) {
				return trimResults(results, opts)
			}
			upWin := clampInt(min(upSharp.Width, upSharp.Height)/12, 25, 91) | 1
			bmUpMean := imageproc.AdaptiveMeanBinarizer(upSharp, upInteg, upWin, 0.05)
			rem = tryPass(upSharp, upInteg, bmUpMean, sx, sy, true, false, true)
			if done() || (countDecoded(results) >= 1 && rem <= 1) {
				return trimResults(results, opts)
			}
		}
	}

	if len(results) == 0 && opts.ReturnUnreadableDetections && len(unreadable) > 0 {
		return dedupUnreadable(unreadable, opts)
	}
	if opts.ReturnUnreadableDetections && opts.Multi {
		for _, u := range dedupUnreadable(unreadable, opts) {
			if !overlapsAnyDecoded(u.Quad, results) {
				results = append(results, u)
			}
		}
	}

	return trimResults(results, opts)
}

func processBitMatrix(
	gray *imageproc.GrayImage,
	bm *imageproc.BitMatrix,
	integ *imageproc.IntegralImage,
	scaleX, scaleY, padOffset float64,
	opts Options,
	existing []Result,
	allowRelaxed, allowTier4, tryHarderScan bool,
) (decoded []Result, unreadable []Result, remainingPatterns int, totalPatterns int, minMS float64) {
	return processBitMatrixWithAbort(gray, bm, integ, scaleX, scaleY, padOffset, opts, existing, allowRelaxed, allowTier4, tryHarderScan, nil, true)
}

func processBitMatrixWithAbort(
	gray *imageproc.GrayImage,
	bm *imageproc.BitMatrix,
	integ *imageproc.IntegralImage,
	scaleX, scaleY, padOffset float64,
	opts Options,
	existing []Result,
	allowRelaxed, allowTier4, tryHarderScan bool,
	abort *atomic.Bool,
	parallelSets bool,
) (decoded []Result, unreadable []Result, remainingPatterns int, totalPatterns int, minMS float64) {
	decoded, unreadable, remainingPatterns, totalPatterns, minMS, _, _ = processBitMatrixWithAbortDetail(gray, bm, integ, scaleX, scaleY, padOffset, opts, existing, allowRelaxed, allowTier4, tryHarderScan, tryHarderScan, abort, parallelSets)
	return
}

func processBitMatrixWithAbortDetail(
	gray *imageproc.GrayImage,
	bm *imageproc.BitMatrix,
	integ *imageproc.IntegralImage,
	scaleX, scaleY, padOffset float64,
	opts Options,
	existing []Result,
	allowRelaxed, allowTier4, tryHarderScan, allowTiny bool,
	abort *atomic.Bool,
	parallelSets bool,
) (decoded []Result, unreadable []Result, remainingPatterns int, totalPatterns int, minMS, maxMS float64, remOrig []geom.PointF) {
	if abort != nil && abort.Load() {
		return nil, nil, 0, 0, 0, 0, nil
	}
	patterns := findFinderPatterns(bm, tryHarderScan, allowTiny)
	return processPatternsOnBitMatrixDetail(gray, bm, integ, patterns, scaleX, scaleY, padOffset, opts, existing, allowRelaxed, allowTier4, abort, parallelSets)
}

func processPatternsOnBitMatrix(
	gray *imageproc.GrayImage,
	bm *imageproc.BitMatrix,
	integ *imageproc.IntegralImage,
	patterns []ConcentricPattern,
	scaleX, scaleY, padOffset float64,
	opts Options,
	existing []Result,
	allowRelaxed, allowTier4 bool,
) (decoded []Result, unreadable []Result, remainingPatterns int, totalPatterns int, minMS float64) {
	return processPatternsOnBitMatrixWithAbort(gray, bm, integ, patterns, scaleX, scaleY, padOffset, opts, existing, allowRelaxed, allowTier4, nil, true)
}

func processPatternsOnBitMatrixWithAbort(
	gray *imageproc.GrayImage,
	bm *imageproc.BitMatrix,
	integ *imageproc.IntegralImage,
	patterns []ConcentricPattern,
	scaleX, scaleY, padOffset float64,
	opts Options,
	existing []Result,
	allowRelaxed, allowTier4 bool,
	abort *atomic.Bool,
	parallelSets bool,
) (decoded []Result, unreadable []Result, remainingPatterns int, totalPatterns int, minMS float64) {
	decoded, unreadable, remainingPatterns, totalPatterns, minMS, _, _ = processPatternsOnBitMatrixDetail(gray, bm, integ, patterns, scaleX, scaleY, padOffset, opts, existing, allowRelaxed, allowTier4, abort, parallelSets)
	return
}

func processPatternsOnBitMatrixDetail(
	gray *imageproc.GrayImage,
	bm *imageproc.BitMatrix,
	integ *imageproc.IntegralImage,
	patterns []ConcentricPattern,
	scaleX, scaleY, padOffset float64,
	opts Options,
	existing []Result,
	allowRelaxed, allowTier4 bool,
	abort *atomic.Bool,
	parallelSets bool,
) (decoded []Result, unreadable []Result, remainingPatterns int, totalPatterns int, minMS, maxMS float64, remOrig []geom.PointF) {
	totalPatterns = len(patterns)
	if totalPatterns == 0 {
		return nil, nil, 0, 0, 0, 0, nil
	}

	minMS = patterns[0].ModuleSize
	maxMS = patterns[0].ModuleSize
	topCheck := totalPatterns
	if topCheck > 6 {
		topCheck = 6
	}
	for _, p := range patterns[1:topCheck] {
		if p.ModuleSize < minMS {
			minMS = p.ModuleSize
		}
		if p.ModuleSize > maxMS {
			maxMS = p.ModuleSize
		}
	}
	if scaleX > 0 {
		minMS /= scaleX
		maxMS /= scaleX
	}

	// Filter out finder patterns already inside decoded QR quads (for Multi mode speedup).
	if len(existing) > 0 {
		filtered := patterns[:0]
		for _, p := range patterns {
			origPt := mapPointToOrig(p.Center, scaleX, scaleY, padOffset)
			if !pointInsideAnyDecoded(origPt, existing) {
				filtered = append(filtered, p)
			}
		}
		patterns = filtered
	}
	if len(patterns) < 2 {
		for _, p := range patterns {
			remOrig = append(remOrig, mapPointToOrig(p.Center, scaleX, scaleY, padOffset))
		}
		return nil, nil, len(patterns), totalPatterns, minMS, maxMS, remOrig
	}

	type tier12Out struct {
		res  *decoder.DecodeResult
		quad geom.Quad
		ctx  planarContext
		ok   bool
	}

	evalTier12 := func(set FinderPatternSet) tier12Out {
		res, quad, ctx, ok := SamplePlanar(gray, bm, integ, set)
		if ok {
			return tier12Out{res: res, quad: quad, ctx: ctx, ok: true}
		}
		if ctx.dim >= 25 && ctx.baseH.IsValid() && ctx.plausible {
			if res2, quad2, ok2 := sampleTiledWithAlign(gray, bm, integ, set, ctx.dim, ctx.baseH, ctx.brAlign, ctx.brAlignFound); ok2 {
				return tier12Out{res: res2, quad: quad2, ctx: ctx, ok: true}
			}
		}
		return tier12Out{ctx: ctx, ok: false}
	}

	workers := runtime.GOMAXPROCS(0)
	if workers > 6 {
		workers = 6
	}
	maxTier3 := 3
	if opts.Multi {
		maxTier3 = 6
	}
	maxTier4 := 2
	if len(existing) > 0 {
		maxTier4 = 1
	}
	tier3Tried := 0
	tier4Tried := 0

	for peelRound := 0; peelRound < 4; peelRound++ {
		decodedBefore := len(decoded)

		strictLimit := 64
		if opts.Multi {
			strictLimit = len(patterns) * 2
			if strictLimit < 120 {
				strictLimit = 120
			}
			if strictLimit > 400 {
				strictLimit = 400
			}
		}
		sets := GenerateFinderPatternSets(patterns, strictLimit, false)
		if allowRelaxed && peelRound == 0 && (!opts.Multi || len(patterns) < 60 || len(sets) < len(patterns)/3) {
			relaxedLimit := 48
			if opts.Multi {
				relaxedLimit = 96
			}
			relaxedSets := GenerateFinderPatternSets(patterns, relaxedLimit, true)
			sets = appendFinderSetsDedup(sets, relaxedSets)
		}
		if peelRound == 0 && len(sets) == 0 {
			synthSets := SynthesizeMissingFinderSets(bm, patterns)
			sets = appendFinderSetsDedup(sets, synthSets)
		}

		maxRunSets := 10
		if opts.Multi {
			maxRunSets = len(patterns) * 2
			if maxRunSets < 120 {
				maxRunSets = 120
			}
			if maxRunSets > 400 {
				maxRunSets = 400
			}
			if allowRelaxed && len(patterns) > 100 {
				capNoise := 96
				if countDecoded(existing) > 0 {
					capNoise = 72
				}
				if maxRunSets > capNoise {
					maxRunSets = capNoise
				}
			}
		}
		if len(sets) > maxRunSets {
			sets = sets[:maxRunSets]
		}
		// Never evaluate hundreds of low-score speckle sets in a single pass; allow up to 400
		// only while high-confidence sets remain (such as dense Multi-QR grids).
		if opts.Multi && len(sets) > 36 {
			cutoff := len(sets)
			lowCount := 0
			maxLow := 48
			minKeepScore := 0.90
			if allowRelaxed {
				maxLow = 36
				minKeepScore = 0.95
			}
			for i, s := range sets {
				if s.Score < minKeepScore {
					lowCount++
					if lowCount > maxLow {
						cutoff = i
						break
					}
				}
			}
			sets = sets[:cutoff]
		}

		const batchSize = 24
		for start := 0; start < len(sets); {
			if abort != nil && abort.Load() {
				break
			}
			if !opts.Multi && len(decoded) > 0 {
				break
			}
			if opts.MaxResults > 0 && countDecoded(existing)+len(decoded) >= opts.MaxResults {
				break
			}

			// In Multi mode with several candidate sets, pre-evaluate Tier 1+2 in parallel batches
			// after filtering sets that already overlap a decoded QR.
			if parallelSets && opts.Multi && workers > 1 && len(sets)-start >= 6 {
				end := start + batchSize
				if end > len(sets) {
					end = len(sets)
				}
				activeIdx := make([]int, 0, end-start)
				estQuads := make([]geom.Quad, end-start)
				for i := start; i < end; i++ {
					eq := mapQuadToOrig(estimateFinderSetQuad(sets[i]), scaleX, scaleY, padOffset)
					estQuads[i-start] = eq
					if !overlapsAnyDecoded(eq, existing) && !overlapsAnyDecoded(eq, decoded) {
						activeIdx = append(activeIdx, i)
					}
				}
				outcomes := make([]tier12Out, end-start)
				if len(activeIdx) >= 2 {
					var nextTask int
					var mu sync.Mutex
					numW := workers
					if numW > len(activeIdx) {
						numW = len(activeIdx)
					}
					var wg sync.WaitGroup
					wg.Add(numW)
					for w := 0; w < numW; w++ {
						go func() {
							defer wg.Done()
							for {
								mu.Lock()
								k := nextTask
								nextTask++
								mu.Unlock()
								if k >= len(activeIdx) {
									return
								}
								idx := activeIdx[k]
								outcomes[idx-start] = evalTier12(sets[idx])
							}
						}()
					}
					wg.Wait()
				} else if len(activeIdx) == 1 {
					idx := activeIdx[0]
					outcomes[idx-start] = evalTier12(sets[idx])
				}

				for i := start; i < end; i++ {
					if opts.MaxResults > 0 && countDecoded(existing)+len(decoded) >= opts.MaxResults {
						break
					}
					estQuad := estQuads[i-start]
					if overlapsAnyDecoded(estQuad, existing) || overlapsAnyDecoded(estQuad, decoded) {
						continue
					}
					out := outcomes[i-start]
					if out.ok {
						r := makeDecodedResult(out.res, mapQuadToOrig(out.quad, scaleX, scaleY, padOffset))
						mergeResults(&decoded, []Result{r}, opts)
						continue
					}
					set := sets[i]
					ctx := out.ctx
					// Tier 3: CoonsPatchGrid curved boundary sampling.
					if ctx.dim >= 21 && ctx.baseH.IsValid() && (ctx.plausible || set.Score >= 0.88) && set.Score >= 0.72 && tier3Tried < maxTier3 {
						tier3Tried++
						if res, quad, ok := SampleCurvedCoons(gray, bm, integ, set, ctx.dim, ctx.baseH); ok {
							r := makeDecodedResult(res, mapQuadToOrig(quad, scaleX, scaleY, padOffset))
							mergeResults(&decoded, []Result{r}, opts)
							continue
						}
					}
					// Tier 4: Parametric deformation unwarping (radial, cylindrical, wave, BR nudge).
					if allowTier4 && ctx.dim >= 21 && ctx.plausible && set.Score >= 0.78 && tier4Tried < maxTier4 {
						tier4Tried++
						if res, quad, ok := SampleParametricDeform(gray, bm, integ, set, ctx.dim, ctx.baseH, ctx.brAlignFound); ok {
							r := makeDecodedResult(res, mapQuadToOrig(quad, scaleX, scaleY, padOffset))
							mergeResults(&decoded, []Result{r}, opts)
							continue
						}
					}
					if opts.ReturnUnreadableDetections {
						unreadable = append(unreadable, Result{
							Quad:        estQuad,
							Orientation: quadOrientationDeg(estQuad),
							Decoded:     false,
						})
					}
				}
				start = end
				continue
			}

			set := sets[start]
			start++

			// Quick spatial overlap check against already-decoded results.
			estQuad := mapQuadToOrig(estimateFinderSetQuad(set), scaleX, scaleY, padOffset)
			if overlapsAnyDecoded(estQuad, existing) || overlapsAnyDecoded(estQuad, decoded) {
				continue
			}

			// Tier 1 + 2: Planar homography, then BilinearTileGrid over alignment centers (version >= 2).
			out := evalTier12(set)
			if out.ok {
				r := makeDecodedResult(out.res, mapQuadToOrig(out.quad, scaleX, scaleY, padOffset))
				mergeResults(&decoded, []Result{r}, opts)
				continue
			}
			ctx := out.ctx

			// Tier 3: CoonsPatchGrid curved boundary sampling.
			if ctx.dim >= 21 && ctx.baseH.IsValid() && (ctx.plausible || set.Score >= 0.88) && set.Score >= 0.72 && tier3Tried < maxTier3 {
				tier3Tried++
				if res, quad, ok := SampleCurvedCoons(gray, bm, integ, set, ctx.dim, ctx.baseH); ok {
					r := makeDecodedResult(res, mapQuadToOrig(quad, scaleX, scaleY, padOffset))
					mergeResults(&decoded, []Result{r}, opts)
					continue
				}
			}

			// Tier 4: Parametric deformation unwarping (radial, cylindrical, wave, BR nudge).
			if allowTier4 && ctx.dim >= 21 && ctx.plausible && set.Score >= 0.78 && tier4Tried < maxTier4 {
				tier4Tried++
				if res, quad, ok := SampleParametricDeform(gray, bm, integ, set, ctx.dim, ctx.baseH, ctx.brAlignFound); ok {
					r := makeDecodedResult(res, mapQuadToOrig(quad, scaleX, scaleY, padOffset))
					mergeResults(&decoded, []Result{r}, opts)
					continue
				}
			}

			if opts.ReturnUnreadableDetections {
				unreadable = append(unreadable, Result{
					Quad:        estQuad,
					Orientation: quadOrientationDeg(estQuad),
					Decoded:     false,
				})
			}
		}

		// In dense Multi-QR mode, peel off patterns belonging to newly decoded symbols
		// and re-group remaining patterns so cross-QR lattice triples cannot hide real QRs.
		if !opts.Multi || len(decoded) == decodedBefore || len(patterns) < 24 ||
			(opts.MaxResults > 0 && countDecoded(existing)+len(decoded) >= opts.MaxResults) {
			break
		}
		beforePats := len(patterns)
		remPats := patterns[:0]
		for _, p := range patterns {
			origPt := mapPointToOrig(p.Center, scaleX, scaleY, padOffset)
			if !pointInsideAnyDecoded(origPt, decoded) {
				remPats = append(remPats, p)
			}
		}
		patterns = remPats
		if len(patterns) < 3 || len(patterns) >= beforePats {
			break
		}
		unreadable = nil
	}

	var remPats []ConcentricPattern
	for _, p := range patterns {
		origPt := mapPointToOrig(p.Center, scaleX, scaleY, padOffset)
		if !pointInsideAnyDecoded(origPt, existing) && !pointInsideAnyDecoded(origPt, decoded) {
			remOrig = append(remOrig, origPt)
			remPats = append(remPats, p)
		}
	}
	remCount := len(remOrig)
	if remCount >= 2 && !hasPlausibleOutsidePair(remPats, scaleX, scaleY, padOffset, nil) {
		remCount = 0
		remOrig = nil
	}

	return decoded, unreadable, remCount, totalPatterns, minMS, maxMS, remOrig
}

func hasPlausibleOutsidePair(pats []ConcentricPattern, scaleX, scaleY, padOffset float64, results []Result) bool {
	var outPts []geom.PointF
	var outMS []float64
	invSX := 1.0
	if scaleX > 0 {
		invSX = 1.0 / scaleX
	}
	for _, p := range pats {
		origPt := mapPointToOrig(p.Center, scaleX, scaleY, padOffset)
		if pointInsideAnyDecoded(origPt, results) {
			continue
		}
		ms := p.ModuleSize * invSX
		for i, qPt := range outPts {
			minM := math.Min(ms, outMS[i])
			maxM := math.Max(ms, outMS[i])
			if minM > 0 && maxM <= minM*2.6 {
				d := origPt.Dist(qPt)
				if d >= minM*10.0 && d <= maxM*250.0 {
					return true
				}
			}
		}
		outPts = append(outPts, origPt)
		outMS = append(outMS, ms)
	}
	return false
}

func makeDecodedResult(dec *decoder.DecodeResult, quad geom.Quad) Result {
	return Result{
		Text:        dec.Text,
		RawBytes:    dec.RawBytes,
		Quad:        quad,
		Version:     dec.Version,
		ECLevel:     dec.ECLevel,
		Orientation: quadOrientationDeg(quad),
		Decoded:     true,
	}
}

func quadOrientationDeg(q geom.Quad) float64 {
	dx := q[1].X - q[0].X
	dy := q[1].Y - q[0].Y
	return math.Atan2(dy, dx) * 180.0 / math.Pi
}

func estimateFinderSetQuad(set FinderPatternSet) geom.Quad {
	estBR := set.TR.Center.Sub(set.TL.Center).Add(set.BL.Center)
	h := geom.SquareToQuadrilateral(set.TL.Center, set.TR.Center, estBR, set.BL.Center)
	if !h.IsValid() {
		return geom.Quad{set.TL.Center, set.TR.Center, estBR, set.BL.Center}
	}
	// Expand by ~3.5 modules outward from center-of-finder square.
	d := dist(set.TL.Center, set.TR.Center)
	ms := (set.TL.ModuleSize + set.TR.ModuleSize + set.BL.ModuleSize) / 3.0
	if d <= 0 || ms <= 0 {
		return geom.Quad{set.TL.Center, set.TR.Center, estBR, set.BL.Center}
	}
	pad := 3.5 * ms / d
	return geom.Quad{
		h.Project(-pad, -pad),
		h.Project(1+pad, -pad),
		h.Project(1+pad, 1+pad),
		h.Project(-pad, 1+pad),
	}
}

func mapPointToOrig(p geom.PointF, scaleX, scaleY, padOffset float64) geom.PointF {
	return geom.PointF{
		X: p.X/scaleX - padOffset,
		Y: p.Y/scaleY - padOffset,
	}
}

func mapQuadToOrig(q geom.Quad, scaleX, scaleY, padOffset float64) geom.Quad {
	var out geom.Quad
	for i := 0; i < 4; i++ {
		out[i] = mapPointToOrig(q[i], scaleX, scaleY, padOffset)
	}
	return out
}

func pointInsideAnyDecoded(p geom.PointF, results []Result) bool {
	for _, r := range results {
		if r.Decoded && r.Quad.ContainsPoint(p) {
			return true
		}
	}
	return false
}

func overlapsAnyDecoded(q geom.Quad, results []Result) bool {
	c := q.Center()
	for _, r := range results {
		if !r.Decoded {
			continue
		}
		if geom.PolygonIoU(q, r.Quad) > 0.30 || r.Quad.ContainsPoint(c) || q.ContainsPoint(r.Quad.Center()) {
			return true
		}
	}
	return false
}

func mergeResults(dst *[]Result, incoming []Result, opts Options) {
	for _, cand := range incoming {
		dup := false
		for _, ex := range *dst {
			if !ex.Decoded || !cand.Decoded {
				continue
			}
			iou := geom.PolygonIoU(ex.Quad, cand.Quad)
			sameRegion := iou > 0.30 || ex.Quad.ContainsPoint(cand.Quad.Center()) || cand.Quad.ContainsPoint(ex.Quad.Center())
			if sameRegion && (ex.Text == cand.Text || iou > 0.50) {
				dup = true
				break
			}
			// If single-QR or identical payload at nearly the same position, deduplicate.
			if !opts.Multi && ex.Text == cand.Text {
				dup = true
				break
			}
		}
		if !dup {
			*dst = append(*dst, cand)
		}
	}
}

func dedupUnreadable(in []Result, opts Options) []Result {
	var out []Result
	for _, cand := range in {
		dup := false
		for _, ex := range out {
			if geom.PolygonIoU(ex.Quad, cand.Quad) > 0.30 || ex.Quad.ContainsPoint(cand.Quad.Center()) {
				dup = true
				break
			}
		}
		if !dup {
			out = append(out, cand)
		}
	}
	return trimResults(out, opts)
}

func appendFinderSetsDedup(base, extra []FinderPatternSet) []FinderPatternSet {
	for _, e := range extra {
		dup := false
		for _, b := range base {
			if b.TL.Center.DistSq(e.TL.Center) < 4.0 &&
				b.TR.Center.DistSq(e.TR.Center) < 4.0 &&
				b.BL.Center.DistSq(e.BL.Center) < 4.0 {
				dup = true
				break
			}
		}
		if !dup {
			base = append(base, e)
		}
	}
	return base
}

func hasDecoded(results []Result) bool {
	for _, r := range results {
		if r.Decoded {
			return true
		}
	}
	return false
}

func countDecoded(results []Result) int {
	n := 0
	for _, r := range results {
		if r.Decoded {
			n++
		}
	}
	return n
}

func trimResults(results []Result, opts Options) []Result {
	if opts.MaxResults > 0 && len(results) > opts.MaxResults {
		return results[:opts.MaxResults]
	}
	return results
}

func padWhiteBorder(g *imageproc.GrayImage, pad int) *imageproc.GrayImage {
	nw, nh := g.Width+2*pad, g.Height+2*pad
	out := imageproc.NewGrayImage(nw, nh)
	for i := range out.Pix {
		out.Pix[i] = 255
	}
	for y := 0; y < g.Height; y++ {
		copy(out.Pix[(y+pad)*nw+pad:(y+pad)*nw+pad+g.Width], g.Pix[y*g.Width:(y+1)*g.Width])
	}
	return out
}
