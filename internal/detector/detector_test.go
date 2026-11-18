package detector

import (
	"math"
	"testing"

	"github.com/firesh/qcore/internal/decoder"
	"github.com/firesh/qcore/internal/geom"
	"github.com/firesh/qcore/internal/imageproc"
)

// --- Minimal QR Encoder Test Helper (Versions 1..4, Byte Mode, EC Level M/L) ---

var testGFExp [512]uint8
var testGFLog [256]uint8

func init() {
	x := 1
	for i := 0; i < 255; i++ {
		testGFExp[i] = uint8(x)
		testGFLog[x] = uint8(i)
		x <<= 1
		if x&0x100 != 0 {
			x ^= 0x11d
		}
	}
	for i := 255; i < 512; i++ {
		testGFExp[i] = testGFExp[i-255]
	}
}

func testGFMul(a, b uint8) uint8 {
	if a == 0 || b == 0 {
		return 0
	}
	return testGFExp[int(testGFLog[a])+int(testGFLog[b])]
}

func testRSGenerator(nEC int) []uint8 {
	g := []uint8{1}
	for i := 0; i < nEC; i++ {
		next := make([]uint8, len(g)+1)
		root := testGFExp[i]
		for j, coef := range g {
			next[j] ^= coef
			next[j+1] ^= testGFMul(coef, root)
		}
		g = next
	}
	return g
}

func testRSEncode(data []byte, nEC int) []byte {
	gen := testRSGenerator(nEC)
	rem := make([]byte, len(data)+nEC)
	copy(rem, data)
	for i := 0; i < len(data); i++ {
		coef := rem[i]
		if coef != 0 {
			for j := 0; j < len(gen); j++ {
				rem[i+j] ^= testGFMul(gen[j], coef)
			}
		}
	}
	out := make([]byte, nEC)
	copy(out, rem[len(data):])
	return out
}

func encodeTestQR(text string, version int, ecLevel int, mask int) [][]bool {
	spec := decoder.ECBlocks[version][ecLevel]
	dim := 17 + 4*version

	// Encode byte mode payload into bit buffer.
	totalDataBits := spec.TotalDataCodewords() * 8
	var bits []bool
	appendBits := func(val, n int) {
		for i := n - 1; i >= 0; i-- {
			bits = append(bits, (val&(1<<i)) != 0)
		}
	}

	appendBits(0x4, 4) // Byte mode indicator.
	if version < 10 {
		appendBits(len(text), 8)
	} else {
		appendBits(len(text), 16)
	}
	for i := 0; i < len(text); i++ {
		appendBits(int(text[i]), 8)
	}
	if len(bits) > totalDataBits {
		panic("encodeTestQR: payload exceeds QR capacity")
	}
	// Terminator (up to 4 zero bits).
	term := min(4, totalDataBits-len(bits))
	appendBits(0, term)
	for len(bits)%8 != 0 {
		bits = append(bits, false)
	}
	padBytes := []byte{0xEC, 0x11}
	for p := 0; len(bits) < totalDataBits; p++ {
		appendBits(int(padBytes[p%2]), 8)
	}

	dataCodewords := make([]byte, spec.TotalDataCodewords())
	for i := range dataCodewords {
		var b byte
		for k := 0; k < 8; k++ {
			b <<= 1
			if bits[i*8+k] {
				b |= 1
			}
		}
		dataCodewords[i] = b
	}

	// Split into RS blocks & interleave.
	numBlocks := spec.TotalBlocks()
	dataBlocks := make([][]byte, numBlocks)
	ecBlocks := make([][]byte, numBlocks)
	off := 0
	maxDataLen := 0
	for b := 0; b < numBlocks; b++ {
		dLen := spec.Group1.DataCodewords
		if b >= spec.Group1.Count {
			dLen = spec.Group2.DataCodewords
		}
		if dLen > maxDataLen {
			maxDataLen = dLen
		}
		dataBlocks[b] = dataCodewords[off : off+dLen]
		ecBlocks[b] = testRSEncode(dataBlocks[b], spec.ECPerBlock)
		off += dLen
	}

	finalCodewords := make([]byte, 0, spec.TotalCodewords)
	for i := 0; i < maxDataLen; i++ {
		for b := 0; b < numBlocks; b++ {
			if i < len(dataBlocks[b]) {
				finalCodewords = append(finalCodewords, dataBlocks[b][i])
			}
		}
	}
	for i := 0; i < spec.ECPerBlock; i++ {
		for b := 0; b < numBlocks; b++ {
			finalCodewords = append(finalCodewords, ecBlocks[b][i])
		}
	}

	// Build module matrix.
	mat := make([][]bool, dim)
	for r := 0; r < dim; r++ {
		mat[r] = make([]bool, dim)
	}

	placeFinder := func(r0, c0 int) {
		for r := 0; r < 7; r++ {
			for c := 0; c < 7; c++ {
				dark := r == 0 || r == 6 || c == 0 || c == 6 || (r >= 2 && r <= 4 && c >= 2 && c <= 4)
				mat[r0+r][c0+c] = dark
			}
		}
	}
	placeFinder(0, 0)
	placeFinder(0, dim-7)
	placeFinder(dim-7, 0)

	// Timing patterns.
	for i := 8; i < dim-8; i++ {
		dark := i%2 == 0
		mat[6][i] = dark
		mat[i][6] = dark
	}

	// Dark module.
	mat[dim-8][8] = true

	// Alignment patterns.
	centers := decoder.AlignmentPatternCenters(version)
	k := len(centers)
	for i, cy := range centers {
		for j, cx := range centers {
			if (i == 0 && j == 0) || (i == 0 && j == k-1) || (i == k-1 && j == 0) {
				continue
			}
			for dr := -2; dr <= 2; dr++ {
				for dc := -2; dc <= 2; dc++ {
					dark := dr == -2 || dr == 2 || dc == -2 || dc == 2 || (dr == 0 && dc == 0)
					mat[cy+dr][cx+dc] = dark
				}
			}
		}
	}

	// Place data bits with mask.
	fnMask := decoder.BuildFunctionPatternMask(version)
	totalBits := len(finalCodewords) * 8
	bitIdx := 0
	readingUp := true
	applyMask := func(r, c int) bool {
		switch mask {
		case 0:
			return (r+c)%2 == 0
		case 1:
			return r%2 == 0
		case 2:
			return c%3 == 0
		case 3:
			return (r+c)%3 == 0
		case 4:
			return (r/2+c/3)%2 == 0
		case 5:
			return (r*c)%2+(r*c)%3 == 0
		case 6:
			return ((r*c)%2+(r*c)%3)%2 == 0
		default:
			return ((r+c)%2+(r*c)%3)%2 == 0
		}
	}

	for col := dim - 1; col > 0; col -= 2 {
		if col == 6 {
			col--
		}
		for i := 0; i < dim; i++ {
			r := i
			if readingUp {
				r = dim - 1 - i
			}
			for cOff := 0; cOff < 2; cOff++ {
				c := col - cOff
				if fnMask[r][c] {
					continue
				}
				var bit bool
				if bitIdx < totalBits {
					bit = (finalCodewords[bitIdx/8] & (1 << (7 - bitIdx%8))) != 0
					bitIdx++
				}
				if applyMask(r, c) {
					bit = !bit
				}
				mat[r][c] = bit
			}
		}
		readingUp = !readingUp
	}

	// Encode 15-bit format info.
	fData := uint16((ecLevel << 3) | mask)
	v := fData << 10
	for i := 4; i >= 0; i-- {
		if v&(1<<(i+10)) != 0 {
			v ^= 0x537 << i
		}
	}
	fmtBits := ((fData << 10) | v) ^ decoder.FormatMask

	setFmt := func(r, c, bitPos int) {
		mat[r][c] = (fmtBits & (1 << bitPos)) != 0
	}
	for i := 0; i <= 5; i++ {
		setFmt(8, i, 14-i)
		setFmt(i, 8, i)
	}
	setFmt(8, 7, 8)
	setFmt(8, 8, 7)
	setFmt(7, 8, 6)
	for i := 0; i <= 6; i++ {
		setFmt(dim-1-i, 8, 14-i)
	}
	for i := 0; i <= 7; i++ {
		setFmt(8, dim-8+i, 7-i)
	}

	if version >= 7 {
		gv := uint32(version) << 12
		rem := gv
		for i := 5; i >= 0; i-- {
			if rem&(1<<(i+12)) != 0 {
				rem ^= 0x1F25 << i
			}
		}
		verBits := gv | rem
		for k := 0; k < 18; k++ {
			bit := (verBits & (1 << k)) != 0
			r := k / 3
			c := k % 3
			mat[r][dim-11+c] = bit
			mat[dim-11+c][r] = bit
		}
	}

	return mat
}

// renderQRToGray renders mat onto a GrayImage using a point-mapping function
// imgToModule(x, y) -> (u, v) in [0, dim] x [0, dim].
func renderQRToGray(w, h int, bg, fg uint8, mat [][]bool, imgToModule func(x, y float64) (float64, float64)) *imageproc.GrayImage {
	g := imageproc.NewGrayImage(w, h)
	dim := len(mat)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			u, v := imgToModule(float64(x)+0.5, float64(y)+0.5)
			c := int(math.Floor(u))
			r := int(math.Floor(v))
			if r >= 0 && r < dim && c >= 0 && c < dim && mat[r][c] {
				g.Pix[y*w+x] = fg
			} else {
				g.Pix[y*w+x] = bg
			}
		}
	}
	return g
}

func TestDetectAndDecode_PlanarAndRotated(t *testing.T) {
	payload := "https://qore.dev/planar-v2"
	mat := encodeTestQR(payload, 2, decoder.ECM, 2)
	dim := len(mat)

	// 1. Clean planar QR.
	ms := 8.0
	pad := 32.0
	w := int(float64(dim)*ms + 2*pad)
	h := w
	gray := renderQRToGray(w, h, 245, 15, mat, func(x, y float64) (float64, float64) {
		return (x - pad) / ms, (y - pad) / ms
	})

	res := DetectAndDecode(gray, Options{})
	if len(res) != 1 || !res[0].Decoded || res[0].Text != payload {
		t.Fatalf("planar decode failed: %+v", res)
	}
	if res[0].Version != 2 || res[0].ECLevel != "M" {
		t.Fatalf("unexpected metadata: version=%d ec=%s", res[0].Version, res[0].ECLevel)
	}

	// 2. Rotated (37 deg) + perspective skewed QR.
	angle := 37.0 * math.Pi / 180.0
	cosA, sinA := math.Cos(angle), math.Sin(angle)
	cx, cy := 220.0, 220.0
	rotGray := renderQRToGray(440, 440, 240, 20, mat, func(x, y float64) (float64, float64) {
		dx := x - cx
		dy := y - cy
		// Inverse rotation from image to module coordinates.
		ux := (dx*cosA + dy*sinA) / 7.5
		vy := (-dx*sinA + dy*cosA) / 7.5
		return ux + float64(dim)*0.5, vy + float64(dim)*0.5
	})

	resRot := DetectAndDecode(rotGray, Options{})
	if len(resRot) != 1 || !resRot[0].Decoded || resRot[0].Text != payload {
		t.Fatalf("rotated decode failed: %+v", resRot)
	}
}

func TestDetectAndDecode_InvertedAndLowContrast(t *testing.T) {
	payload := "INVERTED-LOW-01"
	mat := encodeTestQR(payload, 1, decoder.ECL, 1)
	dim := len(mat)
	ms := 9.0
	pad := 36.0
	w := int(float64(dim)*ms + 2*pad)

	// 1. Inverted QR (white modules on dark background).
	invGray := renderQRToGray(w, w, 20, 235, mat, func(x, y float64) (float64, float64) {
		return (x - pad) / ms, (y - pad) / ms
	})
	resInv := DetectAndDecode(invGray, Options{Multi: true})
	if len(resInv) != 1 || !resInv[0].Decoded || resInv[0].Text != payload {
		t.Fatalf("inverted decode failed: %+v", resInv)
	}

	// 2. Low-contrast QR (fg=115, bg=138, range=23 < HybridBinarizer default 24).
	lowGray := renderQRToGray(w, w, 138, 115, mat, func(x, y float64) (float64, float64) {
		return (x - pad) / ms, (y - pad) / ms
	})
	resLow := DetectAndDecode(lowGray, Options{})
	if len(resLow) != 1 || !resLow[0].Decoded || resLow[0].Text != payload {
		t.Fatalf("low-contrast decode failed: %+v", resLow)
	}
}

func TestDetectAndDecode_BarrelWarped(t *testing.T) {
	payload := "BARREL-WARP-TEST"
	mat := encodeTestQR(payload, 2, decoder.ECM, 3)
	dim := len(mat)

	w, h := 320, 320
	ms := 7.0
	pad := (float64(w) - float64(dim)*ms) * 0.5
	rt := geom.NewRadialTransform(float64(w)*0.5, float64(h)*0.5, float64(w)*0.5, float64(h)*0.5, -0.30, 0)

	// Render barrel-distorted QR: map image pixel (x,y) -> undistorted via Inverse -> module.
	gray := renderQRToGray(w, h, 245, 15, mat, func(x, y float64) (float64, float64) {
		uPt := rt.Inverse(geom.PointF{X: x, Y: y})
		return (uPt.X - pad) / ms, (uPt.Y - pad) / ms
	})

	res := DetectAndDecode(gray, Options{Multi: true})
	if len(res) != 1 || !res[0].Decoded || res[0].Text != payload {
		t.Fatalf("barrel-warped decode failed: %+v", res)
	}
}

func TestDetectAndDecode_CylindricalWarped(t *testing.T) {
	payload := "CYLINDER-WARP-350"
	mat := encodeTestQR(payload, 2, decoder.ECM, 4)
	dim := len(mat)

	w, h := 360, 360
	ms := 7.5
	pad := (float64(w) - float64(dim)*ms) * 0.5
	ct := geom.NewCylindricalTransform(float64(w)*0.5, float64(h)*0.5, 350.0, 0, 0)

	// Render cylindrical-warped QR: map image pixel (x,y) -> flat unwrapped via Inverse -> module.
	gray := renderQRToGray(w, h, 245, 15, mat, func(x, y float64) (float64, float64) {
		flat, ok := ct.Inverse(geom.PointF{X: x, Y: y})
		if !ok {
			return -1, -1
		}
		return (flat.X - pad) / ms, (flat.Y - pad) / ms
	})

	res := DetectAndDecode(gray, Options{Multi: true})
	if len(res) != 1 || !res[0].Decoded || res[0].Text != payload {
		t.Fatalf("cylindrical-warped decode failed: %+v", res)
	}
}

func TestDetectAndDecode_MultiQR(t *testing.T) {
	p1 := "MULTI-QR-ALPHA"
	p2 := "MULTI-QR-BETA"
	p3 := "MULTI-QR-GAMMA"
	m1 := encodeTestQR(p1, 1, decoder.ECL, 0)
	m2 := encodeTestQR(p2, 2, decoder.ECM, 1)
	m3 := encodeTestQR(p3, 1, decoder.ECL, 5)

	w, h := 600, 480
	g := imageproc.NewGrayImage(w, h)
	for i := range g.Pix {
		g.Pix[i] = 245
	}

	stamp := func(mat [][]bool, ox, oy, ms float64) {
		dim := len(mat)
		for r := 0; r < dim; r++ {
			for c := 0; c < dim; c++ {
				if !mat[r][c] {
					continue
				}
				x0 := int(math.Round(ox + float64(c)*ms))
				x1 := int(math.Round(ox + float64(c+1)*ms))
				y0 := int(math.Round(oy + float64(r)*ms))
				y1 := int(math.Round(oy + float64(r+1)*ms))
				for y := max(0, y0); y < min(h, y1); y++ {
					for x := max(0, x0); x < min(w, x1); x++ {
						g.Pix[y*w+x] = 15
					}
				}
			}
		}
	}

	stamp(m1, 30, 30, 7.0)
	stamp(m2, 300, 40, 8.0)
	stamp(m3, 160, 270, 6.0)

	res := DetectAndDecode(g, Options{Multi: true})
	got := make(map[string]bool)
	for _, r := range res {
		if r.Decoded {
			got[r.Text] = true
		}
	}
	for _, want := range []string{p1, p2, p3} {
		if !got[want] {
			t.Fatalf("multi-QR missed %q; decoded=%v", want, got)
		}
	}
}

func TestDetectAndDecode_SynthesizeMissingFinder(t *testing.T) {
	payload := "SYNTH-FINDER"
	mat := encodeTestQR(payload, 2, decoder.ECH, 2)
	dim := len(mat)
	ms := 8.0
	pad := 32.0
	w := int(float64(dim)*ms + 2*pad)

	gray := renderQRToGray(w, w, 245, 15, mat, func(x, y float64) (float64, float64) {
		return (x - pad) / ms, (y - pad) / ms
	})

	// White-out the entire top-right finder pattern (7x7 modules) to simulate glare damage.
	x0 := int(pad + float64(dim-7)*ms)
	x1 := int(pad + float64(dim)*ms)
	y0 := int(pad)
	y1 := int(pad + 7.0*ms)
	for y := y0; y <= y1; y++ {
		for x := x0; x <= x1; x++ {
			gray.Pix[y*w+x] = 245
		}
	}

	res := DetectAndDecode(gray, Options{Multi: true})
	if len(res) != 1 || !res[0].Decoded || res[0].Text != payload {
		t.Fatalf("missing finder synthesis failed: %+v", res)
	}
}

func TestDetectAndDecode_Tier2TiledAndTier3Coons(t *testing.T) {
	// 1. Tier 2 (SampleTiled): Version 7 (dim = 45) with smooth 2D pillow warp (~5 px)
	// that breaks planar homography while alignment pattern tiles [6,22,38] track it.
	payload2 := "TIER2-TILED-ALIGNMENT-TEST-V7"
	mat2 := encodeTestQR(payload2, 7, decoder.ECM, 2)
	dim2 := len(mat2)

	w2, h2 := 400, 400
	ms2 := 7.0
	pad2 := (float64(w2) - float64(dim2)*ms2) * 0.5
	span2 := float64(dim2) * ms2
	gray2 := renderQRToGray(w2, h2, 245, 15, mat2, func(x, y float64) (float64, float64) {
		nx := (x - pad2) / span2
		ny := (y - pad2) / span2
		// Smooth interior nonlinear bow (~6 px): SamplePlanar fails, SampleTiled tracks via 3x3 alignment grid.
		dx := 6.0 * math.Sin(math.Pi*nx) * math.Sin(math.Pi*ny)
		dy := 6.0 * math.Sin(math.Pi*nx) * math.Sin(math.Pi*ny)
		return (x - dx - pad2) / ms2, (y - dy - pad2) / ms2
	})

	bm2 := imageproc.HybridBinarizer(gray2, 24)
	integ2 := imageproc.NewIntegralImage(gray2)
	pats2 := FindFinderPatterns(bm2, false)
	sets2 := GenerateFinderPatternSets(pats2, 4, false)
	if len(sets2) == 0 {
		t.Fatalf("expected finder sets for Tier 2 v7 QR")
	}
	_, _, ctx2, okPlanar := SamplePlanar(gray2, bm2, integ2, sets2[0])
	if okPlanar {
		t.Fatalf("expected SamplePlanar to fail on 6px interior-bowed v7 QR")
	}
	resTiled, _, okTiled := SampleTiled(gray2, bm2, integ2, sets2[0], ctx2.dim, ctx2.baseH)
	if !okTiled || resTiled.Text != payload2 {
		t.Fatalf("SampleTiled failed (dim=%d): ok=%v res=%+v", ctx2.dim, okTiled, resTiled)
	}

	// 2. Tier 3 (SampleCurvedCoons): Version 2 (dim = 25) with parabolic curved edges
	// (no interior alignment grid; CoonsPatchGrid traces the 4 curved boundaries).
	payload3 := "TIER3-COONS-CURVE"
	mat3 := encodeTestQR(payload3, 2, decoder.ECM, 1)
	dim3 := len(mat3)

	w3, h3 := 300, 300
	ms3 := 8.0
	pad3 := (float64(w3) - float64(dim3)*ms3) * 0.5
	qrSpan := float64(dim3) * ms3
	bow := 12.0
	gray3 := renderQRToGray(w3, h3, 245, 15, mat3, func(x, y float64) (float64, float64) {
		nx := (x - pad3) / qrSpan
		ny := (y - pad3) / qrSpan
		// Smooth Coons-compatible edge bow (0 at corners, +bow at edge midpoints).
		dx := bow * math.Sin(math.Pi*ny)
		dy := bow * math.Sin(math.Pi*nx)
		return (x - dx - pad3) / ms3, (y - dy - pad3) / ms3
	})

	bm3 := imageproc.HybridBinarizer(gray3, 24)
	integ3 := imageproc.NewIntegralImage(gray3)
	pats3 := FindFinderPatterns(bm3, false)
	sets3 := GenerateFinderPatternSets(pats3, 4, false)
	if len(sets3) == 0 {
		t.Fatalf("expected finder sets for Tier 3 curved QR")
	}
	_, _, ctx3, okP3 := SamplePlanar(gray3, bm3, integ3, sets3[0])
	if okP3 {
		t.Fatalf("expected SamplePlanar to fail on bowed v2 QR")
	}
	resCoons, _, okCoons := SampleCurvedCoons(gray3, bm3, integ3, sets3[0], ctx3.dim, ctx3.baseH)
	if !okCoons || resCoons.Text != payload3 {
		t.Fatalf("SampleCurvedCoons failed: ok=%v res=%+v", okCoons, resCoons)
	}

	// 3. Full cascade DetectAndDecode on sinusoidal WaveTransform.
	payloadW := "TIER4-WAVE-DEFORM"
	matW := encodeTestQR(payloadW, 2, decoder.ECM, 3)
	dimW := len(matW)
	wW, hW := 320, 320
	msW := 8.0
	padW := (float64(wW) - float64(dimW)*msW) * 0.5
	wt := geom.NewWaveTransform(4.5, 2.0*math.Pi/200.0, 0.2, 4.5, 2.0*math.Pi/200.0, 0.6)
	grayW := renderQRToGray(wW, hW, 245, 15, matW, func(x, y float64) (float64, float64) {
		uPt := wt.Inverse(geom.PointF{X: x, Y: y})
		return (uPt.X - padW) / msW, (uPt.Y - padW) / msW
	})
	resW := DetectAndDecode(grayW, Options{Multi: true})
	if len(resW) != 1 || !resW[0].Decoded || resW[0].Text != payloadW {
		t.Fatalf("wave-warped decode failed: %+v", resW)
	}
}

func TestDetectAndDecode_PureBarcodeAndUnreadable(t *testing.T) {
	payload := "PURE-BARCODE"
	mat := encodeTestQR(payload, 1, decoder.ECM, 0)
	dim := len(mat)

	// Zero quiet zone (padding = 0).
	ms := 8.0
	w := int(float64(dim) * ms)
	gray := renderQRToGray(w, w, 245, 15, mat, func(x, y float64) (float64, float64) {
		return x / ms, y / ms
	})

	res := DetectAndDecode(gray, Options{PureBarcode: true})
	if len(res) != 1 || !res[0].Decoded || res[0].Text != payload {
		t.Fatalf("pure barcode decode failed: %+v", res)
	}

	// Corrupt all data modules while preserving the 3 finder patterns -> ReturnUnreadableDetections.
	corrupt := renderQRToGray(w+64, w+64, 245, 15, mat, func(x, y float64) (float64, float64) {
		return (x - 32.0) / ms, (y - 32.0) / ms
	})
	// Wipe center data area (modules 8..13 x 8..13) to white so RS fails completely.
	for y := 32 + int(8*ms); y <= 32+int(14*ms); y++ {
		for x := 32 + int(8*ms); x <= 32+int(14*ms); x++ {
			corrupt.Pix[y*(w+64)+x] = 245
		}
	}
	// Also wipe format info strips at row 8 / col 8.
	for x := 32; x < 32+w; x++ {
		for dy := 0; dy < int(ms); dy++ {
			corrupt.Pix[(32+int(8*ms)+dy)*(w+64)+x] = 245
		}
	}

	un := DetectAndDecode(corrupt, Options{ReturnUnreadableDetections: true})
	if len(un) == 0 || un[0].Decoded {
		t.Fatalf("expected unreadable detection with Decoded=false, got: %+v", un)
	}
}

func TestMergedFinderExterior(t *testing.T) {
	const payload = "merged finder"
	mat := encodeTestQR(payload, 2, decoder.ECM, 1)
	for _, ms := range []int{6, 15, 30} {
		for _, colors := range [][3]uint8{{170, 26, 255}, {255, 0, 255}, {0, 255, 0}, {235, 20, 20}} {
			size := (len(mat) + 8) * ms
			gray := imageproc.NewGrayImage(size, size)
			for y := 0; y < size; y++ {
				for x := 0; x < size; x++ {
					r, c := y/ms-4, x/ms-4
					v := colors[2]
					if r >= 0 && c >= 0 && r < len(mat) && c < len(mat) {
						v = colors[1]
						if mat[r][c] {
							v = colors[0]
						}
					}
					gray.Pix[y*size+x] = v
				}
			}
			for _, multi := range []bool{false, true} {
				got := DetectAndDecode(gray, Options{Multi: multi})
				if len(got) != 1 || got[0].Text != payload {
					t.Fatalf("module=%d colors=%v multi=%t got=%v", ms, colors, multi, got)
				}
			}
		}
	}
}
