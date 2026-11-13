package imageproc

import (
	"bytes"
	"encoding/binary"
	"image"
	"image/color"
	"image/gif"
	"image/jpeg"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestExtractChannelAndGrayOps(t *testing.T) {
	// Test RGBA / NRGBA channel extraction
	nrgba := image.NewNRGBA(image.Rect(1, 2, 4, 4)) // 3x2 with non-zero Min
	nrgba.SetNRGBA(1, 2, color.NRGBA{R: 255, G: 0, B: 0, A: 255})
	nrgba.SetNRGBA(2, 2, color.NRGBA{R: 0, G: 255, B: 0, A: 255})
	nrgba.SetNRGBA(3, 2, color.NRGBA{R: 0, G: 0, B: 255, A: 255})
	nrgba.SetNRGBA(1, 3, color.NRGBA{R: 200, G: 200, B: 200, A: 255})
	nrgba.SetNRGBA(2, 3, color.NRGBA{R: 100, G: 50, B: 25, A: 255})
	nrgba.SetNRGBA(3, 3, color.NRGBA{R: 0, G: 0, B: 0, A: 255})

	red := ExtractChannel(nrgba, ChannelRed)
	if red.Width != 3 || red.Height != 2 {
		t.Fatalf("unexpected dims: %dx%d", red.Width, red.Height)
	}
	if red.At(0, 0) != 255 || red.At(1, 0) != 0 || red.At(0, 1) != 200 {
		t.Fatalf("unexpected red channel values: %v", red.Pix)
	}

	green := ExtractChannel(nrgba, ChannelGreen)
	if green.At(1, 0) != 255 || green.At(0, 0) != 0 {
		t.Fatalf("unexpected green channel values: %v", green.Pix)
	}

	blue := ExtractChannel(nrgba, ChannelBlue)
	if blue.At(2, 0) != 255 || blue.At(0, 0) != 0 {
		t.Fatalf("unexpected blue channel values: %v", blue.Pix)
	}

	sat := ExtractChannel(nrgba, ChannelSaturation)
	if sat.At(0, 0) != 255 || sat.At(0, 1) != 0 || sat.At(2, 1) != 0 {
		t.Fatalf("unexpected saturation values: %v", sat.Pix)
	}

	luma := FromImage(nrgba)
	if luma.At(1, 0) <= luma.At(0, 0) || luma.At(0, 0) <= luma.At(2, 0) {
		t.Fatalf("expected green > red > blue in luma, got %d, %d, %d", luma.At(1, 0), luma.At(0, 0), luma.At(2, 0))
	}

	// Test YCbCr direct Y-plane copy with stride & non-zero bounds
	ycbcr := image.NewYCbCr(image.Rect(2, 2, 6, 5), image.YCbCrSubsampleRatio420)
	for y := 2; y < 5; y++ {
		for x := 2; x < 6; x++ {
			ycbcr.Y[ycbcr.YOffset(x, y)] = uint8(10*y + x)
			ci := ycbcr.COffset(x, y)
			ycbcr.Cb[ci] = 128
			ycbcr.Cr[ci] = 128
		}
	}
	gy := FromImage(ycbcr)
	if gy.Width != 4 || gy.Height != 3 {
		t.Fatalf("unexpected YCbCr gray dims: %dx%d", gy.Width, gy.Height)
	}
	if gy.At(0, 0) != 22 || gy.At(3, 2) != 45 {
		t.Fatalf("unexpected YCbCr copied values: (0,0)=%d (3,2)=%d", gy.At(0, 0), gy.At(3, 2))
	}

	// Test Gray direct copy, At clamping, SampleBilinear, Invert, Clone, Crop
	g := NewGrayImage(2, 2)
	g.Pix = []uint8{0, 100, 200, 100}
	if g.At(-5, -5) != 0 || g.At(5, 5) != 100 {
		t.Fatalf("clamping failed")
	}
	if got := g.SampleBilinear(0.5, 0.5); math.Abs(got-100.0) > 1e-9 {
		t.Fatalf("bilinear center = %v, want 100", got)
	}
	inv := g.Invert()
	if inv.At(0, 0) != 255 || inv.At(1, 0) != 155 {
		t.Fatalf("invert failed: %v", inv.Pix)
	}
	if g.At(0, 0) != 0 {
		t.Fatalf("invert mutated source")
	}
	cropped := g.Crop(1, 0, 5, 2)
	if cropped.Width != 1 || cropped.Height != 2 || cropped.At(0, 0) != 100 || cropped.At(0, 1) != 100 {
		t.Fatalf("crop failed: %+v", cropped)
	}
}

// buildJPEGWithExifOrientation encodes img as JPEG and injects an APP1 EXIF segment
// containing tag 0x0112 = orient using either little-endian ("II") or big-endian ("MM").
func buildJPEGWithExifOrientation(t *testing.T, img image.Image, orient uint16, littleEndian bool) []byte {
	t.Helper()
	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, img, &jpeg.Options{Quality: 100}); err != nil {
		t.Fatalf("jpeg.Encode: %v", err)
	}
	raw := buf.Bytes()
	if len(raw) < 2 || raw[0] != 0xFF || raw[1] != 0xD8 {
		t.Fatalf("invalid JPEG SOI")
	}

	var order binary.ByteOrder = binary.BigEndian
	endianTag := []byte("MM")
	if littleEndian {
		order = binary.LittleEndian
		endianTag = []byte("II")
	}

	// TIFF header (8) + IFD0 count (2) + 1 entry (12) + next IFD offset (4) = 26 bytes.
	tiff := make([]byte, 26)
	copy(tiff[0:2], endianTag)
	order.PutUint16(tiff[2:4], 0x002A)
	order.PutUint32(tiff[4:8], 8)     // IFD0 offset
	order.PutUint16(tiff[8:10], 1)    // 1 entry
	order.PutUint16(tiff[10:12], 0x0112)
	order.PutUint16(tiff[12:14], 3) // SHORT
	order.PutUint32(tiff[14:18], 1) // count 1
	order.PutUint16(tiff[18:20], orient)
	order.PutUint32(tiff[22:26], 0) // end of IFD

	payload := append([]byte("Exif\x00\x00"), tiff...)
	segLen := uint16(len(payload) + 2)
	app1 := make([]byte, 4+len(payload))
	app1[0] = 0xFF
	app1[1] = 0xE1
	binary.BigEndian.PutUint16(app1[2:4], segLen)
	copy(app1[4:], payload)

	out := make([]byte, 0, len(raw)+len(app1))
	out = append(out, raw[:2]...)
	out = append(out, app1...)
	out = append(out, raw[2:]...)
	return out
}

func TestReadImageWithOrientation(t *testing.T) {
	// Asymmetric 24x16 upright pattern: white background, black L-shape in top-left.
	// Using 8x8 blocks avoids JPEG 4:2:0 / DCT edge ringing across quadrants.
	const w, h = 24, 16
	upright := image.NewGray(image.Rect(0, 0, w, h))
	for i := range upright.Pix {
		upright.Pix[i] = 255
	}
	// Mark (0..7, 0..7) = black(0), (8..15, 0..7) = dark gray(60), rest = white(255)
	for y := 0; y < 8; y++ {
		for x := 0; x < 8; x++ {
			upright.SetGray(x, y, color.Gray{Y: 0})
			upright.SetGray(x+8, y, color.Gray{Y: 60})
		}
	}

	dir := t.TempDir()

	for orient := 1; orient <= 8; orient++ {
		// Create stored image that, when transformed by orient, equals upright.
		// Inverse transform: for each upright (dx, dy), stored(sx, sy) = upright(dx, dy).
		sw, sh := w, h
		if orient >= 5 {
			sw, sh = h, w
		}
		stored := image.NewGray(image.Rect(0, 0, sw, sh))
		for dy := 0; dy < h; dy++ {
			for dx := 0; dx < w; dx++ {
				sx, sy := mapOrientedCoord(dx, dy, sw, sh, orient)
				stored.SetGray(sx, sy, upright.GrayAt(dx, dy))
			}
		}

		for _, le := range []bool{true, false} {
			jpgBytes := buildJPEGWithExifOrientation(t, stored, uint16(orient), le)
			p := filepath.Join(dir, "test.jpg")
			if err := os.WriteFile(p, jpgBytes, 0o600); err != nil {
				t.Fatalf("write jpeg: %v", err)
			}
			gotImg, err := ReadImageWithOrientation(p)
			if err != nil {
				t.Fatalf("orient=%d le=%v ReadImageWithOrientation: %v", orient, le, err)
			}
			gotGray := FromImage(gotImg)
			if gotGray.Width != w || gotGray.Height != h {
				t.Fatalf("orient=%d le=%v got dims %dx%d, want %dx%d", orient, le, gotGray.Width, gotGray.Height, w, h)
			}
			// Check centers of key 8x8 blocks: (4,4) should be near 0, (12,4) near 60, (20,12) near 255.
			if gotGray.At(4, 4) > 25 {
				t.Errorf("orient=%d le=%v (4,4)=%d want <25", orient, le, gotGray.At(4, 4))
			}
			if gotGray.At(12, 4) < 35 || gotGray.At(12, 4) > 85 {
				t.Errorf("orient=%d le=%v (12,4)=%d want ~60", orient, le, gotGray.At(12, 4))
			}
			if gotGray.At(20, 12) < 230 {
				t.Errorf("orient=%d le=%v (20,12)=%d want >230", orient, le, gotGray.At(20, 12))
			}
		}
	}

	// Also verify PNG and GIF decoding.
	pngPath := filepath.Join(dir, "test.png")
	var pngBuf bytes.Buffer
	if err := png.Encode(&pngBuf, upright); err != nil {
		t.Fatalf("png.Encode: %v", err)
	}
	if err := os.WriteFile(pngPath, pngBuf.Bytes(), 0o600); err != nil {
		t.Fatalf("write png: %v", err)
	}
	if img, err := ReadImageWithOrientation(pngPath); err != nil || img.Bounds().Dx() != w {
		t.Fatalf("PNG read failed: img=%v err=%v", img, err)
	}

	gifPath := filepath.Join(dir, "test.gif")
	var gifBuf bytes.Buffer
	pal := color.Palette{color.Black, color.White}
	pImg := image.NewPaletted(image.Rect(0, 0, 8, 8), pal)
	pImg.SetColorIndex(2, 2, 1)
	if err := gif.Encode(&gifBuf, pImg, nil); err != nil {
		t.Fatalf("gif.Encode: %v", err)
	}
	if err := os.WriteFile(gifPath, gifBuf.Bytes(), 0o600); err != nil {
		t.Fatalf("write gif: %v", err)
	}
	if img, err := ReadImageWithOrientation(gifPath); err != nil || img.Bounds().Dx() != 8 {
		t.Fatalf("GIF read failed: img=%v err=%v", img, err)
	}
}

func TestBitMatrixAndCursors(t *testing.T) {
	bm := NewBitMatrix(20, 15)
	bm.Set(3, 4)
	bm.SetVal(5, 6, true)
	if !bm.Get(3, 4) || !bm.Get(5, 6) || bm.Get(0, 0) || bm.Get(-1, 0) {
		t.Fatalf("BitMatrix Get/Set mismatch")
	}
	bm.ClearRect(2, 3, 5, 5)
	if bm.Get(3, 4) {
		t.Fatalf("ClearRect did not clear (3,4)")
	}
	row := bm.RowSlice(6)
	if len(row) != 20 || row[5] != 1 {
		t.Fatalf("RowSlice mismatch")
	}
	clone := bm.Clone()
	clone.Invert()
	if !clone.Get(0, 0) || clone.Get(5, 6) || !bm.Get(5, 6) {
		t.Fatalf("Invert/Clone mismatch")
	}

	// CursorF ReadSymmetricPattern5 on 1:1:3:1:1 finder pattern (scale 2 -> widths 2,2,6,2,2).
	// Pattern on row y=5: W(0..1) B(2..3) W(4..5) B(6..11) W(12..13) B(14..15) W(16..19)
	pat := NewBitMatrix(20, 11)
	setSpan := func(x0, x1, y int) {
		for x := x0; x < x1; x++ {
			pat.Set(x, y)
		}
	}
	setSpan(2, 4, 5)
	setSpan(6, 12, 5)
	setSpan(14, 16, 5)

	cf := NewCursorF(pat, 8.5, 5.5, 1, 0)
	widths, ok := cf.ReadSymmetricPattern5(30)
	if !ok {
		t.Fatalf("ReadSymmetricPattern5 failed")
	}
	want := [5]int{2, 2, 6, 2, 2}
	if widths != want {
		t.Fatalf("ReadSymmetricPattern5 = %v, want %v", widths, want)
	}

	// StepToEdge with backup=true vs backup=false
	cf2 := NewCursorF(pat, 6.5, 5.5, 1, 0)
	steps := cf2.StepToEdge(1, 20, true)
	if steps != 6 || int(math.Floor(cf2.X)) != 11 {
		t.Fatalf("StepToEdge(backup=true) steps=%d X=%v", steps, cf2.X)
	}

	// CursorI perimeter trace of a 3x3 black square with white border.
	sq := NewBitMatrix(7, 7)
	for y := 2; y <= 4; y++ {
		for x := 2; x <= 4; x++ {
			sq.Set(x, y)
		}
	}
	ci := NewCursorI(sq, 2, 2, DirRight)
	if !ci.EdgeAtLeft() {
		t.Fatalf("expected edge at left of (2,2) facing right")
	}
	visited := 0
	for {
		if !ci.StepAlongEdge(-1, false) {
			t.Fatalf("StepAlongEdge failed at (%d,%d)", ci.X, ci.Y)
		}
		visited++
		if ci.X == 2 && ci.Y == 2 {
			break
		}
		if visited > 20 {
			t.Fatalf("perimeter loop did not close")
		}
	}
	if visited != 8 {
		t.Fatalf("3x3 square perimeter = %d steps, want 8", visited)
	}
}

func TestIntegralImageExactness(t *testing.T) {
	g := NewGrayImage(17, 13)
	for y := 0; y < g.Height; y++ {
		for x := 0; x < g.Width; x++ {
			g.Pix[y*g.Width+x] = uint8((x*37 + y*53 + 11) % 256)
		}
	}
	integ := NewIntegralImage(g)

	rects := [][4]int{
		{0, 0, 17, 13},
		{2, 3, 10, 9},
		{-3, -2, 5, 4},
		{12, 8, 25, 20},
		{5, 5, 6, 6},
	}
	for _, r := range rects {
		x0, y0, x1, y1 := r[0], r[1], r[2], r[3]
		gotSum := integ.RectSum(x0, y0, x1, y1)
		gotMean, gotStd := integ.RectMeanStd(x0, y0, x1, y1)

		var wantSum, wantSq uint64
		var count int
		for y := max(0, y0); y < min(g.Height, y1); y++ {
			for x := max(0, x0); x < min(g.Width, x1); x++ {
				v := uint64(g.At(x, y))
				wantSum += v
				wantSq += v * v
				count++
			}
		}
		if gotSum != wantSum {
			t.Fatalf("rect %v sum = %d, want %d", r, gotSum, wantSum)
		}
		wantMean := float64(wantSum) / float64(count)
		wantStd := math.Sqrt(math.Max(0, float64(wantSq)/float64(count)-wantMean*wantMean))
		if math.Abs(gotMean-wantMean) > 1e-9 || math.Abs(gotStd-wantStd) > 1e-9 {
			t.Fatalf("rect %v mean/std = (%v,%v), want (%v,%v)", r, gotMean, gotStd, wantMean, wantStd)
		}
	}
}

func TestBinarizersAndLowContrast(t *testing.T) {
	// Synthetic 60x60 checkerboard of 6x6 modules (not aligned to 8x8 blocks) with uneven illumination.
	const size = 60
	const mod = 6
	g := NewGrayImage(size, size)
	expected := NewBitMatrix(size, size)
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			isBlack := ((x/mod)+(y/mod))%2 == 0
			expected.SetVal(x, y, isBlack)
			base := 70.0 + 110.0*float64(x)/float64(size-1) // strong left-to-right lighting gradient
			if isBlack {
				g.Pix[y*size+x] = uint8(base - 45)
			} else {
				g.Pix[y*size+x] = uint8(base + 45)
			}
		}
	}

	integ := NewIntegralImage(g)
	hyb := HybridBinarizer(g, 24)
	adapt := AdaptiveMeanBinarizer(g, integ, 17, 0.05)
	sauv := SauvolaBinarizer(g, integ, 17, 0.2)

	checkAccuracy := func(name string, bm *BitMatrix, minMatch float64) {
		t.Helper()
		matches := 0
		for i := range bm.Bits {
			if bm.Bits[i] == expected.Bits[i] {
				matches++
			}
		}
		ratio := float64(matches) / float64(len(bm.Bits))
		if ratio < minMatch {
			t.Errorf("%s accuracy %.3f < %.3f", name, ratio, minMatch)
		}
	}

	checkAccuracy("HybridBinarizer(24)", hyb, 0.90)
	checkAccuracy("AdaptiveMeanBinarizer", adapt, 0.92)
	checkAccuracy("SauvolaBinarizer", sauv, 0.90)

	// Global Otsu on bimodal high-contrast image (no gradient).
	bimodal := NewGrayImage(32, 32)
	for y := 0; y < 32; y++ {
		for x := 0; x < 32; x++ {
			if x < 16 {
				bimodal.Pix[y*32+x] = 40
			} else {
				bimodal.Pix[y*32+x] = 210
			}
		}
	}
	otsu := GlobalOtsuBinarizer(bimodal)
	if !otsu.Get(4, 4) || otsu.Get(24, 4) {
		t.Fatalf("GlobalOtsuBinarizer failed on bimodal image")
	}

	// Very low contrast image with 4x4 modules: dark=120, light=132 (dynamic range = 12).
	// Default minDynamicRange=24 suppresses flat/low-contrast blocks; minDynamicRange=6 recovers them.
	lowC := NewGrayImage(32, 32)
	for y := 0; y < 32; y++ {
		for x := 0; x < 32; x++ {
			if (x/4+y/4)%2 == 0 {
				lowC.Pix[y*32+x] = 120
			} else {
				lowC.Pix[y*32+x] = 132
			}
		}
	}
	hybStrict := HybridBinarizer(lowC, 24)
	hybLow := HybridBinarizer(lowC, 6)
	lowMatches := 0
	strictBlacks := 0
	for y := 0; y < 32; y++ {
		for x := 0; x < 32; x++ {
			wantBlack := (x/4+y/4)%2 == 0
			if hybLow.Get(x, y) == wantBlack {
				lowMatches++
			}
			if hybStrict.Get(x, y) {
				strictBlacks++
			}
		}
	}
	if lowMatches < 32*32-32 {
		t.Fatalf("HybridBinarizer(minDynamicRange=6) failed on low contrast: %d/1024", lowMatches)
	}
	if strictBlacks != 0 {
		t.Fatalf("HybridBinarizer(minDynamicRange=24) should suppress range=12 blocks, got %d blacks", strictBlacks)
	}

	// Verify HybridBinarizer propagates threshold into large solid black squares (> 16x16).
	bigSq := NewGrayImage(40, 40)
	for i := range bigSq.Pix {
		bigSq.Pix[i] = 240
	}
	for y := 10; y < 30; y++ {
		for x := 10; x < 30; x++ {
			bigSq.Pix[y*40+x] = 20
		}
	}
	hybSq := HybridBinarizer(bigSq, 24)
	if !hybSq.Get(20, 20) || hybSq.Get(2, 2) {
		t.Fatalf("HybridBinarizer failed large black square center/background: center=%v bg=%v", hybSq.Get(20, 20), hybSq.Get(2, 2))
	}

	// LocalThresholdAt should track local average.
	ltLeft := LocalThresholdAt(g, integ, 4, 24, 15)
	ltRight := LocalThresholdAt(g, integ, 54, 24, 15)
	if ltRight <= ltLeft {
		t.Fatalf("expected right local threshold (%v) > left (%v)", ltRight, ltLeft)
	}
}

func TestFilters(t *testing.T) {
	// CLAHE should expand narrow-range histogram contrast.
	narrow := NewGrayImage(64, 64)
	for y := 0; y < 64; y++ {
		for x := 0; x < 64; x++ {
			narrow.Pix[y*64+x] = uint8(110 + ((x+y)%16))
		}
	}
	eq := CLAHE(narrow, 4, 3.0)
	minV, maxV := uint8(255), uint8(0)
	for _, p := range eq.Pix {
		if p < minV {
			minV = p
		}
		if p > maxV {
			maxV = p
		}
	}
	if maxV-minV <= 50 {
		t.Fatalf("CLAHE did not expand contrast enough: min=%d max=%d", minV, maxV)
	}

	// GaussianBlur3x3 + UnsharpMask + LaplacianSharpen on step edge.
	step := NewGrayImage(9, 9)
	for y := 0; y < 9; y++ {
		for x := 0; x < 9; x++ {
			if x >= 4 {
				step.Pix[y*9+x] = 200
			} else {
				step.Pix[y*9+x] = 50
			}
		}
	}
	blurred := GaussianBlur3x3(step)
	if blurred.At(3, 4) <= 50 || blurred.At(4, 4) >= 200 {
		t.Fatalf("GaussianBlur3x3 did not smooth edge: (3,4)=%d (4,4)=%d", blurred.At(3, 4), blurred.At(4, 4))
	}
	unsharp := UnsharpMask(blurred, 1.5)
	if unsharp.At(4, 4) <= blurred.At(4, 4) || unsharp.At(3, 4) >= blurred.At(3, 4) {
		t.Fatalf("UnsharpMask did not increase edge contrast")
	}
	lap := LaplacianSharpen(blurred, 0.5)
	if lap.At(4, 4) <= blurred.At(4, 4) || lap.At(3, 4) >= blurred.At(3, 4) {
		t.Fatalf("LaplacianSharpen did not increase edge contrast")
	}

	// MedianFilter3x3 removes single salt impulse noise.
	noisy := NewGrayImage(7, 7)
	for i := range noisy.Pix {
		noisy.Pix[i] = 40
	}
	noisy.Pix[3*7+3] = 255
	med := MedianFilter3x3(noisy)
	if med.At(3, 3) != 40 {
		t.Fatalf("MedianFilter3x3 failed to remove salt impulse: got %d", med.At(3, 3))
	}

	// Verify median9 network matches exact sorted median across 1000 pseudo-random 9-byte windows.
	seed := uint32(0x12345678)
	for iter := 0; iter < 1000; iter++ {
		var vals [9]uint8
		for k := 0; k < 9; k++ {
			seed = seed*1664525 + 1013904223
			vals[k] = uint8(seed >> 24)
		}
		got := median9(vals[0], vals[1], vals[2], vals[3], vals[4], vals[5], vals[6], vals[7], vals[8])
		sorted := vals
		for i := 1; i < 9; i++ {
			key := sorted[i]
			j := i - 1
			for j >= 0 && sorted[j] > key {
				sorted[j+1] = sorted[j]
				j--
			}
			sorted[j+1] = key
		}
		if got != sorted[4] {
			t.Fatalf("median9(%v) = %d, want %d", vals, got, sorted[4])
		}
	}

	// UpscaleBicubic preserves constant regions and scales dimensions accurately.
	small := NewGrayImage(4, 4)
	for y := 0; y < 4; y++ {
		for x := 0; x < 4; x++ {
			small.Pix[y*4+x] = uint8(x * 60)
		}
	}
	up := UpscaleBicubic(small, 2.0)
	if up.Width != 8 || up.Height != 8 {
		t.Fatalf("UpscaleBicubic dims = %dx%d, want 8x8", up.Width, up.Height)
	}
	if up.At(0, 0) > 10 || up.At(7, 0) < 170 {
		t.Fatalf("UpscaleBicubic boundary values unexpected: (0,0)=%d (7,0)=%d", up.At(0, 0), up.At(7, 0))
	}

	// DownscaleArea averages 2x2 blocks accurately.
	hi := NewGrayImage(4, 4)
	for y := 0; y < 4; y++ {
		for x := 0; x < 4; x++ {
			if x < 2 && y < 2 {
				hi.Pix[y*4+x] = 100
			} else {
				hi.Pix[y*4+x] = 200
			}
		}
	}
	down := DownscaleArea(hi, 0.5)
	if down.Width != 2 || down.Height != 2 {
		t.Fatalf("DownscaleArea dims = %dx%d, want 2x2", down.Width, down.Height)
	}
	if down.At(0, 0) != 100 || down.At(1, 0) != 200 || down.At(0, 1) != 200 || down.At(1, 1) != 200 {
		t.Fatalf("DownscaleArea values unexpected: %v", down.Pix)
	}
}
