package imageproc

import (
	"image"
	"testing"
)

func makeBenchGray(w, h int) *GrayImage {
	g := NewGrayImage(w, h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			if ((x/8)+(y/8))%2 == 0 {
				g.Pix[y*w+x] = uint8(40 + (x+y)%30)
			} else {
				g.Pix[y*w+x] = uint8(200 + (x+y)%30)
			}
		}
	}
	return g
}

func BenchmarkNewGrayImage(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = NewGrayImage(1024, 1024)
	}
}

func BenchmarkExtractChannel_RGBA(b *testing.B) {
	img := image.NewRGBA(image.Rect(0, 0, 1024, 1024))
	for i := 0; i < len(img.Pix); i += 4 {
		img.Pix[i] = 120
		img.Pix[i+1] = 180
		img.Pix[i+2] = 60
		img.Pix[i+3] = 255
	}
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = ExtractChannel(img, ChannelLuma)
	}
}

func BenchmarkExtractChannel_NRGBA(b *testing.B) {
	img := image.NewNRGBA(image.Rect(0, 0, 1024, 1024))
	for i := 0; i < len(img.Pix); i += 4 {
		img.Pix[i] = 120
		img.Pix[i+1] = 180
		img.Pix[i+2] = 60
		img.Pix[i+3] = 255
	}
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = ExtractChannel(img, ChannelLuma)
	}
}

func BenchmarkExtractChannel_YCbCr(b *testing.B) {
	img := image.NewYCbCr(image.Rect(0, 0, 1024, 1024), image.YCbCrSubsampleRatio420)
	for i := range img.Y {
		img.Y[i] = uint8(i)
	}
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = ExtractChannel(img, ChannelLuma)
	}
}

func BenchmarkExtractChannel_Gray(b *testing.B) {
	img := image.NewGray(image.Rect(0, 0, 1024, 1024))
	for i := range img.Pix {
		img.Pix[i] = uint8(i)
	}
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = ExtractChannel(img, ChannelLuma)
	}
}

func BenchmarkNewIntegralImage(b *testing.B) {
	g := makeBenchGray(1024, 1024)
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = NewIntegralImage(g)
	}
}

func BenchmarkIntegralRectSum(b *testing.B) {
	g := makeBenchGray(1024, 1024)
	integ := NewIntegralImage(g)
	var acc uint64
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		x := (i * 17) & 1023
		y := (i * 31) & 1023
		acc += integ.RectSum(x-10, y-10, x+11, y+11)
	}
	if acc == 1<<63 {
		b.Fatal(acc)
	}
}

func BenchmarkHybridBinarizer(b *testing.B) {
	g := makeBenchGray(1024, 1024)
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = HybridBinarizer(g, 24)
	}
}

func BenchmarkAdaptiveMeanBinarizer(b *testing.B) {
	g := makeBenchGray(1024, 1024)
	integ := NewIntegralImage(g)
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = AdaptiveMeanBinarizer(g, integ, 31, 0.05)
	}
}

func BenchmarkSauvolaBinarizer(b *testing.B) {
	g := makeBenchGray(1024, 1024)
	integ := NewIntegralImage(g)
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = SauvolaBinarizer(g, integ, 31, 0.2)
	}
}

func BenchmarkGlobalOtsuBinarizer(b *testing.B) {
	g := makeBenchGray(1024, 1024)
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = GlobalOtsuBinarizer(g)
	}
}

func BenchmarkLocalThresholdAt_WithInteg(b *testing.B) {
	g := makeBenchGray(1024, 1024)
	integ := NewIntegralImage(g)
	var acc float64
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		x := float64((i*17)&1023) + 0.35
		y := float64((i*31)&1023) + 0.65
		acc += LocalThresholdAt(g, integ, x, y, 21)
	}
	if acc < 0 {
		b.Fatal(acc)
	}
}

func BenchmarkSampleBilinear(b *testing.B) {
	g := makeBenchGray(1024, 1024)
	var acc float64
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		x := float64((i*17)&1023) + 0.35
		y := float64((i*31)&1023) + 0.65
		acc += g.SampleBilinear(x, y)
	}
	if acc < 0 {
		b.Fatal(acc)
	}
}

func BenchmarkBitMatrixGet(b *testing.B) {
	bm := NewBitMatrix(1024, 1024)
	bm.Set(512, 512)
	var cnt int
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if bm.Get((i*17)&1023, (i*31)&1023) {
			cnt++
		}
	}
	if cnt < 0 {
		b.Fatal(cnt)
	}
}

func BenchmarkDownscaleArea(b *testing.B) {
	g := makeBenchGray(1024, 1024)
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = DownscaleArea(g, 0.5)
	}
}

func BenchmarkDownscaleArea_NonInteger(b *testing.B) {
	g := makeBenchGray(1024, 1024)
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = DownscaleArea(g, 0.6)
	}
}

func BenchmarkUpscaleBicubic(b *testing.B) {
	g := makeBenchGray(512, 512)
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = UpscaleBicubic(g, 2.0)
	}
}

func BenchmarkCLAHE(b *testing.B) {
	g := makeBenchGray(1024, 1024)
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = CLAHE(g, 8, 3.0)
	}
}

func BenchmarkUnsharpMask(b *testing.B) {
	g := makeBenchGray(1024, 1024)
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = UnsharpMask(g, 1.5)
	}
}

func BenchmarkMedianFilter3x3(b *testing.B) {
	g := makeBenchGray(1024, 1024)
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = MedianFilter3x3(g)
	}
}
