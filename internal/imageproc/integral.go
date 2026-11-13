package imageproc

import "math"

// IntegralImage stores summed-area tables for pixel intensities and squared intensities
// with 1-pixel zero-padded top and left borders so rectangle queries are branchless after clamping.
type IntegralImage struct {
	Sum    []uint64
	SqSum  []uint64
	Width  int
	Height int
}

// NewIntegralImage builds an IntegralImage from g.
func NewIntegralImage(g *GrayImage) *IntegralImage {
	if g == nil || g.Width <= 0 || g.Height <= 0 {
		return &IntegralImage{}
	}
	w, h := g.Width, g.Height
	stride := w + 1
	n := stride * (h + 1)
	buf := make([]uint64, 2*n)
	sum := buf[:n]
	sqSum := buf[n:]

	for y := 0; y < h; y++ {
		var rowSum uint64
		var rowSqSum uint64
		srcRow := g.Pix[y*w : y*w+w]
		prevSum := sum[y*stride : y*stride+stride]
		curSum := sum[(y+1)*stride : (y+1)*stride+stride]
		prevSq := sqSum[y*stride : y*stride+stride]
		curSq := sqSum[(y+1)*stride : (y+1)*stride+stride]

		// Hoist slice bounds checks out of the inner pixel loop.
		_ = srcRow[w-1]
		_ = prevSum[w]
		_ = curSum[w]
		_ = prevSq[w]
		_ = curSq[w]

		x := 0
		for ; x+3 < w; x += 4 {
			v0 := uint64(srcRow[x])
			v1 := uint64(srcRow[x+1])
			v2 := uint64(srcRow[x+2])
			v3 := uint64(srcRow[x+3])

			rs0 := rowSum + v0
			rs1 := rs0 + v1
			rs2 := rs1 + v2
			rs3 := rs2 + v3
			rowSum = rs3

			rq0 := rowSqSum + v0*v0
			rq1 := rq0 + v1*v1
			rq2 := rq1 + v2*v2
			rq3 := rq2 + v3*v3
			rowSqSum = rq3

			curSum[x+1] = prevSum[x+1] + rs0
			curSum[x+2] = prevSum[x+2] + rs1
			curSum[x+3] = prevSum[x+3] + rs2
			curSum[x+4] = prevSum[x+4] + rs3

			curSq[x+1] = prevSq[x+1] + rq0
			curSq[x+2] = prevSq[x+2] + rq1
			curSq[x+3] = prevSq[x+3] + rq2
			curSq[x+4] = prevSq[x+4] + rq3
		}
		for ; x < w; x++ {
			v := uint64(srcRow[x])
			rowSum += v
			rowSqSum += v * v
			curSum[x+1] = prevSum[x+1] + rowSum
			curSq[x+1] = prevSq[x+1] + rowSqSum
		}
	}

	return &IntegralImage{
		Sum:    sum,
		SqSum:  sqSum,
		Width:  w,
		Height: h,
	}
}

// newSumOnlyIntegralImage builds an IntegralImage populating only Sum (skipping SqSum)
// for callers that only need local means.
func newSumOnlyIntegralImage(g *GrayImage) *IntegralImage {
	if g == nil || g.Width <= 0 || g.Height <= 0 {
		return &IntegralImage{}
	}
	w, h := g.Width, g.Height
	stride := w + 1
	sum := make([]uint64, stride*(h+1))

	for y := 0; y < h; y++ {
		var rowSum uint64
		srcRow := g.Pix[y*w : y*w+w]
		prevSum := sum[y*stride : y*stride+stride]
		curSum := sum[(y+1)*stride : (y+1)*stride+stride]
		_ = srcRow[w-1]
		_ = prevSum[w]
		_ = curSum[w]

		x := 0
		for ; x+3 < w; x += 4 {
			rs0 := rowSum + uint64(srcRow[x])
			rs1 := rs0 + uint64(srcRow[x+1])
			rs2 := rs1 + uint64(srcRow[x+2])
			rs3 := rs2 + uint64(srcRow[x+3])
			rowSum = rs3

			curSum[x+1] = prevSum[x+1] + rs0
			curSum[x+2] = prevSum[x+2] + rs1
			curSum[x+3] = prevSum[x+3] + rs2
			curSum[x+4] = prevSum[x+4] + rs3
		}
		for ; x < w; x++ {
			rowSum += uint64(srcRow[x])
			curSum[x+1] = prevSum[x+1] + rowSum
		}
	}
	return &IntegralImage{
		Sum:    sum,
		Width:  w,
		Height: h,
	}
}

func (ii *IntegralImage) clampRect(x0, y0, x1, y1 int) (int, int, int, int, bool) {
	if ii == nil || ii.Width <= 0 || ii.Height <= 0 {
		return 0, 0, 0, 0, false
	}
	if x0 < 0 {
		x0 = 0
	}
	if y0 < 0 {
		y0 = 0
	}
	if x1 > ii.Width {
		x1 = ii.Width
	}
	if y1 > ii.Height {
		y1 = ii.Height
	}
	if x1 <= x0 || y1 <= y0 {
		return 0, 0, 0, 0, false
	}
	return x0, y0, x1, y1, true
}

// RectSum returns the sum of pixel values in the half-open rectangle [x0, x1) x [y0, y1),
// clamped to the image bounds.
func (ii *IntegralImage) RectSum(x0, y0, x1, y1 int) uint64 {
	if ii == nil || ii.Width <= 0 || ii.Height <= 0 {
		return 0
	}
	// Unchecked fast path when caller coordinates are already within [0, Width] x [0, Height].
	if uint(x0) <= uint(x1) && uint(x1) <= uint(ii.Width) && uint(y0) <= uint(y1) && uint(y1) <= uint(ii.Height) {
		stride := ii.Width + 1
		sum := ii.Sum
		i11 := y1*stride + x1
		if i11 < len(sum) {
			return sum[i11] - sum[y0*stride+x1] - sum[y1*stride+x0] + sum[y0*stride+x0]
		}
		return 0
	}
	return ii.rectSumClamped(x0, y0, x1, y1)
}

func (ii *IntegralImage) rectSumClamped(x0, y0, x1, y1 int) uint64 {
	cx0, cy0, cx1, cy1, ok := ii.clampRect(x0, y0, x1, y1)
	if !ok {
		return 0
	}
	stride := ii.Width + 1
	sum := ii.Sum
	i11 := cy1*stride + cx1
	_ = sum[i11]
	return sum[i11] - sum[cy0*stride+cx1] - sum[cy1*stride+cx0] + sum[cy0*stride+cx0]
}

// RectMeanStd returns the arithmetic mean and population standard deviation of pixel values
// in the clamped half-open rectangle [x0, x1) x [y0, y1).
func (ii *IntegralImage) RectMeanStd(x0, y0, x1, y1 int) (mean, std float64) {
	cx0, cy0, cx1, cy1, ok := ii.clampRect(x0, y0, x1, y1)
	if !ok {
		return 0, 0
	}
	area := float64((cx1 - cx0) * (cy1 - cy0))
	stride := ii.Width + 1
	i00 := cy0*stride + cx0
	i01 := cy0*stride + cx1
	i10 := cy1*stride + cx0
	i11 := cy1*stride + cx1

	s := ii.Sum
	sq := ii.SqSum
	_ = s[i11]
	_ = sq[i11]

	sum := s[i11] - s[i01] - s[i10] + s[i00]
	sqSum := sq[i11] - sq[i01] - sq[i10] + sq[i00]

	mean = float64(sum) / area
	variance := float64(sqSum)/area - mean*mean
	if variance < 0 {
		variance = 0
	}
	std = math.Sqrt(variance)
	return mean, std
}
