package imageproc

import "math"

// HybridBinarizer binarizes g using ZXing-style 8x8 block min/max/mean estimation
// followed by 5x5 block threshold smoothing.
// minDynamicRange defaults to 24 if < 0; smaller values (e.g. 6) recover low-contrast codes.
func HybridBinarizer(g *GrayImage, minDynamicRange int) *BitMatrix {
	if g == nil || g.Width <= 0 || g.Height <= 0 {
		return NewBitMatrix(0, 0)
	}
	if minDynamicRange < 0 {
		minDynamicRange = 24
	}

	w, h := g.Width, g.Height
	const blockSize = 8
	subW := (w + blockSize - 1) >> 3
	subH := (h + blockSize - 1) >> 3
	fullSubW := w >> 3
	fullSubH := h >> 3

	blackPoints := make([]int32, subW*subH)
	minVals := make([]int32, subW*subH) // >=0 stores minVal for low-dynamic-range blocks; -1 otherwise
	pix := g.Pix
	par := w*h >= 131072

	// Pass 1: compute per-block sum/min/max in parallel.
	parallelFor(fullSubH, par, func(by0, by1 int) {
		for by := by0; by < by1; by++ {
			yBase := (by << 3) * w
			bpRow := blackPoints[by*subW : by*subW+subW]
			mvRow := minVals[by*subW : by*subW+subW]
			for bx := 0; bx < fullSubW; bx++ {
				off := yBase + (bx << 3)
				sum := 0
				minVal := 255
				maxVal := 0
				for dy := 0; dy < 8; dy++ {
					row := pix[off : off+8]
					_ = row[7]
					p0, p1, p2, p3 := int(row[0]), int(row[1]), int(row[2]), int(row[3])
					p4, p5, p6, p7 := int(row[4]), int(row[5]), int(row[6]), int(row[7])
					sum += p0 + p1 + p2 + p3 + p4 + p5 + p6 + p7

					mn0, mx0 := p0, p0
					if p1 < mn0 {
						mn0 = p1
					} else if p1 > mx0 {
						mx0 = p1
					}
					if p2 < mn0 {
						mn0 = p2
					} else if p2 > mx0 {
						mx0 = p2
					}
					if p3 < mn0 {
						mn0 = p3
					} else if p3 > mx0 {
						mx0 = p3
					}
					if p4 < mn0 {
						mn0 = p4
					} else if p4 > mx0 {
						mx0 = p4
					}
					if p5 < mn0 {
						mn0 = p5
					} else if p5 > mx0 {
						mx0 = p5
					}
					if p6 < mn0 {
						mn0 = p6
					} else if p6 > mx0 {
						mx0 = p6
					}
					if p7 < mn0 {
						mn0 = p7
					} else if p7 > mx0 {
						mx0 = p7
					}
					if mn0 < minVal {
						minVal = mn0
					}
					if mx0 > maxVal {
						maxVal = mx0
					}
					off += w
				}
				if maxVal-minVal <= minDynamicRange {
					mvRow[bx] = int32(minVal)
				} else {
					mvRow[bx] = -1
					bpRow[bx] = int32(sum >> 6)
				}
			}
		}
	})

	// Pass 2: sequential neighbor propagation on the tiny (subW x subH) grid.
	for by := 0; by < fullSubH; by++ {
		bpRow := blackPoints[by*subW : by*subW+subW]
		mvRow := minVals[by*subW : by*subW+subW]
		var prevBPRow []int32
		if by > 0 {
			prevBPRow = blackPoints[(by-1)*subW : (by-1)*subW+subW]
		}
		for bx := 0; bx < fullSubW; bx++ {
			mv := mvRow[bx]
			if mv >= 0 {
				avg := mv >> 1
				if by > 0 && bx > 0 {
					neighborAvg := (prevBPRow[bx] + 2*bpRow[bx-1] + prevBPRow[bx-1]) >> 2
					if mv < neighborAvg {
						avg = neighborAvg
					}
				}
				bpRow[bx] = avg
			}
		}
	}

	if fullSubW < subW {
		for by := 0; by < subH; by++ {
			calcPartialBlock(pix, w, h, fullSubW, by, subW, minDynamicRange, blackPoints)
		}
	}
	if fullSubH < subH {
		for bx := 0; bx < fullSubW; bx++ {
			calcPartialBlock(pix, w, h, bx, fullSubH, subW, minDynamicRange, blackPoints)
		}
	}

	out := NewBitMatrix(w, h)
	dstBits := out.Bits

	// Build 2D prefix sum of blackPoints so every 5x5 block average is 4 ops.
	bpStride := subW + 1
	bpPrefix := make([]int32, bpStride*(subH+1))
	for by := 0; by < subH; by++ {
		var rowSum int32
		src := blackPoints[by*subW : by*subW+subW]
		prev := bpPrefix[by*bpStride : by*bpStride+bpStride]
		cur := bpPrefix[(by+1)*bpStride : (by+1)*bpStride+bpStride]
		for bx := 0; bx < subW; bx++ {
			rowSum += src[bx]
			cur[bx+1] = prev[bx+1] + rowSum
		}
	}

	// Threshold full 8x8 blocks in parallel with branchless comparison and 5x5 O(1) prefix lookup.
	if subW >= 5 && subH >= 5 {
		parallelFor(fullSubH, par, func(byStart, byEnd int) {
			for by := byStart; by < byEnd; by++ {
				cy := by
				if cy < 2 {
					cy = 2
				} else if cy > subH-3 {
					cy = subH - 3
				}
				sy0 := cy - 2
				sy1 := cy + 3
				pRow0 := bpPrefix[sy0*bpStride:]
				pRow1 := bpPrefix[sy1*bpStride:]
				yBase := (by << 3) * w

				for bx := 0; bx < fullSubW; bx++ {
					cx := bx
					if cx < 2 {
						cx = 2
					} else if cx > subW-3 {
						cx = subW - 3
					}
					sx0 := cx - 2
					sx1 := cx + 3
					sum5x5 := pRow1[sx1] - pRow0[sx1] - pRow1[sx0] + pRow0[sx0]
					t := int32((uint32(sum5x5)*83887)>>21) + 1

					off := yBase + (bx << 3)
					for dy := 0; dy < 8; dy++ {
						s := pix[off : off+8]
						d := dstBits[off : off+8]
						_ = s[7]
						_ = d[7]
						d[0] = uint8(uint32(int32(s[0])-t) >> 31)
						d[1] = uint8(uint32(int32(s[1])-t) >> 31)
						d[2] = uint8(uint32(int32(s[2])-t) >> 31)
						d[3] = uint8(uint32(int32(s[3])-t) >> 31)
						d[4] = uint8(uint32(int32(s[4])-t) >> 31)
						d[5] = uint8(uint32(int32(s[5])-t) >> 31)
						d[6] = uint8(uint32(int32(s[6])-t) >> 31)
						d[7] = uint8(uint32(int32(s[7])-t) >> 31)
						off += w
					}
				}
			}
		})

		if fullSubW < subW {
			for by := 0; by < subH; by++ {
				thresholdPartialBlockPrefix(pix, dstBits, w, h, fullSubW, by, subW, subH, bpPrefix, bpStride)
			}
		}
		if fullSubH < subH {
			for bx := 0; bx < fullSubW; bx++ {
				thresholdPartialBlockPrefix(pix, dstBits, w, h, bx, fullSubH, subW, subH, bpPrefix, bpStride)
			}
		}
		return out
	}

	// Small-image fallback (width < 40 or height < 40).
	for by := 0; by < subH; by++ {
		sy0, sy1 := smoothBlockRange(by, subH)
		py0 := by << 3
		py1 := min(h, py0+8)
		for bx := 0; bx < subW; bx++ {
			sx0, sx1 := smoothBlockRange(bx, subW)
			sum := bpPrefix[(sy1+1)*bpStride+(sx1+1)] - bpPrefix[sy0*bpStride+(sx1+1)] -
				bpPrefix[(sy1+1)*bpStride+sx0] + bpPrefix[sy0*bpStride+sx0]
			cnt := int32((sy1 - sy0 + 1) * (sx1 - sx0 + 1))
			t := sum/cnt + 1
			px0 := bx << 3
			px1 := min(w, px0+8)
			for y := py0; y < py1; y++ {
				s := pix[y*w:]
				d := dstBits[y*w:]
				for x := px0; x < px1; x++ {
					d[x] = uint8(uint32(int32(s[x])-t) >> 31)
				}
			}
		}
	}
	return out
}

func calcPartialBlock(pix []uint8, w, h, bx, by, subW, minDynamicRange int, blackPoints []int32) {
	x0 := bx << 3
	x1 := min(w, x0+8)
	y0 := by << 3
	y1 := min(h, y0+8)
	sum := 0
	minVal := 255
	maxVal := 0
	for y := y0; y < y1; y++ {
		row := pix[y*w:]
		for x := x0; x < x1; x++ {
			p := int(row[x])
			sum += p
			if p < minVal {
				minVal = p
			}
			if p > maxVal {
				maxVal = p
			}
		}
	}
	count := (x1 - x0) * (y1 - y0)
	avg := int32(sum / count)
	if maxVal-minVal <= minDynamicRange {
		avg = int32(minVal >> 1)
		if by > 0 && bx > 0 {
			neighborAvg := (blackPoints[(by-1)*subW+bx] +
				2*blackPoints[by*subW+bx-1] +
				blackPoints[(by-1)*subW+bx-1]) >> 2
			if int32(minVal) < neighborAvg {
				avg = neighborAvg
			}
		}
	}
	blackPoints[by*subW+bx] = avg
}

func thresholdPartialBlockPrefix(pix, dstBits []uint8, w, h, bx, by, subW, subH int, bpPrefix []int32, bpStride int) {
	cx := clampInt(bx, 2, subW-3)
	cy := clampInt(by, 2, subH-3)
	sum5x5 := bpPrefix[(cy+3)*bpStride+(cx+3)] - bpPrefix[(cy-2)*bpStride+(cx+3)] -
		bpPrefix[(cy+3)*bpStride+(cx-2)] + bpPrefix[(cy-2)*bpStride+(cx-2)]
	t := int32((uint32(sum5x5)*83887)>>21) + 1

	px0 := bx << 3
	px1 := min(w, px0+8)
	py0 := by << 3
	py1 := min(h, py0+8)
	for y := py0; y < py1; y++ {
		s := pix[y*w:]
		d := dstBits[y*w:]
		for x := px0; x < px1; x++ {
			d[x] = uint8(uint32(int32(s[x])-t) >> 31)
		}
	}
}

func smoothBlockRange(idx, total int) (int, int) {
	if total <= 5 {
		return 0, total - 1
	}
	center := idx
	if center < 2 {
		center = 2
	} else if center > total-3 {
		center = total - 3
	}
	return center - 2, center + 2
}

// AdaptiveMeanBinarizer applies Bradley-Roth adaptive mean thresholding:
// pixel is marked black if float64(pix) < mean*(1.0 - bias).
// If integ is nil, an IntegralImage is computed internally.
func AdaptiveMeanBinarizer(g *GrayImage, integ *IntegralImage, windowSize int, bias float64) *BitMatrix {
	if g == nil || g.Width <= 0 || g.Height <= 0 {
		return NewBitMatrix(0, 0)
	}
	w, h := g.Width, g.Height
	if bias >= 1.0 {
		return NewBitMatrix(w, h)
	}
	if integ == nil {
		integ = newSumOnlyIntegralImage(g)
	}
	if windowSize < 3 {
		windowSize = 15
	}
	half := windowSize / 2
	out := NewBitMatrix(w, h)
	scale := 1.0 - bias
	stride := w + 1
	sumBuf := integ.Sum

	ixStart := half
	ixEnd := w - windowSize + half + 1
	iyStart := half
	iyEnd := h - windowSize + half + 1
	if ixEnd < ixStart {
		ixStart, ixEnd = w, w
	}
	if iyEnd < iyStart {
		iyStart, iyEnd = h, h
	}

	interiorArea := windowSize * windowSize
	pixScale := uint64(math.Round((float64(interiorArea) / scale) * 65536.0))

	adaptiveRowBorder := func(y int) {
		cy0 := max(0, y-half)
		cy1 := min(h, y-half+windowSize)
		dy := cy1 - cy0
		r0 := sumBuf[cy0*stride : cy0*stride+stride]
		r1 := sumBuf[cy1*stride : cy1*stride+stride]
		srcRow := g.Pix[y*w : y*w+w]
		dstRow := out.Bits[y*w : y*w+w]
		for x := 0; x < w; x++ {
			cx0 := max(0, x-half)
			cx1 := min(w, x-half+windowSize)
			area := float64((cx1 - cx0) * dy)
			s := r1[cx1] - r0[cx1] - r1[cx0] + r0[cx0]
			if float64(srcRow[x]) < (float64(s)/area)*scale {
				dstRow[x] = 1
			}
		}
	}

	for y := 0; y < min(iyStart, h); y++ {
		adaptiveRowBorder(y)
	}

	if iyStart < iyEnd {
		nInt := ixEnd - ixStart
		par := w*h >= 131072
		parallelFor(iyEnd-iyStart, par, func(dy0, dy1 int) {
			for y := iyStart + dy0; y < iyStart+dy1; y++ {
				y0 := y - half
				y1 := y0 + windowSize
				r0 := sumBuf[y0*stride : y0*stride+stride]
				r1 := sumBuf[y1*stride : y1*stride+stride]
				srcRow := g.Pix[y*w : y*w+w]
				dstRow := out.Bits[y*w : y*w+w]

				for x := 0; x < ixStart; x++ {
					cx1 := min(w, x-half+windowSize)
					s := r1[cx1] - r0[cx1] - r1[0] + r0[0]
					if float64(srcRow[x]) < (float64(s)/float64(cx1*windowSize))*scale {
						dstRow[x] = 1
					}
				}

				if nInt > 0 {
					x0Base := ixStart - half
					x1Base := x0Base + windowSize
					r0L := r0[x0Base : x0Base+nInt]
					r0R := r0[x1Base : x1Base+nInt]
					r1L := r1[x0Base : x0Base+nInt]
					r1R := r1[x1Base : x1Base+nInt]
					src := srcRow[ixStart : ixStart+nInt]
					dst := dstRow[ixStart : ixStart+nInt]
					_ = r0L[nInt-1]
					_ = r0R[nInt-1]
					_ = r1L[nInt-1]
					_ = r1R[nInt-1]
					_ = src[nInt-1]
					_ = dst[nInt-1]

					i := 0
					for ; i+3 < nInt; i += 4 {
						s0 := (r1R[i] - r0R[i] - r1L[i] + r0L[i]) << 16
						s1 := (r1R[i+1] - r0R[i+1] - r1L[i+1] + r0L[i+1]) << 16
						s2 := (r1R[i+2] - r0R[i+2] - r1L[i+2] + r0L[i+2]) << 16
						s3 := (r1R[i+3] - r0R[i+3] - r1L[i+3] + r0L[i+3]) << 16
						dst[i] = uint8(uint64(int64(uint64(src[i])*pixScale)-int64(s0)) >> 63)
						dst[i+1] = uint8(uint64(int64(uint64(src[i+1])*pixScale)-int64(s1)) >> 63)
						dst[i+2] = uint8(uint64(int64(uint64(src[i+2])*pixScale)-int64(s2)) >> 63)
						dst[i+3] = uint8(uint64(int64(uint64(src[i+3])*pixScale)-int64(s3)) >> 63)
					}
					for ; i < nInt; i++ {
						s := (r1R[i] - r0R[i] - r1L[i] + r0L[i]) << 16
						dst[i] = uint8(uint64(int64(uint64(src[i])*pixScale)-int64(s)) >> 63)
					}
				}

				for x := ixEnd; x < w; x++ {
					cx0 := max(0, x-half)
					s := r1[w] - r0[w] - r1[cx0] + r0[cx0]
					if float64(srcRow[x]) < (float64(s)/float64((w-cx0)*windowSize))*scale {
						dstRow[x] = 1
					}
				}
			}
		})
	}

	for y := max(iyStart, iyEnd); y < h; y++ {
		adaptiveRowBorder(y)
	}
	return out
}

// SauvolaBinarizer applies Sauvola local thresholding:
// T = mean * (1 + k * (std/128.0 - 1)).
// If integ is nil, an IntegralImage is computed internally.
func SauvolaBinarizer(g *GrayImage, integ *IntegralImage, windowSize int, k float64) *BitMatrix {
	if g == nil || g.Width <= 0 || g.Height <= 0 {
		return NewBitMatrix(0, 0)
	}
	if integ == nil {
		integ = NewIntegralImage(g)
	}
	if windowSize < 3 {
		windowSize = 15
	}
	half := windowSize / 2
	w, h := g.Width, g.Height
	out := NewBitMatrix(w, h)
	stride := w + 1
	sumBuf := integ.Sum
	sqBuf := integ.SqSum

	ixStart := half
	ixEnd := w - windowSize + half + 1
	iyStart := half
	iyEnd := h - windowSize + half + 1
	if ixEnd < ixStart {
		ixStart, ixEnd = w, w
	}
	if iyEnd < iyStart {
		iyStart, iyEnd = h, h
	}

	invArea := 1.0 / float64(windowSize*windowSize)
	oneMinusK := 1.0 - k
	kDiv128 := k / 128.0

	sauvolaBorderRow := func(y int) {
		y0 := y - half
		y1 := y0 + windowSize
		srcRow := g.Pix[y*w : y*w+w]
		dstRow := out.Bits[y*w : y*w+w]
		for x := 0; x < w; x++ {
			x0 := x - half
			x1 := x0 + windowSize
			mean, std := integ.RectMeanStd(x0, y0, x1, y1)
			thresh := mean * (oneMinusK + kDiv128*std)
			if float64(srcRow[x]) < thresh {
				dstRow[x] = 1
			}
		}
	}

	for y := 0; y < min(iyStart, h); y++ {
		sauvolaBorderRow(y)
	}

	if iyStart < iyEnd {
		nInt := ixEnd - ixStart
		par := w*h >= 131072
		parallelFor(iyEnd-iyStart, par, func(dy0, dy1 int) {
			for y := iyStart + dy0; y < iyStart+dy1; y++ {
				y0 := y - half
				y1 := y0 + windowSize
				r0 := sumBuf[y0*stride : y0*stride+stride]
				r1 := sumBuf[y1*stride : y1*stride+stride]
				sq0 := sqBuf[y0*stride : y0*stride+stride]
				sq1 := sqBuf[y1*stride : y1*stride+stride]
				srcRow := g.Pix[y*w : y*w+w]
				dstRow := out.Bits[y*w : y*w+w]

				for x := 0; x < ixStart; x++ {
					mean, std := integ.RectMeanStd(x-half, y0, x-half+windowSize, y1)
					if float64(srcRow[x]) < mean*(oneMinusK+kDiv128*std) {
						dstRow[x] = 1
					}
				}

				if nInt > 0 {
					x0Base := ixStart - half
					x1Base := x0Base + windowSize
					r0L := r0[x0Base : x0Base+nInt]
					r0R := r0[x1Base : x1Base+nInt]
					r1L := r1[x0Base : x0Base+nInt]
					r1R := r1[x1Base : x1Base+nInt]
					sq0L := sq0[x0Base : x0Base+nInt]
					sq0R := sq0[x1Base : x1Base+nInt]
					sq1L := sq1[x0Base : x0Base+nInt]
					sq1R := sq1[x1Base : x1Base+nInt]
					src := srcRow[ixStart : ixStart+nInt]
					dst := dstRow[ixStart : ixStart+nInt]
					_ = r0L[nInt-1]
					_ = r0R[nInt-1]
					_ = r1L[nInt-1]
					_ = r1R[nInt-1]
					_ = sq0L[nInt-1]
					_ = sq0R[nInt-1]
					_ = sq1L[nInt-1]
					_ = sq1R[nInt-1]
					_ = src[nInt-1]
					_ = dst[nInt-1]

					for i := 0; i < nInt; i++ {
						s := r1R[i] - r0R[i] - r1L[i] + r0L[i]
						sq := sq1R[i] - sq0R[i] - sq1L[i] + sq0L[i]
						mean := float64(s) * invArea
						variance := float64(sq)*invArea - mean*mean
						if variance < 0 {
							variance = 0
						}
						thresh := mean * (oneMinusK + kDiv128*math.Sqrt(variance))
						if float64(src[i]) < thresh {
							dst[i] = 1
						}
					}
				}

				for x := ixEnd; x < w; x++ {
					mean, std := integ.RectMeanStd(x-half, y0, x-half+windowSize, y1)
					if float64(srcRow[x]) < mean*(oneMinusK+kDiv128*std) {
						dstRow[x] = 1
					}
				}
			}
		})
	}

	for y := max(iyStart, iyEnd); y < h; y++ {
		sauvolaBorderRow(y)
	}
	return out
}

// GlobalOtsuBinarizer computes a global threshold via 256-bin Otsu between-class
// variance maximization and binarizes g (pixels <= threshold become black).
func GlobalOtsuBinarizer(g *GrayImage) *BitMatrix {
	if g == nil || g.Width <= 0 || g.Height <= 0 {
		return NewBitMatrix(0, 0)
	}
	pix := g.Pix
	n := len(pix)
	par := n >= 131072

	var partHists [8][256]int
	numWorkers := parallelIndexed(n, par, func(worker, s0, s1 int) {
		var h0, h1, h2, h3 [256]int
		p := pix[s0:s1]
		m := len(p)
		i := 0
		for ; i+3 < m; i += 4 {
			h0[p[i]]++
			h1[p[i+1]]++
			h2[p[i+2]]++
			h3[p[i+3]]++
		}
		for ; i < m; i++ {
			h0[p[i]]++
		}
		dstH := &partHists[worker]
		for b := 0; b < 256; b++ {
			dstH[b] = h0[b] + h1[b] + h2[b] + h3[b]
		}
	})

	var hist [256]int
	var sumAll float64
	for w := 0; w < numWorkers; w++ {
		srcH := &partHists[w]
		for b := 0; b < 256; b++ {
			hist[b] += srcH[b]
		}
	}
	for b := 0; b < 256; b++ {
		sumAll += float64(b * hist[b])
	}

	var sumB float64
	var wB int
	maxVar := 0.0
	bestSum := 0
	bestCount := 0

	for t := 0; t < 256; t++ {
		wB += hist[t]
		if wB == 0 {
			continue
		}
		wF := n - wB
		if wF == 0 {
			break
		}
		sumB += float64(t * hist[t])
		mB := sumB / float64(wB)
		mF := (sumAll - sumB) / float64(wF)
		diff := mB - mF
		varBetween := float64(wB) * float64(wF) * diff * diff
		if varBetween > maxVar+1e-9 {
			maxVar = varBetween
			bestSum = t
			bestCount = 1
		} else if maxVar > 0 && math.Abs(varBetween-maxVar) <= 1e-9 {
			bestSum += t
			bestCount++
		}
	}

	thresh := uint8(127)
	if bestCount > 0 {
		thresh = uint8(bestSum / bestCount)
	}

	out := NewBitMatrix(g.Width, g.Height)
	dst := out.Bits
	t := int32(thresh) + 1
	parallelFor(n, par, func(s0, s1 int) {
		p := pix[s0:s1]
		d := dst[s0:s1]
		m := len(p)
		j := 0
		for ; j+3 < m; j += 4 {
			d[j] = uint8(uint32(int32(p[j])-t) >> 31)
			d[j+1] = uint8(uint32(int32(p[j+1])-t) >> 31)
			d[j+2] = uint8(uint32(int32(p[j+2])-t) >> 31)
			d[j+3] = uint8(uint32(int32(p[j+3])-t) >> 31)
		}
		for ; j < m; j++ {
			d[j] = uint8(uint32(int32(p[j])-t) >> 31)
		}
	})
	return out
}

// LocalThresholdAt returns the local mean threshold in a windowSize x windowSize neighborhood
// around (x, y) for soft-confidence module sampling.
func LocalThresholdAt(g *GrayImage, integ *IntegralImage, x, y float64, windowSize int) float64 {
	if g == nil || g.Width <= 0 || g.Height <= 0 {
		return 0
	}
	if windowSize < 1 {
		windowSize = 15
	}
	var cx, cy int
	if x >= 0 {
		cx = int(x + 0.5)
	} else {
		cx = int(math.Round(x))
	}
	if y >= 0 {
		cy = int(y + 0.5)
	} else {
		cy = int(math.Round(y))
	}
	half := windowSize / 2
	x0 := cx - half
	y0 := cy - half
	x1 := x0 + windowSize
	y1 := y0 + windowSize
	w, h := g.Width, g.Height

	if integ != nil && integ.Width == w && integ.Height == h {
		stride := w + 1
		s := integ.Sum
		if uint(x0) <= uint(x1) && uint(x1) <= uint(w) && uint(y0) <= uint(y1) && uint(y1) <= uint(h) {
			i11 := y1*stride + x1
			if i11 < len(s) {
				sum := s[i11] - s[y0*stride+x1] - s[y1*stride+x0] + s[y0*stride+x0]
				return float64(sum) / float64(windowSize*windowSize)
			}
			return 0
		}
		if x0 < 0 {
			x0 = 0
		}
		if y0 < 0 {
			y0 = 0
		}
		if x1 > w {
			x1 = w
		}
		if y1 > h {
			y1 = h
		}
		if x1 <= x0 || y1 <= y0 {
			return float64(g.At(cx, cy))
		}
		i11 := y1*stride + x1
		if i11 < len(s) {
			sum := s[i11] - s[y0*stride+x1] - s[y1*stride+x0] + s[y0*stride+x0]
			return float64(sum) / float64((x1-x0)*(y1-y0))
		}
		return 0
	}

	if x0 < 0 {
		x0 = 0
	}
	if y0 < 0 {
		y0 = 0
	}
	if x1 > w {
		x1 = w
	}
	if y1 > h {
		y1 = h
	}
	if x1 <= x0 || y1 <= y0 {
		return float64(g.At(cx, cy))
	}
	var sum uint64
	pix := g.Pix
	for py := y0; py < y1; py++ {
		row := pix[py*w : py*w+x1]
		for px := x0; px < x1; px++ {
			sum += uint64(row[px])
		}
	}
	return float64(sum) / float64((x1-x0)*(y1-y0))
}
