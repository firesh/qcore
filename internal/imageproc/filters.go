package imageproc

import "math"

func clampUint8(v float64) uint8 {
	if v <= 0 {
		return 0
	}
	if v >= 255 {
		return 255
	}
	return uint8(math.Round(v))
}

func clampByteInt(v int32) uint8 {
	if v <= 0 {
		return 0
	}
	if v >= 255 {
		return 255
	}
	return uint8(v)
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

// CLAHE applies Contrast Limited Adaptive Histogram Equalization over a tileGrid x tileGrid
// partition with the given normalized clipLimit (e.g. 2.0–4.0).
func CLAHE(g *GrayImage, tileGrid int, clipLimit float64) *GrayImage {
	if g == nil || g.Width <= 0 || g.Height <= 0 {
		return NewGrayImage(0, 0)
	}
	w, h := g.Width, g.Height
	if tileGrid <= 0 {
		tileGrid = 8
	}
	gx := min(tileGrid, w)
	gy := min(tileGrid, h)
	par := w*h >= 131072

	xBounds := make([]int, gx+1)
	for i := 0; i <= gx; i++ {
		xBounds[i] = i * w / gx
	}
	yBounds := make([]int, gy+1)
	for j := 0; j <= gy; j++ {
		yBounds[j] = j * h / gy
	}

	luts := make([][256]float32, gx*gy)
	parallelFor(gy, par, func(ty0, ty1 int) {
		for ty := ty0; ty < ty1; ty++ {
			y0, y1 := yBounds[ty], yBounds[ty+1]
			for tx := 0; tx < gx; tx++ {
				x0, x1 := xBounds[tx], xBounds[tx+1]
				area := (x1 - x0) * (y1 - y0)
				if area <= 0 {
					continue
				}

				var hist [256]int
				span := x1 - x0
				for y := y0; y < y1; y++ {
					row := g.Pix[y*w+x0 : y*w+x1]
					x := 0
					for ; x+3 < span; x += 4 {
						hist[row[x]]++
						hist[row[x+1]]++
						hist[row[x+2]]++
						hist[row[x+3]]++
					}
					for ; x < span; x++ {
						hist[row[x]]++
					}
				}

				if clipLimit > 0 {
					clipVal := int(math.Round(clipLimit * float64(area) / 256.0))
					if clipVal < 1 {
						clipVal = 1
					}
					excess := 0
					for b := 0; b < 256; b++ {
						if hist[b] > clipVal {
							excess += hist[b] - clipVal
							hist[b] = clipVal
						}
					}
					avgInc := excess / 256
					rem := excess % 256
					if avgInc > 0 {
						for b := 0; b < 256; b++ {
							hist[b] += avgInc
						}
					}
					if rem > 0 {
						step := 256 / rem
						if step < 1 {
							step = 1
						}
						for i := 0; i < rem; i++ {
							bin := (i * step) % 256
							hist[bin]++
						}
					}
				}

				lut := &luts[ty*gx+tx]
				cdf := 0
				scale := float32(255.0 / float64(area))
				for b := 0; b < 256; b++ {
					cdf += hist[b]
					lut[b] = float32(cdf) * scale
				}
			}
		}
	})

	cx := make([]float64, gx)
	for tx := 0; tx < gx; tx++ {
		cx[tx] = 0.5 * float64(xBounds[tx]+xBounds[tx+1]-1)
	}
	cy := make([]float64, gy)
	for ty := 0; ty < gy; ty++ {
		cy[ty] = 0.5 * float64(yBounds[ty]+yBounds[ty+1]-1)
	}

	type interp1D struct {
		i0, i1 int
		f      float32
	}
	xMap := make([]interp1D, w)
	for x := 0; x < w; x++ {
		fx := float64(x)
		if gx == 1 || fx <= cx[0] {
			xMap[x] = interp1D{0, 0, 0}
		} else if fx >= cx[gx-1] {
			xMap[x] = interp1D{gx - 1, gx - 1, 0}
		} else {
			tx := 0
			for tx+1 < gx && cx[tx+1] < fx {
				tx++
			}
			span := cx[tx+1] - cx[tx]
			var alpha float32
			if span > 0 {
				alpha = float32((fx - cx[tx]) / span)
			}
			xMap[x] = interp1D{tx, tx + 1, alpha}
		}
	}

	out := NewGrayImage(w, h)
	_ = xMap[w-1]
	parallelFor(h, par, func(y0, y1 int) {
		for y := y0; y < y1; y++ {
			fy := float64(y)
			var ty0, ty1 int
			var alphaY float32
			if gy == 1 || fy <= cy[0] {
				ty0, ty1, alphaY = 0, 0, 0
			} else if fy >= cy[gy-1] {
				ty0, ty1, alphaY = gy-1, gy-1, 0
			} else {
				ty := 0
				for ty+1 < gy && cy[ty+1] < fy {
					ty++
				}
				span := cy[ty+1] - cy[ty]
				if span > 0 {
					alphaY = float32((fy - cy[ty]) / span)
				}
				ty0, ty1 = ty, ty+1
			}

			rowLuts0 := luts[ty0*gx : ty0*gx+gx]
			rowLuts1 := luts[ty1*gx : ty1*gx+gx]
			srcRow := g.Pix[y*w : y*w+w]
			dstRow := out.Pix[y*w : y*w+w]
			_ = srcRow[w-1]
			_ = dstRow[w-1]

			for x := 0; x < w; x++ {
				v := srcRow[x]
				xm := xMap[x]
				v00 := rowLuts0[xm.i0][v]
				v10 := rowLuts0[xm.i1][v]
				v01 := rowLuts1[xm.i0][v]
				v11 := rowLuts1[xm.i1][v]
				top := v00 + xm.f*(v10-v00)
				bot := v01 + xm.f*(v11-v01)
				dstRow[x] = uint8(top + alphaY*(bot-top) + 0.5)
			}
		}
	})
	return out
}

// GaussianBlur3x3 applies a 3x3 [1 2 1; 2 4 2; 1 2 1]/16 binomial Gaussian filter
// with replicated border clamping.
func GaussianBlur3x3(g *GrayImage) *GrayImage {
	if g == nil || g.Width <= 0 || g.Height <= 0 {
		return NewGrayImage(0, 0)
	}
	w, h := g.Width, g.Height
	out := NewGrayImage(w, h)
	if w < 3 || h < 3 {
		for y := 0; y < h; y++ {
			r0 := g.Pix[max(0, y-1)*w:]
			r1 := g.Pix[y*w:]
			r2 := g.Pix[min(h-1, y+1)*w:]
			dst := out.Pix[y*w:]
			for x := 0; x < w; x++ {
				xm := max(0, x-1)
				xp := min(w-1, x+1)
				sum := int(r0[xm]) + 2*int(r0[x]) + int(r0[xp]) +
					2*int(r1[xm]) + 4*int(r1[x]) + 2*int(r1[xp]) +
					int(r2[xm]) + 2*int(r2[x]) + int(r2[xp])
				dst[x] = uint8((sum + 8) >> 4)
			}
		}
		return out
	}

	par := w*h >= 131072
	parallelFor(h, par, func(y0, y1 int) {
		for y := y0; y < y1; y++ {
			r0 := g.Pix[max(0, y-1)*w : max(0, y-1)*w+w]
			r1 := g.Pix[y*w : y*w+w]
			r2 := g.Pix[min(h-1, y+1)*w : min(h-1, y+1)*w+w]
			dst := out.Pix[y*w : y*w+w]
			_ = r0[w-1]
			_ = r1[w-1]
			_ = r2[w-1]
			_ = dst[w-1]

			dst[0] = uint8((3*int(r0[0]) + int(r0[1]) + 6*int(r1[0]) + 2*int(r1[1]) + 3*int(r2[0]) + int(r2[1]) + 8) >> 4)
			for x := 1; x < w-1; x++ {
				sum := int(r0[x-1]) + 2*int(r0[x]) + int(r0[x+1]) +
					2*int(r1[x-1]) + 4*int(r1[x]) + 2*int(r1[x+1]) +
					int(r2[x-1]) + 2*int(r2[x]) + int(r2[x+1])
				dst[x] = uint8((sum + 8) >> 4)
			}
			last := w - 1
			dst[last] = uint8((int(r0[last-1]) + 3*int(r0[last]) + 2*int(r1[last-1]) + 6*int(r1[last]) + int(r2[last-1]) + 3*int(r2[last]) + 8) >> 4)
		}
	})
	return out
}

// UnsharpMask sharpens g using out = g + amount*(g - GaussianBlur3x3(g)).
func UnsharpMask(g *GrayImage, amount float64) *GrayImage {
	if g == nil || g.Width <= 0 || g.Height <= 0 {
		return NewGrayImage(0, 0)
	}
	if amount == 0 {
		return g.Clone()
	}
	w, h := g.Width, g.Height
	out := NewGrayImage(w, h)
	amtFP := int32(math.Round(amount * 256.0))

	if w < 3 || h < 3 {
		for y := 0; y < h; y++ {
			r0 := g.Pix[max(0, y-1)*w:]
			r1 := g.Pix[y*w:]
			r2 := g.Pix[min(h-1, y+1)*w:]
			dst := out.Pix[y*w:]
			for x := 0; x < w; x++ {
				xm := max(0, x-1)
				xp := min(w-1, x+1)
				sum := int32(r0[xm]) + 2*int32(r0[x]) + int32(r0[xp]) +
					2*int32(r1[xm]) + 4*int32(r1[x]) + int32(r1[xp]) +
					int32(r2[xm]) + 2*int32(r2[x]) + int32(r2[xp])
				blur := (sum + 8) >> 4
				p := int32(r1[x])
				dst[x] = clampByteInt(p + ((amtFP*(p-blur) + 128) >> 8))
			}
		}
		return out
	}

	par := w*h >= 131072
	parallelFor(h, par, func(y0, y1 int) {
		for y := y0; y < y1; y++ {
			r0 := g.Pix[max(0, y-1)*w : max(0, y-1)*w+w]
			r1 := g.Pix[y*w : y*w+w]
			r2 := g.Pix[min(h-1, y+1)*w : min(h-1, y+1)*w+w]
			dst := out.Pix[y*w : y*w+w]
			_ = r0[w-1]
			_ = r1[w-1]
			_ = r2[w-1]
			_ = dst[w-1]

			p0 := int32(r1[0])
			b0 := (3*int32(r0[0]) + int32(r0[1]) + 6*p0 + 2*int32(r1[1]) + 3*int32(r2[0]) + int32(r2[1]) + 8) >> 4
			dst[0] = clampByteInt(p0 + ((amtFP*(p0-b0) + 128) >> 8))

			for x := 1; x < w-1; x++ {
				p := int32(r1[x])
				sum := int32(r0[x-1]) + 2*int32(r0[x]) + int32(r0[x+1]) +
					2*int32(r1[x-1]) + 4*p + 2*int32(r1[x+1]) +
					int32(r2[x-1]) + 2*int32(r2[x]) + int32(r2[x+1])
				blur := (sum + 8) >> 4
				dst[x] = clampByteInt(p + ((amtFP*(p-blur) + 128) >> 8))
			}

			last := w - 1
			pl := int32(r1[last])
			bl := (int32(r0[last-1]) + 3*int32(r0[last]) + 2*int32(r1[last-1]) + 6*pl + int32(r2[last-1]) + 3*int32(r2[last]) + 8) >> 4
			dst[last] = clampByteInt(pl + ((amtFP*(pl-bl) + 128) >> 8))
		}
	})
	return out
}

// LaplacianSharpen sharpens g using a 4-neighbor discrete Laplacian kernel:
// out = p + strength * (4*p - up - down - left - right).
func LaplacianSharpen(g *GrayImage, strength float64) *GrayImage {
	if g == nil || g.Width <= 0 || g.Height <= 0 {
		return NewGrayImage(0, 0)
	}
	if strength == 0 {
		return g.Clone()
	}
	w, h := g.Width, g.Height
	out := NewGrayImage(w, h)
	strFP := int32(math.Round(strength * 256.0))

	if w < 3 || h < 3 {
		for y := 0; y < h; y++ {
			rUp := g.Pix[max(0, y-1)*w:]
			rMid := g.Pix[y*w:]
			rDown := g.Pix[min(h-1, y+1)*w:]
			dst := out.Pix[y*w:]
			for x := 0; x < w; x++ {
				xm := max(0, x-1)
				xp := min(w-1, x+1)
				center := int32(rMid[x])
				lap := 4*center - int32(rUp[x]) - int32(rDown[x]) - int32(rMid[xm]) - int32(rMid[xp])
				dst[x] = clampByteInt(center + ((strFP*lap + 128) >> 8))
			}
		}
		return out
	}

	par := w*h >= 131072
	parallelFor(h, par, func(y0, y1 int) {
		for y := y0; y < y1; y++ {
			rUp := g.Pix[max(0, y-1)*w : max(0, y-1)*w+w]
			rMid := g.Pix[y*w : y*w+w]
			rDown := g.Pix[min(h-1, y+1)*w : min(h-1, y+1)*w+w]
			dst := out.Pix[y*w : y*w+w]
			_ = rUp[w-1]
			_ = rMid[w-1]
			_ = rDown[w-1]
			_ = dst[w-1]

			c0 := int32(rMid[0])
			lap0 := 3*c0 - int32(rUp[0]) - int32(rDown[0]) - int32(rMid[1])
			dst[0] = clampByteInt(c0 + ((strFP*lap0 + 128) >> 8))

			for x := 1; x < w-1; x++ {
				c := int32(rMid[x])
				lap := 4*c - int32(rUp[x]) - int32(rDown[x]) - int32(rMid[x-1]) - int32(rMid[x+1])
				dst[x] = clampByteInt(c + ((strFP*lap + 128) >> 8))
			}

			last := w - 1
			cl := int32(rMid[last])
			lapL := 3*cl - int32(rUp[last]) - int32(rDown[last]) - int32(rMid[last-1])
			dst[last] = clampByteInt(cl + ((strFP*lapL + 128) >> 8))
		}
	})
	return out
}

// median9 returns the exact median of 9 bytes using Paeth's 19-operation branchless network.
func median9(p0, p1, p2, p3, p4, p5, p6, p7, p8 uint8) uint8 {
	if p1 > p2 {
		p1, p2 = p2, p1
	}
	if p0 > p1 {
		p0, p1 = p1, p0
	}
	if p1 > p2 {
		p1, p2 = p2, p1
	}
	if p4 > p5 {
		p4, p5 = p5, p4
	}
	if p3 > p4 {
		p3, p4 = p4, p3
	}
	if p4 > p5 {
		p4, p5 = p5, p4
	}
	if p7 > p8 {
		p7, p8 = p8, p7
	}
	if p6 > p7 {
		p6, p7 = p7, p6
	}
	if p7 > p8 {
		p7, p8 = p8, p7
	}

	if p0 > p3 {
		p3 = p0
	}
	if p3 > p6 {
		p6 = p3
	}
	if p5 < p8 {
		p8 = p5
	}
	if p8 < p2 {
		p2 = p8
	}
	if p4 > p7 {
		p4, p7 = p7, p4
	}
	if p1 > p4 {
		p4 = p1
	}
	if p4 > p7 {
		p4 = p7
	}
	if p4 > p2 {
		p4, p2 = p2, p4
	}
	if p6 > p4 {
		p4 = p6
	}
	if p4 > p2 {
		p4 = p2
	}
	return p4
}

// MedianFilter3x3 applies a 3x3 median filter with replicated border clamping.
func MedianFilter3x3(g *GrayImage) *GrayImage {
	if g == nil || g.Width <= 0 || g.Height <= 0 {
		return NewGrayImage(0, 0)
	}
	w, h := g.Width, g.Height
	out := NewGrayImage(w, h)

	if w < 3 || h < 3 {
		for y := 0; y < h; y++ {
			r0 := g.Pix[max(0, y-1)*w:]
			r1 := g.Pix[y*w:]
			r2 := g.Pix[min(h-1, y+1)*w:]
			dst := out.Pix[y*w:]
			for x := 0; x < w; x++ {
				xm := max(0, x-1)
				xp := min(w-1, x+1)
				dst[x] = median9(r0[xm], r0[x], r0[xp], r1[xm], r1[x], r1[xp], r2[xm], r2[x], r2[xp])
			}
		}
		return out
	}

	par := w*h >= 65536
	parallelFor(h, par, func(y0, y1 int) {
		for y := y0; y < y1; y++ {
			r0 := g.Pix[max(0, y-1)*w : max(0, y-1)*w+w]
			r1 := g.Pix[y*w : y*w+w]
			r2 := g.Pix[min(h-1, y+1)*w : min(h-1, y+1)*w+w]
			dst := out.Pix[y*w : y*w+w]
			_ = r0[w-1]
			_ = r1[w-1]
			_ = r2[w-1]
			_ = dst[w-1]

			dst[0] = median9(r0[0], r0[0], r0[1], r1[0], r1[0], r1[1], r2[0], r2[0], r2[1])
			for x := 1; x < w-1; x++ {
				dst[x] = median9(
					r0[x-1], r0[x], r0[x+1],
					r1[x-1], r1[x], r1[x+1],
					r2[x-1], r2[x], r2[x+1],
				)
			}
			last := w - 1
			dst[last] = median9(
				r0[last-1], r0[last], r0[last],
				r1[last-1], r1[last], r1[last],
				r2[last-1], r2[last], r2[last],
			)
		}
	})
	return out
}

func catmullRomWeight32(d float64) float32 {
	t := math.Abs(d)
	if t <= 1.0 {
		return float32((1.5*t-2.5)*t*t + 1.0)
	}
	if t < 2.0 {
		return float32(((-0.5*t+2.5)*t-4.0)*t + 2.0)
	}
	return 0.0
}

// UpscaleBicubic resizes g by scale using separable Catmull-Rom bicubic interpolation.
func UpscaleBicubic(g *GrayImage, scale float64) *GrayImage {
	if g == nil || g.Width <= 0 || g.Height <= 0 || scale <= 0 {
		return NewGrayImage(0, 0)
	}
	if scale == 1.0 {
		return g.Clone()
	}
	srcW, srcH := g.Width, g.Height
	dstW := max(1, int(math.Round(float64(srcW)*scale)))
	dstH := max(1, int(math.Round(float64(srcH)*scale)))
	par := dstW*dstH >= 131072

	type tap4F struct {
		ix             int
		w0, w1, w2, w3 float32
	}
	xTaps := make([]tap4F, dstW)
	ratioX := float64(srcW) / float64(dstW)
	dxIntStart := dstW
	dxIntEnd := 0

	for dx := 0; dx < dstW; dx++ {
		sx := (float64(dx)+0.5)*ratioX - 0.5
		ix := int(math.Floor(sx))
		fx := sx - float64(ix)
		xTaps[dx] = tap4F{
			ix: ix,
			w0: catmullRomWeight32(fx + 1.0),
			w1: catmullRomWeight32(fx),
			w2: catmullRomWeight32(fx - 1.0),
			w3: catmullRomWeight32(fx - 2.0),
		}
		if ix >= 1 && ix+2 < srcW {
			if dx < dxIntStart {
				dxIntStart = dx
			}
			dxIntEnd = dx + 1
		}
	}
	if dxIntStart > dxIntEnd {
		dxIntStart = dstW
		dxIntEnd = dstW
	}

	// Horizontal pass into float32 intermediate buffer (dstW x srcH) to leverage FMA.
	tmp := make([]float32, dstW*srcH)
	_ = xTaps[dstW-1]
	parallelFor(srcH, par, func(y0, y1 int) {
		for y := y0; y < y1; y++ {
			srcRow := g.Pix[y*srcW : y*srcW+srcW]
			dstRow := tmp[y*dstW : y*dstW+dstW]
			_ = srcRow[srcW-1]
			_ = dstRow[dstW-1]

			for dx := 0; dx < dxIntStart; dx++ {
				t := xTaps[dx]
				i0 := clampInt(t.ix-1, 0, srcW-1)
				i1 := clampInt(t.ix, 0, srcW-1)
				i2 := clampInt(t.ix+1, 0, srcW-1)
				i3 := clampInt(t.ix+2, 0, srcW-1)
				dstRow[dx] = t.w0*float32(srcRow[i0]) + t.w1*float32(srcRow[i1]) +
					t.w2*float32(srcRow[i2]) + t.w3*float32(srcRow[i3])
			}
			for dx := dxIntStart; dx < dxIntEnd; dx++ {
				t := xTaps[dx]
				s := srcRow[t.ix-1 : t.ix+3]
				_ = s[3]
				dstRow[dx] = t.w0*float32(s[0]) + t.w1*float32(s[1]) +
					t.w2*float32(s[2]) + t.w3*float32(s[3])
			}
			for dx := dxIntEnd; dx < dstW; dx++ {
				t := xTaps[dx]
				i0 := clampInt(t.ix-1, 0, srcW-1)
				i1 := clampInt(t.ix, 0, srcW-1)
				i2 := clampInt(t.ix+1, 0, srcW-1)
				i3 := clampInt(t.ix+2, 0, srcW-1)
				dstRow[dx] = t.w0*float32(srcRow[i0]) + t.w1*float32(srcRow[i1]) +
					t.w2*float32(srcRow[i2]) + t.w3*float32(srcRow[i3])
			}
		}
	})

	// Vertical pass into uint8 output (dstW x dstH).
	out := NewGrayImage(dstW, dstH)
	ratioY := float64(srcH) / float64(dstH)
	parallelFor(dstH, par, func(dy0, dy1 int) {
		for dy := dy0; dy < dy1; dy++ {
			sy := (float64(dy)+0.5)*ratioY - 0.5
			iy := int(math.Floor(sy))
			fy := sy - float64(iy)
			w0 := catmullRomWeight32(fy + 1.0)
			w1 := catmullRomWeight32(fy)
			w2 := catmullRomWeight32(fy - 1.0)
			w3 := catmullRomWeight32(fy - 2.0)

			r0 := tmp[clampInt(iy-1, 0, srcH-1)*dstW : clampInt(iy-1, 0, srcH-1)*dstW+dstW]
			r1 := tmp[clampInt(iy, 0, srcH-1)*dstW : clampInt(iy, 0, srcH-1)*dstW+dstW]
			r2 := tmp[clampInt(iy+1, 0, srcH-1)*dstW : clampInt(iy+1, 0, srcH-1)*dstW+dstW]
			r3 := tmp[clampInt(iy+2, 0, srcH-1)*dstW : clampInt(iy+2, 0, srcH-1)*dstW+dstW]
			dstRow := out.Pix[dy*dstW : dy*dstW+dstW]
			_ = r0[dstW-1]
			_ = r1[dstW-1]
			_ = r2[dstW-1]
			_ = r3[dstW-1]
			_ = dstRow[dstW-1]

			for dx := 0; dx < dstW; dx++ {
				v := w0*r0[dx] + w1*r1[dx] + w2*r2[dx] + w3*r3[dx]
				if v <= 0 {
					dstRow[dx] = 0
				} else if v >= 255 {
					dstRow[dx] = 255
				} else {
					dstRow[dx] = uint8(v + 0.5)
				}
			}
		}
	})
	return out
}

// DownscaleArea resizes g by scale using fractional area averaging over each destination pixel's
// source footprint.
func DownscaleArea(g *GrayImage, scale float64) *GrayImage {
	if g == nil || g.Width <= 0 || g.Height <= 0 || scale <= 0 {
		return NewGrayImage(0, 0)
	}
	if scale == 1.0 {
		return g.Clone()
	}
	srcW, srcH := g.Width, g.Height
	dstW := max(1, int(math.Round(float64(srcW)*scale)))
	dstH := max(1, int(math.Round(float64(srcH)*scale)))
	par := srcW*srcH >= 131072

	// Exact 2x2 integer box downscale fast path (zero temp allocations).
	if srcW == dstW*2 && srcH == dstH*2 {
		out := NewGrayImage(dstW, dstH)
		parallelFor(dstH, par, func(dy0, dy1 int) {
			for dy := dy0; dy < dy1; dy++ {
				r0 := g.Pix[(dy*2)*srcW : (dy*2)*srcW+srcW]
				r1 := g.Pix[(dy*2+1)*srcW : (dy*2+1)*srcW+srcW]
				dstRow := out.Pix[dy*dstW : dy*dstW+dstW]
				_ = r0[srcW-1]
				_ = r1[srcW-1]
				_ = dstRow[dstW-1]
				for dx, sx := 0, 0; dx < dstW; dx, sx = dx+1, sx+2 {
					dstRow[dx] = uint8((uint32(r0[sx]) + uint32(r0[sx+1]) + uint32(r1[sx]) + uint32(r1[sx+1]) + 2) >> 2)
				}
			}
		})
		return out
	}

	// Exact integer kx*ky box downscale fast path (zero temp allocations).
	if srcW%dstW == 0 && srcH%dstH == 0 {
		kx := srcW / dstW
		ky := srcH / dstH
		area := uint32(kx * ky)
		halfArea := area >> 1
		out := NewGrayImage(dstW, dstH)
		parallelFor(dstH, par, func(dy0, dy1 int) {
			for dy := dy0; dy < dy1; dy++ {
				sy0 := dy * ky
				dstRow := out.Pix[dy*dstW : dy*dstW+dstW]
				_ = dstRow[dstW-1]
				for dx := 0; dx < dstW; dx++ {
					sx0 := dx * kx
					var sum uint32
					for sy := sy0; sy < sy0+ky; sy++ {
						row := g.Pix[sy*srcW+sx0 : sy*srcW+sx0+kx]
						for i := 0; i < kx; i++ {
							sum += uint32(row[i])
						}
					}
					dstRow[dx] = uint8((sum + halfArea) / area)
				}
			}
		})
		return out
	}

	// General separable 2-pass float32 area downscaler.
	ratioX := float64(srcW) / float64(dstW)
	ratioY := float64(srcH) / float64(dstH)

	type span1D struct {
		start int
		end   int
		w0    float32
		wEnd  float32
		invW  float32
	}
	xSpans := make([]span1D, dstW)
	for dx := 0; dx < dstW; dx++ {
		sx0 := float64(dx) * ratioX
		sx1 := float64(dx+1) * ratioX
		if sx1 > float64(srcW) {
			sx1 = float64(srcW)
		}
		i0 := clampInt(int(math.Floor(sx0)), 0, srcW-1)
		i1 := clampInt(int(math.Ceil(sx1))-1, i0, srcW-1)
		var w0, wEnd float32
		if i0 == i1 {
			w0 = float32(sx1 - sx0)
			wEnd = w0
		} else {
			w0 = float32(float64(i0+1) - sx0)
			wEnd = float32(sx1 - float64(i1))
		}
		spanW := float32(sx1 - sx0)
		var invW float32
		if spanW > 0 {
			invW = 1.0 / spanW
		}
		xSpans[dx] = span1D{start: i0, end: i1, w0: w0, wEnd: wEnd, invW: invW}
	}

	// Pass 1: horizontal area average -> float32 (dstW x srcH).
	tmp := make([]float32, dstW*srcH)
	_ = xSpans[dstW-1]
	parallelFor(srcH, par, func(sy0, sy1 int) {
		for sy := sy0; sy < sy1; sy++ {
			srcRow := g.Pix[sy*srcW : sy*srcW+srcW]
			tmpRow := tmp[sy*dstW : sy*dstW+dstW]
			_ = srcRow[srcW-1]
			_ = tmpRow[dstW-1]
			for dx := 0; dx < dstW; dx++ {
				xs := xSpans[dx]
				if xs.start == xs.end {
					tmpRow[dx] = float32(srcRow[xs.start])
				} else {
					sum := float32(srcRow[xs.start]) * xs.w0
					for sx := xs.start + 1; sx < xs.end; sx++ {
						sum += float32(srcRow[sx])
					}
					sum += float32(srcRow[xs.end]) * xs.wEnd
					tmpRow[dx] = sum * xs.invW
				}
			}
		}
	})

	// Pass 2: vertical area average -> uint8 (dstW x dstH), sequential across dx.
	out := NewGrayImage(dstW, dstH)
	parallelFor(dstH, par, func(dy0, dy1 int) {
		acc := make([]float32, dstW)
		_ = acc[dstW-1]
		for dy := dy0; dy < dy1; dy++ {
			sy0 := float64(dy) * ratioY
			sy1 := float64(dy+1) * ratioY
			if sy1 > float64(srcH) {
				sy1 = float64(srcH)
			}
			j0 := clampInt(int(math.Floor(sy0)), 0, srcH-1)
			j1 := clampInt(int(math.Ceil(sy1))-1, j0, srcH-1)
			invH := float32(1.0 / (sy1 - sy0))

			if j0 == j1 {
				tmpRow := tmp[j0*dstW : j0*dstW+dstW]
				dstRow := out.Pix[dy*dstW : dy*dstW+dstW]
				_ = tmpRow[dstW-1]
				_ = dstRow[dstW-1]
				for dx := 0; dx < dstW; dx++ {
					dstRow[dx] = uint8(tmpRow[dx] + 0.5)
				}
				continue
			}

			wy0 := float32(float64(j0+1) - sy0)
			row0 := tmp[j0*dstW : j0*dstW+dstW]
			_ = row0[dstW-1]
			for dx := 0; dx < dstW; dx++ {
				acc[dx] = row0[dx] * wy0
			}

			for sy := j0 + 1; sy < j1; sy++ {
				tRow := tmp[sy*dstW : sy*dstW+dstW]
				_ = tRow[dstW-1]
				for dx := 0; dx < dstW; dx++ {
					acc[dx] += tRow[dx]
				}
			}

			wyLast := float32(sy1 - float64(j1))
			rowLast := tmp[j1*dstW : j1*dstW+dstW]
			dstRow := out.Pix[dy*dstW : dy*dstW+dstW]
			_ = rowLast[dstW-1]
			_ = dstRow[dstW-1]
			for dx := 0; dx < dstW; dx++ {
				v := (acc[dx] + rowLast[dx]*wyLast) * invH
				if v <= 0 {
					dstRow[dx] = 0
				} else if v >= 255 {
					dstRow[dx] = 255
				} else {
					dstRow[dx] = uint8(v + 0.5)
				}
			}
		}
	})
	return out
}
