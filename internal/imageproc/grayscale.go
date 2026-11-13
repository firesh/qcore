package imageproc

import (
	"bytes"
	"encoding/binary"
	"image"
	"image/color"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"math"
	"os"
)

// ChannelMode selects which color channel or projection to extract into a GrayImage.
type ChannelMode int

const (
	ChannelLuma ChannelMode = iota
	ChannelRed
	ChannelGreen
	ChannelBlue
	ChannelSaturation
)

// GrayImage stores an 8-bit single-channel grayscale image in row-major order.
type GrayImage struct {
	Pix    []uint8
	Width  int
	Height int
}

// NewGrayImage allocates a new GrayImage of size w x h initialized to zeros.
func NewGrayImage(w, h int) *GrayImage {
	if w <= 0 || h <= 0 {
		return &GrayImage{Width: 0, Height: 0}
	}
	return &GrayImage{
		Pix:    make([]uint8, w*h),
		Width:  w,
		Height: h,
	}
}

// FromImage converts any image.Image to a luminance GrayImage.
func FromImage(img image.Image) *GrayImage {
	return ExtractChannel(img, ChannelLuma)
}

// ExtractChannel converts an image.Image into a single-channel GrayImage using mode.
func ExtractChannel(img image.Image, mode ChannelMode) *GrayImage {
	if img == nil {
		return NewGrayImage(0, 0)
	}
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	if w <= 0 || h <= 0 {
		return NewGrayImage(0, 0)
	}
	out := NewGrayImage(w, h)

	switch src := img.(type) {
	case *image.Gray:
		extractFromGray(src, b, out, mode)
		return out
	case *image.YCbCr:
		extractFromYCbCr(src, b, out, mode)
		return out
	case *image.RGBA:
		extractFromRGBA(src, b, out, mode)
		return out
	case *image.NRGBA:
		extractFromNRGBA(src, b, out, mode)
		return out
	}

	// Fallback for arbitrary image.Image implementations.
	for y := 0; y < h; y++ {
		sy := b.Min.Y + y
		rowOff := y * w
		for x := 0; x < w; x++ {
			r32, g32, b32, _ := img.At(b.Min.X+x, sy).RGBA()
			r := uint8(r32 >> 8)
			g := uint8(g32 >> 8)
			bl := uint8(b32 >> 8)
			out.Pix[rowOff+x] = channelValue(r, g, bl, mode)
		}
	}
	return out
}

func extractFromGray(src *image.Gray, b image.Rectangle, out *GrayImage, mode ChannelMode) {
	w, h := out.Width, out.Height
	if mode == ChannelSaturation {
		// Grayscale images have zero saturation everywhere.
		return
	}
	for y := 0; y < h; y++ {
		sOff := src.PixOffset(b.Min.X, b.Min.Y+y)
		dOff := y * w
		copy(out.Pix[dOff:dOff+w], src.Pix[sOff:sOff+w])
	}
}

func extractFromYCbCr(src *image.YCbCr, b image.Rectangle, out *GrayImage, mode ChannelMode) {
	w, h := out.Width, out.Height
	if mode == ChannelLuma {
		for y := 0; y < h; y++ {
			yOff := src.YOffset(b.Min.X, b.Min.Y+y)
			dOff := y * w
			copy(out.Pix[dOff:dOff+w], src.Y[yOff:yOff+w])
		}
		return
	}
	for y := 0; y < h; y++ {
		sy := b.Min.Y + y
		dOff := y * w
		for x := 0; x < w; x++ {
			sx := b.Min.X + x
			yi := src.YOffset(sx, sy)
			ci := src.COffset(sx, sy)
			r, g, bl := color.YCbCrToRGB(src.Y[yi], src.Cb[ci], src.Cr[ci])
			out.Pix[dOff+x] = channelValue(r, g, bl, mode)
		}
	}
}

func extractFromRGBA(src *image.RGBA, b image.Rectangle, out *GrayImage, mode ChannelMode) {
	w, h := out.Width, out.Height
	rowBytes := w * 4
	par := w*h >= 131072

	switch mode {
	case ChannelLuma:
		parallelFor(h, par, func(y0, y1 int) {
			for y := y0; y < y1; y++ {
				sOff := src.PixOffset(b.Min.X, b.Min.Y+y)
				row := src.Pix[sOff : sOff+rowBytes]
				dst := out.Pix[y*w : y*w+w]
				_ = row[rowBytes-1]
				_ = dst[w-1]
				x := 0
				p := 0
				for ; x+3 < w; x, p = x+4, p+16 {
					dst[x] = uint8((19595*uint32(row[p]) + 38470*uint32(row[p+1]) + 7471*uint32(row[p+2]) + 1<<15) >> 16)
					dst[x+1] = uint8((19595*uint32(row[p+4]) + 38470*uint32(row[p+5]) + 7471*uint32(row[p+6]) + 1<<15) >> 16)
					dst[x+2] = uint8((19595*uint32(row[p+8]) + 38470*uint32(row[p+9]) + 7471*uint32(row[p+10]) + 1<<15) >> 16)
					dst[x+3] = uint8((19595*uint32(row[p+12]) + 38470*uint32(row[p+13]) + 7471*uint32(row[p+14]) + 1<<15) >> 16)
				}
				for ; x < w; x, p = x+1, p+4 {
					dst[x] = uint8((19595*uint32(row[p]) + 38470*uint32(row[p+1]) + 7471*uint32(row[p+2]) + 1<<15) >> 16)
				}
			}
		})
	case ChannelRed, ChannelGreen, ChannelBlue:
		ch := int(mode - ChannelRed) // 0=R, 1=G, 2=B
		parallelFor(h, par, func(y0, y1 int) {
			for y := y0; y < y1; y++ {
				sOff := src.PixOffset(b.Min.X, b.Min.Y+y)
				row := src.Pix[sOff : sOff+rowBytes]
				dst := out.Pix[y*w : y*w+w]
				_ = row[rowBytes-1]
				_ = dst[w-1]
				for x, p := 0, 0; x < w; x, p = x+1, p+4 {
					a := row[p+3]
					v := row[p+ch]
					if a != 255 {
						if a == 0 {
							v = 0
						} else {
							v = uint8(min(255, int(v)*255/int(a)))
						}
					}
					dst[x] = v
				}
			}
		})
	default: // ChannelSaturation
		parallelFor(h, par, func(y0, y1 int) {
			for y := y0; y < y1; y++ {
				sOff := src.PixOffset(b.Min.X, b.Min.Y+y)
				row := src.Pix[sOff : sOff+rowBytes]
				dst := out.Pix[y*w : y*w+w]
				_ = row[rowBytes-1]
				_ = dst[w-1]
				for x, p := 0, 0; x < w; x, p = x+1, p+4 {
					r, g, bl, a := row[p], row[p+1], row[p+2], row[p+3]
					if a != 255 {
						if a == 0 {
							dst[x] = 0
							continue
						}
						r = uint8(min(255, int(r)*255/int(a)))
						g = uint8(min(255, int(g)*255/int(a)))
						bl = uint8(min(255, int(bl)*255/int(a)))
					}
					dst[x] = channelValue(r, g, bl, ChannelSaturation)
				}
			}
		})
	}
}

func extractFromNRGBA(src *image.NRGBA, b image.Rectangle, out *GrayImage, mode ChannelMode) {
	w, h := out.Width, out.Height
	rowBytes := w * 4
	par := w*h >= 131072

	switch mode {
	case ChannelLuma:
		parallelFor(h, par, func(y0, y1 int) {
			for y := y0; y < y1; y++ {
				sOff := src.PixOffset(b.Min.X, b.Min.Y+y)
				row := src.Pix[sOff : sOff+rowBytes]
				dst := out.Pix[y*w : y*w+w]
				_ = row[rowBytes-1]
				_ = dst[w-1]
				x := 0
				p := 0
				for ; x+3 < w; x, p = x+4, p+16 {
					dst[x] = uint8((19595*uint32(row[p]) + 38470*uint32(row[p+1]) + 7471*uint32(row[p+2]) + 1<<15) >> 16)
					dst[x+1] = uint8((19595*uint32(row[p+4]) + 38470*uint32(row[p+5]) + 7471*uint32(row[p+6]) + 1<<15) >> 16)
					dst[x+2] = uint8((19595*uint32(row[p+8]) + 38470*uint32(row[p+9]) + 7471*uint32(row[p+10]) + 1<<15) >> 16)
					dst[x+3] = uint8((19595*uint32(row[p+12]) + 38470*uint32(row[p+13]) + 7471*uint32(row[p+14]) + 1<<15) >> 16)
				}
				for ; x < w; x, p = x+1, p+4 {
					dst[x] = uint8((19595*uint32(row[p]) + 38470*uint32(row[p+1]) + 7471*uint32(row[p+2]) + 1<<15) >> 16)
				}
			}
		})
	case ChannelRed, ChannelGreen, ChannelBlue:
		ch := int(mode - ChannelRed) // 0=R, 1=G, 2=B
		parallelFor(h, par, func(y0, y1 int) {
			for y := y0; y < y1; y++ {
				sOff := src.PixOffset(b.Min.X, b.Min.Y+y)
				row := src.Pix[sOff : sOff+rowBytes]
				dst := out.Pix[y*w : y*w+w]
				_ = row[rowBytes-1]
				_ = dst[w-1]
				x := 0
				p := ch
				for ; x+3 < w; x, p = x+4, p+16 {
					dst[x] = row[p]
					dst[x+1] = row[p+4]
					dst[x+2] = row[p+8]
					dst[x+3] = row[p+12]
				}
				for ; x < w; x, p = x+1, p+4 {
					dst[x] = row[p]
				}
			}
		})
	default: // ChannelSaturation
		parallelFor(h, par, func(y0, y1 int) {
			for y := y0; y < y1; y++ {
				sOff := src.PixOffset(b.Min.X, b.Min.Y+y)
				row := src.Pix[sOff : sOff+rowBytes]
				dst := out.Pix[y*w : y*w+w]
				_ = row[rowBytes-1]
				_ = dst[w-1]
				for x, p := 0, 0; x < w; x, p = x+1, p+4 {
					dst[x] = channelValue(row[p], row[p+1], row[p+2], ChannelSaturation)
				}
			}
		})
	}
}

func channelValue(r, g, b uint8, mode ChannelMode) uint8 {
	switch mode {
	case ChannelLuma:
		return uint8((19595*uint32(r) + 38470*uint32(g) + 7471*uint32(b) + 1<<15) >> 16)
	case ChannelRed:
		return r
	case ChannelGreen:
		return g
	case ChannelBlue:
		return b
	case ChannelSaturation:
		mx := r
		if g > mx {
			mx = g
		}
		if b > mx {
			mx = b
		}
		if mx == 0 {
			return 0
		}
		mn := r
		if g < mn {
			mn = g
		}
		if b < mn {
			mn = b
		}
		return uint8((int(mx-mn) * 255) / int(mx))
	default:
		return uint8((19595*uint32(r) + 38470*uint32(g) + 7471*uint32(b) + 1<<15) >> 16)
	}
}

// At returns the pixel value at (x, y), clamping coordinates to image bounds.
// Returns 0 if the image is empty.
func (g *GrayImage) At(x, y int) uint8 {
	if g == nil || g.Width <= 0 || g.Height <= 0 || len(g.Pix) == 0 {
		return 0
	}
	if x < 0 {
		x = 0
	} else if x >= g.Width {
		x = g.Width - 1
	}
	if y < 0 {
		y = 0
	} else if y >= g.Height {
		y = g.Height - 1
	}
	return g.Pix[y*g.Width+x]
}

// SampleBilinear samples the image at sub-pixel coordinates (x, y) using bilinear interpolation.
func (g *GrayImage) SampleBilinear(x, y float64) float64 {
	if g == nil || g.Width <= 0 || g.Height <= 0 || len(g.Pix) == 0 {
		return 0
	}
	w, h := g.Width, g.Height
	if x >= 0 && y >= 0 {
		ix := int(x)
		iy := int(y)
		if ix+1 < w && iy+1 < h {
			fx := x - float64(ix)
			fy := y - float64(iy)
			off := iy*w + ix
			pix := g.Pix
			row1 := pix[off+w : off+w+2]
			row0 := pix[off : off+2]
			p00 := float64(row0[0])
			p10 := float64(row0[1])
			p01 := float64(row1[0])
			p11 := float64(row1[1])
			top := p00 + fx*(p10-p00)
			bot := p01 + fx*(p11-p01)
			return top + fy*(bot-top)
		}
	}

	if x < 0 {
		x = 0
	} else if maxX := float64(w - 1); x > maxX {
		x = maxX
	}
	if y < 0 {
		y = 0
	} else if maxY := float64(h - 1); y > maxY {
		y = maxY
	}

	x0 := int(math.Floor(x))
	y0 := int(math.Floor(y))
	x1 := x0 + 1
	if x1 >= w {
		x1 = w - 1
	}
	y1 := y0 + 1
	if y1 >= h {
		y1 = h - 1
	}

	fx := x - float64(x0)
	fy := y - float64(y0)

	p00 := float64(g.Pix[y0*w+x0])
	p10 := float64(g.Pix[y0*w+x1])
	p01 := float64(g.Pix[y1*w+x0])
	p11 := float64(g.Pix[y1*w+x1])

	top := p00 + fx*(p10-p00)
	bot := p01 + fx*(p11-p01)
	return top + fy*(bot-top)
}

// Invert returns a new GrayImage with every pixel value inverted (255 - p).
func (g *GrayImage) Invert() *GrayImage {
	if g == nil || g.Width <= 0 || g.Height <= 0 {
		return NewGrayImage(0, 0)
	}
	out := NewGrayImage(g.Width, g.Height)
	for i, v := range g.Pix {
		out.Pix[i] = 255 - v
	}
	return out
}

// Clone returns an independent copy of the GrayImage.
func (g *GrayImage) Clone() *GrayImage {
	if g == nil || g.Width <= 0 || g.Height <= 0 {
		return NewGrayImage(0, 0)
	}
	out := NewGrayImage(g.Width, g.Height)
	copy(out.Pix, g.Pix)
	return out
}

// Crop extracts the sub-region [x0, x1) x [y0, y1), clamped to image bounds.
func (g *GrayImage) Crop(x0, y0, x1, y1 int) *GrayImage {
	if g == nil || g.Width <= 0 || g.Height <= 0 {
		return NewGrayImage(0, 0)
	}
	if x0 < 0 {
		x0 = 0
	}
	if y0 < 0 {
		y0 = 0
	}
	if x1 > g.Width {
		x1 = g.Width
	}
	if y1 > g.Height {
		y1 = g.Height
	}
	if x1 <= x0 || y1 <= y0 {
		return NewGrayImage(0, 0)
	}
	cw, ch := x1-x0, y1-y0
	out := NewGrayImage(cw, ch)
	for y := 0; y < ch; y++ {
		sOff := (y0+y)*g.Width + x0
		dOff := y * cw
		copy(out.Pix[dOff:dOff+cw], g.Pix[sOff:sOff+cw])
	}
	return out
}

// ReadImageWithOrientation reads an image file (PNG, JPEG, or GIF) and applies
// any JPEG EXIF orientation tag (0x0112, values 1..8) so the returned image is upright.
func ReadImageWithOrientation(path string) (image.Image, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return decodeImageWithOrientation(data)
}

func decodeImageWithOrientation(data []byte) (image.Image, error) {
	img, format, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	if format == "jpeg" {
		if orient := parseJPEGExifOrientation(data); orient >= 2 && orient <= 8 {
			return applyOrientation(img, orient), nil
		}
	}
	return img, nil
}

// parseJPEGExifOrientation inspects JPEG APP1 EXIF IFD0 for tag 0x0112 (Orientation).
// Returns 1 if not found or invalid.
func parseJPEGExifOrientation(data []byte) int {
	if len(data) < 4 || data[0] != 0xFF || data[1] != 0xD8 {
		return 1
	}
	pos := 2
	for pos+4 <= len(data) {
		if data[pos] != 0xFF {
			break
		}
		// Skip fill 0xFF bytes.
		for pos < len(data) && data[pos] == 0xFF {
			pos++
		}
		if pos >= len(data) {
			break
		}
		marker := data[pos]
		pos++

		// Standalone markers without length.
		if marker == 0x00 || marker == 0x01 || (marker >= 0xD0 && marker <= 0xD9) {
			if marker == 0xD9 { // EOI
				break
			}
			continue
		}
		// SOS starts entropy-coded scan; no APP1 metadata follows.
		if marker == 0xDA {
			break
		}
		if pos+2 > len(data) {
			break
		}
		segLen := int(binary.BigEndian.Uint16(data[pos : pos+2]))
		if segLen < 2 || pos+segLen > len(data) {
			break
		}
		if marker == 0xE1 && segLen >= 8 { // APP1
			payload := data[pos+2 : pos+segLen]
			if bytes.HasPrefix(payload, []byte("Exif\x00\x00")) {
				if orient := parseTIFFOrientation(payload[6:]); orient >= 1 && orient <= 8 {
					return orient
				}
			}
		}
		pos += segLen
	}
	return 1
}

func parseTIFFOrientation(tiff []byte) int {
	if len(tiff) < 8 {
		return 1
	}
	var order binary.ByteOrder
	switch {
	case tiff[0] == 'I' && tiff[1] == 'I':
		order = binary.LittleEndian
	case tiff[0] == 'M' && tiff[1] == 'M':
		order = binary.BigEndian
	default:
		return 1
	}
	if order.Uint16(tiff[2:4]) != 0x002A {
		return 1
	}
	ifdOff := int(order.Uint32(tiff[4:8]))
	if ifdOff < 8 || ifdOff+2 > len(tiff) {
		return 1
	}
	numEntries := int(order.Uint16(tiff[ifdOff : ifdOff+2]))
	entryStart := ifdOff + 2
	for i := 0; i < numEntries; i++ {
		off := entryStart + i*12
		if off+12 > len(tiff) {
			break
		}
		tag := order.Uint16(tiff[off : off+2])
		if tag == 0x0112 {
			typ := order.Uint16(tiff[off+2 : off+4])
			count := order.Uint32(tiff[off+4 : off+8])
			if typ == 3 && count >= 1 { // SHORT
				val := int(order.Uint16(tiff[off+8 : off+10]))
				if val >= 1 && val <= 8 {
					return val
				}
			}
			return 1
		}
	}
	return 1
}

func applyOrientation(img image.Image, orient int) image.Image {
	if orient <= 1 || orient > 8 {
		return img
	}
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	if w <= 0 || h <= 0 {
		return img
	}
	dstW, dstH := w, h
	if orient >= 5 {
		dstW, dstH = h, w
	}

	// Fast path for *image.Gray.
	if gray, ok := img.(*image.Gray); ok {
		dst := image.NewGray(image.Rect(0, 0, dstW, dstH))
		for dy := 0; dy < dstH; dy++ {
			for dx := 0; dx < dstW; dx++ {
				sx, sy := mapOrientedCoord(dx, dy, w, h, orient)
				dst.SetGray(dx, dy, gray.GrayAt(b.Min.X+sx, b.Min.Y+sy))
			}
		}
		return dst
	}

	dst := image.NewNRGBA(image.Rect(0, 0, dstW, dstH))
	for dy := 0; dy < dstH; dy++ {
		for dx := 0; dx < dstW; dx++ {
			sx, sy := mapOrientedCoord(dx, dy, w, h, orient)
			dst.Set(dx, dy, img.At(b.Min.X+sx, b.Min.Y+sy))
		}
	}
	return dst
}

// mapOrientedCoord maps upright destination coordinate (dx, dy) to source (sx, sy)
// for EXIF orientation 1..8 given source dimensions (w, h).
func mapOrientedCoord(dx, dy, w, h, orient int) (int, int) {
	switch orient {
	case 2: // Flip horizontal
		return w - 1 - dx, dy
	case 3: // Rotate 180
		return w - 1 - dx, h - 1 - dy
	case 4: // Flip vertical
		return dx, h - 1 - dy
	case 5: // Transpose
		return dy, dx
	case 6: // Rotate 90 CW
		return dy, h - 1 - dx
	case 7: // Transverse
		return w - 1 - dy, h - 1 - dx
	case 8: // Rotate 270 CW (90 CCW)
		return w - 1 - dy, dx
	default:
		return dx, dy
	}
}
