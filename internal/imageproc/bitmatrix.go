package imageproc

// BitMatrix is a dense binary 2D grid stored as one byte per pixel (1 = black, 0 = white)
// for fast, branchless row scanning.
type BitMatrix struct {
	Bits   []uint8
	Width  int
	Height int
}

// NewBitMatrix allocates a BitMatrix of size w x h initialized to all white (0).
func NewBitMatrix(w, h int) *BitMatrix {
	if w <= 0 || h <= 0 {
		return &BitMatrix{Width: 0, Height: 0}
	}
	return &BitMatrix{
		Bits:   make([]uint8, w*h),
		Width:  w,
		Height: h,
	}
}

// IsIn reports whether (x, y) is within the matrix bounds.
func (bm *BitMatrix) IsIn(x, y int) bool {
	return bm != nil && uint(x) < uint(bm.Width) && uint(y) < uint(bm.Height)
}

// Get returns true if (x, y) is black (1), or false if white (0) or out of bounds.
func (bm *BitMatrix) Get(x, y int) bool {
	if bm == nil || uint(x) >= uint(bm.Width) || uint(y) >= uint(bm.Height) {
		return false
	}
	return bm.Bits[y*bm.Width+x] != 0
}

// Set marks (x, y) as black (1) if within bounds.
func (bm *BitMatrix) Set(x, y int) {
	if bm != nil && uint(x) < uint(bm.Width) && uint(y) < uint(bm.Height) {
		bm.Bits[y*bm.Width+x] = 1
	}
}

// Clear marks (x, y) as white (0) if within bounds.
func (bm *BitMatrix) Clear(x, y int) {
	if bm != nil && uint(x) < uint(bm.Width) && uint(y) < uint(bm.Height) {
		bm.Bits[y*bm.Width+x] = 0
	}
}

// SetVal sets (x, y) to black (1) when black is true, or white (0) otherwise.
func (bm *BitMatrix) SetVal(x, y int, black bool) {
	if bm == nil || uint(x) >= uint(bm.Width) || uint(y) >= uint(bm.Height) {
		return
	}
	if black {
		bm.Bits[y*bm.Width+x] = 1
	} else {
		bm.Bits[y*bm.Width+x] = 0
	}
}

// Invert flips every bit in-place (0 <-> 1).
func (bm *BitMatrix) Invert() {
	if bm == nil {
		return
	}
	for i, v := range bm.Bits {
		bm.Bits[i] = v ^ 1
	}
}

// Clone returns an independent copy of the BitMatrix.
func (bm *BitMatrix) Clone() *BitMatrix {
	if bm == nil || bm.Width <= 0 || bm.Height <= 0 {
		return NewBitMatrix(0, 0)
	}
	out := NewBitMatrix(bm.Width, bm.Height)
	copy(out.Bits, bm.Bits)
	return out
}

// ClearRect sets all pixels in [x0, x1) x [y0, y1) to white (0), clamped to bounds.
func (bm *BitMatrix) ClearRect(x0, y0, x1, y1 int) {
	if bm == nil || bm.Width <= 0 || bm.Height <= 0 {
		return
	}
	if x0 < 0 {
		x0 = 0
	}
	if y0 < 0 {
		y0 = 0
	}
	if x1 > bm.Width {
		x1 = bm.Width
	}
	if y1 > bm.Height {
		y1 = bm.Height
	}
	if x1 <= x0 || y1 <= y0 {
		return
	}
	for y := y0; y < y1; y++ {
		row := bm.Bits[y*bm.Width+x0 : y*bm.Width+x1]
		clear(row)
	}
}

// RowSlice returns the underlying byte slice for row y, or nil if y is out of bounds.
func (bm *BitMatrix) RowSlice(y int) []uint8 {
	if bm == nil || y < 0 || y >= bm.Height || bm.Width <= 0 {
		return nil
	}
	off := y * bm.Width
	return bm.Bits[off : off+bm.Width]
}
