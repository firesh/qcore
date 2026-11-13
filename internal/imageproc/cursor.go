package imageproc

import "math"

// Direction represents a cardinal 2D grid direction in screen coordinates (+X right, +Y down).
type Direction int

const (
	DirRight Direction = iota
	DirDown
	DirLeft
	DirUp
)

// Delta returns the unit (dx, dy) integer step for d.
func (d Direction) Delta() (int, int) {
	switch d & 3 {
	case DirRight:
		return 1, 0
	case DirDown:
		return 0, 1
	case DirLeft:
		return -1, 0
	case DirUp:
		return 0, -1
	default:
		return 0, 0
	}
}

// TurnLeft returns the direction rotated 90 degrees counter-clockwise in screen coordinates.
func (d Direction) TurnLeft() Direction {
	return (d + 3) & 3
}

// TurnRight returns the direction rotated 90 degrees clockwise in screen coordinates.
func (d Direction) TurnRight() Direction {
	return (d + 1) & 3
}

// TurnBack returns the opposite direction.
func (d Direction) TurnBack() Direction {
	return (d + 2) & 3
}

// CursorF is a sub-pixel floating-point ray cursor over a BitMatrix.
type CursorF struct {
	Img *BitMatrix
	X   float64
	Y   float64
	DX  float64
	DY  float64
}

// NewCursorF constructs a CursorF at (x, y) with normalized unit direction (dx, dy).
func NewCursorF(bm *BitMatrix, x, y, dx, dy float64) CursorF {
	length := math.Hypot(dx, dy)
	if length > 0 {
		dx /= length
		dy /= length
	} else {
		dx, dy = 0, 0
	}
	return CursorF{
		Img: bm,
		X:   x,
		Y:   y,
		DX:  dx,
		DY:  dy,
	}
}

func (c CursorF) pixelCoord() (int, int) {
	return int(math.Floor(c.X)), int(math.Floor(c.Y))
}

// IsIn reports whether the cursor's current integer pixel is inside the BitMatrix bounds.
func (c CursorF) IsIn() bool {
	if c.Img == nil {
		return false
	}
	ix, iy := c.pixelCoord()
	return c.Img.IsIn(ix, iy)
}

// IsBlack reports whether the pixel under the cursor is black.
func (c CursorF) IsBlack() bool {
	if c.Img == nil {
		return false
	}
	ix, iy := c.pixelCoord()
	return c.Img.Get(ix, iy)
}

// Step advances the cursor by n units along (DX, DY) and returns whether it remains in bounds.
func (c *CursorF) Step(n float64) bool {
	c.X += n * c.DX
	c.Y += n * c.DY
	return c.IsIn()
}

// TurnBack reverses the cursor direction.
func (c *CursorF) TurnBack() {
	c.DX = -c.DX
	c.DY = -c.DY
}

// StepToEdge steps forward one unit at a time until nth black/white transitions are crossed.
// If backup is true, the cursor steps back by 1 unit after reaching the nth edge so it sits
// on the last pixel before the transition. Returns the number of steps taken, or 0 if out of
// bounds or maxSteps is exceeded.
func (c *CursorF) StepToEdge(nth int, maxSteps int, backup bool) int {
	if !c.IsIn() || nth <= 0 || maxSteps <= 0 || (c.DX == 0 && c.DY == 0) {
		return 0
	}
	lastColor := c.IsBlack()
	steps := 0
	for steps < maxSteps {
		if !c.Step(1) {
			return 0
		}
		steps++
		curColor := c.IsBlack()
		if curColor != lastColor {
			nth--
			if nth == 0 {
				if backup {
					c.Step(-1)
				}
				return steps
			}
			lastColor = curColor
		}
	}
	return 0
}

// ReadSymmetricPattern5 reads a 5-bar B:W:B:W:B run-length pattern [w0, w1, w2, w3, w4]
// symmetrically in both the forward and backward ray directions starting from a black pixel
// inside the center bar (w2). Returns false if the start pixel is not black, an edge is missing
// before the image boundary, or the total width exceeds maxTotal (when maxTotal > 0).
func (c *CursorF) ReadSymmetricPattern5(maxTotal int) ([5]int, bool) {
	var zero [5]int
	if !c.IsIn() || !c.IsBlack() || (c.DX == 0 && c.DY == 0) {
		return zero, false
	}
	limit := maxTotal
	if limit <= 0 {
		limit = c.Img.Width + c.Img.Height
	}

	fwd := *c
	f2 := fwd.StepToEdge(1, limit, false)
	if f2 == 0 {
		return zero, false
	}
	f3 := fwd.StepToEdge(1, limit-f2, false)
	if f3 == 0 {
		return zero, false
	}
	f4 := fwd.StepToEdge(1, limit-f2-f3, false)
	if f4 == 0 {
		return zero, false
	}

	bwd := *c
	bwd.TurnBack()
	b2 := bwd.StepToEdge(1, limit, false)
	if b2 == 0 {
		return zero, false
	}
	b1 := bwd.StepToEdge(1, limit-b2, false)
	if b1 == 0 {
		return zero, false
	}
	b0 := bwd.StepToEdge(1, limit-b2-b1, false)
	if b0 == 0 {
		return zero, false
	}

	w2 := f2 + b2 - 1
	total := b0 + b1 + w2 + f3 + f4
	if maxTotal > 0 && total > maxTotal {
		return zero, false
	}
	return [5]int{b0, b1, w2, f3, f4}, true
}

// CursorI is an integer grid cursor for tracing connected component edges on a BitMatrix.
type CursorI struct {
	Img *BitMatrix
	X   int
	Y   int
	DX  int
	DY  int
}

// NewCursorI creates an integer cursor at (x, y) facing Direction dir.
func NewCursorI(bm *BitMatrix, x, y int, dir Direction) CursorI {
	dx, dy := dir.Delta()
	return CursorI{
		Img: bm,
		X:   x,
		Y:   y,
		DX:  dx,
		DY:  dy,
	}
}

// IsIn reports whether the cursor is inside the BitMatrix bounds.
func (c CursorI) IsIn() bool {
	return c.Img != nil && c.Img.IsIn(c.X, c.Y)
}

// IsBlack reports whether the current pixel is black.
func (c CursorI) IsBlack() bool {
	return c.Img != nil && c.Img.Get(c.X, c.Y)
}

// Step moves the cursor n pixels along (DX, DY) and returns whether it remains in bounds.
func (c *CursorI) Step(n int) bool {
	c.X += n * c.DX
	c.Y += n * c.DY
	return c.IsIn()
}

// TurnLeft rotates the cursor 90 degrees counter-clockwise in screen coordinates (+X right, +Y down).
func (c *CursorI) TurnLeft() {
	c.DX, c.DY = c.DY, -c.DX
}

// TurnRight rotates the cursor 90 degrees clockwise in screen coordinates (+X right, +Y down).
func (c *CursorI) TurnRight() {
	c.DX, c.DY = -c.DY, c.DX
}

// TurnBack reverses the cursor direction.
func (c *CursorI) TurnBack() {
	c.DX, c.DY = -c.DX, -c.DY
}

func (c CursorI) edgeAt(ox, oy int) bool {
	if !c.IsIn() {
		return false
	}
	nx, ny := c.X+ox, c.Y+oy
	if !c.Img.IsIn(nx, ny) {
		return false
	}
	return c.Img.Get(c.X, c.Y) != c.Img.Get(nx, ny)
}

// EdgeAtFront reports whether there is a black/white edge one pixel ahead.
func (c CursorI) EdgeAtFront() bool {
	return c.edgeAt(c.DX, c.DY)
}

// EdgeAtBack reports whether there is a black/white edge one pixel behind.
func (c CursorI) EdgeAtBack() bool {
	return c.edgeAt(-c.DX, -c.DY)
}

// EdgeAtLeft reports whether there is a black/white edge one pixel to the left.
func (c CursorI) EdgeAtLeft() bool {
	return c.edgeAt(c.DY, -c.DX)
}

// EdgeAtRight reports whether there is a black/white edge one pixel to the right.
func (c CursorI) EdgeAtRight() bool {
	return c.edgeAt(-c.DY, c.DX)
}

func (c *CursorI) turn(dir int) {
	if dir < 0 {
		c.TurnLeft()
	} else {
		c.TurnRight()
	}
}

func (c CursorI) edgeAtSide(dir int) bool {
	if dir < 0 {
		return c.EdgeAtLeft()
	}
	return c.EdgeAtRight()
}

func (c CursorI) edgeAtFrontDiag(dir int) bool {
	if dir < 0 {
		return c.edgeAt(c.DX+c.DY, c.DY-c.DX)
	}
	return c.edgeAt(c.DX-c.DY, c.DY+c.DX)
}

// StepAlongEdge advances the cursor by one pixel along the boundary on side dir
// (dir < 0 keeps the contrasting edge on the left; dir > 0 keeps it on the right).
// When straight is true, the cursor avoids turning toward side dir if the contrasting
// pixel lies diagonally ahead. Returns false if the cursor cannot step safely inside bounds.
func (c *CursorI) StepAlongEdge(dir int, straight bool) bool {
	if !c.IsIn() || dir == 0 || (c.DX == 0 && c.DY == 0) {
		return false
	}

	if !c.edgeAtSide(dir) {
		if !straight || c.EdgeAtFront() || !c.edgeAtFrontDiag(dir) {
			c.turn(dir)
		}
	} else if c.EdgeAtFront() {
		c.turn(-dir)
		if c.EdgeAtFront() {
			c.turn(-dir)
		}
		if c.EdgeAtFront() {
			return false
		}
	}

	nx, ny := c.X+c.DX, c.Y+c.DY
	if !c.Img.IsIn(nx, ny) {
		return false
	}
	if c.Img.Get(nx, ny) != c.Img.Get(c.X, c.Y) {
		return false
	}
	c.X, c.Y = nx, ny
	return true
}
