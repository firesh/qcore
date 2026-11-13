package decoder

const gfPrimitive = 0x11D // x^8 + x^4 + x^3 + x^2 + 1

var (
	expTable [512]uint8
	logTable [256]uint8
)

func init() {
	x := 1
	for i := 0; i < 255; i++ {
		expTable[i] = uint8(x)
		expTable[i+255] = uint8(x)
		logTable[x] = uint8(i)
		x <<= 1
		if x&0x100 != 0 {
			x ^= gfPrimitive
		}
	}
	expTable[510] = expTable[0]
	expTable[511] = expTable[1]
}

func gfAdd(a, b uint8) uint8 {
	return a ^ b
}

func gfMul(a, b uint8) uint8 {
	if a == 0 || b == 0 {
		return 0
	}
	return expTable[int(logTable[a])+int(logTable[b])]
}

func gfDiv(a, b uint8) uint8 {
	if b == 0 {
		panic("gf256: division by zero")
	}
	if a == 0 {
		return 0
	}
	return expTable[int(logTable[a])-int(logTable[b])+255]
}

func gfInv(a uint8) uint8 {
	if a == 0 {
		panic("gf256: inverse of zero")
	}
	return expTable[255-int(logTable[a])]
}

func gfPow(a uint8, n int) uint8 {
	if n == 0 {
		return 1
	}
	if a == 0 {
		return 0
	}
	// Reduce n modulo 255 first to avoid overflow for large |n|.
	m := n % 255
	if m < 0 {
		m += 255
	}
	return expTable[(int(logTable[a])*m)%255]
}

// Polynomial helpers use ascending degree representation: p[i] is the
// coefficient of x^i.

func polyAdd(a, b []uint8) []uint8 {
	n := len(a)
	if len(b) > n {
		n = len(b)
	}
	out := make([]uint8, n)
	copy(out, a)
	for i, v := range b {
		out[i] ^= v
	}
	return out
}

func polyScale(p []uint8, s uint8) []uint8 {
	if s == 0 {
		return make([]uint8, len(p))
	}
	out := make([]uint8, len(p))
	for i, v := range p {
		out[i] = gfMul(v, s)
	}
	return out
}

func polyMul(a, b []uint8) []uint8 {
	if len(a) == 0 || len(b) == 0 {
		return nil
	}
	out := make([]uint8, len(a)+len(b)-1)
	for i, av := range a {
		if av == 0 {
			continue
		}
		for j, bv := range b {
			if bv != 0 {
				out[i+j] ^= gfMul(av, bv)
			}
		}
	}
	return out
}

func polyEval(p []uint8, x uint8) uint8 {
	if len(p) == 0 {
		return 0
	}
	if x == 0 {
		return p[0]
	}
	var res uint8
	for i := len(p) - 1; i >= 0; i-- {
		res = gfAdd(gfMul(res, x), p[i])
	}
	return res
}
