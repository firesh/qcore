package decoder

import (
	"fmt"
	"math/bits"
	"sync"
)

const versionGolayPoly uint32 = 0x1F25

// ECBlockGroup describes a group of identical Reed-Solomon blocks in a QR symbol.
type ECBlockGroup struct {
	Count         int
	DataCodewords int
}

// ECBlockSpec describes the Reed-Solomon block structure for a (version, ecLevel) pair.
type ECBlockSpec struct {
	TotalCodewords int
	ECPerBlock     int
	Group1         ECBlockGroup
	Group2         ECBlockGroup
}

// TotalDataCodewords returns the sum of data codewords across both groups.
func (s ECBlockSpec) TotalDataCodewords() int {
	return s.Group1.Count*s.Group1.DataCodewords + s.Group2.Count*s.Group2.DataCodewords
}

// TotalBlocks returns the total number of RS blocks across both groups.
func (s ECBlockSpec) TotalBlocks() int {
	return s.Group1.Count + s.Group2.Count
}

var alignmentCentersTable = [41][]int{
	1:  nil,
	2:  {6, 18},
	3:  {6, 22},
	4:  {6, 26},
	5:  {6, 30},
	6:  {6, 34},
	7:  {6, 22, 38},
	8:  {6, 24, 42},
	9:  {6, 26, 46},
	10: {6, 28, 50},
	11: {6, 30, 54},
	12: {6, 32, 58},
	13: {6, 34, 62},
	14: {6, 26, 46, 66},
	15: {6, 26, 48, 70},
	16: {6, 26, 50, 74},
	17: {6, 30, 54, 78},
	18: {6, 30, 56, 82},
	19: {6, 30, 58, 86},
	20: {6, 34, 62, 90},
	21: {6, 28, 50, 72, 94},
	22: {6, 26, 50, 74, 98},
	23: {6, 30, 54, 78, 102},
	24: {6, 28, 54, 80, 106},
	25: {6, 32, 58, 84, 110},
	26: {6, 30, 58, 86, 114},
	27: {6, 34, 62, 90, 118},
	28: {6, 26, 50, 74, 98, 122},
	29: {6, 30, 54, 78, 102, 126},
	30: {6, 26, 52, 78, 104, 130},
	31: {6, 30, 56, 82, 108, 134},
	32: {6, 34, 60, 86, 112, 138},
	33: {6, 30, 58, 86, 114, 142},
	34: {6, 34, 62, 90, 118, 146},
	35: {6, 30, 54, 78, 102, 126, 150},
	36: {6, 24, 50, 76, 102, 128, 154},
	37: {6, 28, 54, 80, 106, 132, 158},
	38: {6, 32, 58, 84, 110, 136, 162},
	39: {6, 26, 54, 82, 110, 138, 166},
	40: {6, 30, 58, 86, 114, 142, 170},
}

// ECBlocks maps [version][ecLevel] (for version 1..40 and ECM/ECL/ECH/ECQ) to its ISO/IEC 18004 RS block layout.
var ECBlocks = [41][4]ECBlockSpec{
	1: {
		ECL: {26, 7, ECBlockGroup{1, 19}, ECBlockGroup{0, 0}},
		ECM: {26, 10, ECBlockGroup{1, 16}, ECBlockGroup{0, 0}},
		ECQ: {26, 13, ECBlockGroup{1, 13}, ECBlockGroup{0, 0}},
		ECH: {26, 17, ECBlockGroup{1, 9}, ECBlockGroup{0, 0}},
	},
	2: {
		ECL: {44, 10, ECBlockGroup{1, 34}, ECBlockGroup{0, 0}},
		ECM: {44, 16, ECBlockGroup{1, 28}, ECBlockGroup{0, 0}},
		ECQ: {44, 22, ECBlockGroup{1, 22}, ECBlockGroup{0, 0}},
		ECH: {44, 28, ECBlockGroup{1, 16}, ECBlockGroup{0, 0}},
	},
	3: {
		ECL: {70, 15, ECBlockGroup{1, 55}, ECBlockGroup{0, 0}},
		ECM: {70, 26, ECBlockGroup{1, 44}, ECBlockGroup{0, 0}},
		ECQ: {70, 18, ECBlockGroup{2, 17}, ECBlockGroup{0, 0}},
		ECH: {70, 22, ECBlockGroup{2, 13}, ECBlockGroup{0, 0}},
	},
	4: {
		ECL: {100, 20, ECBlockGroup{1, 80}, ECBlockGroup{0, 0}},
		ECM: {100, 18, ECBlockGroup{2, 32}, ECBlockGroup{0, 0}},
		ECQ: {100, 26, ECBlockGroup{2, 24}, ECBlockGroup{0, 0}},
		ECH: {100, 16, ECBlockGroup{4, 9}, ECBlockGroup{0, 0}},
	},
	5: {
		ECL: {134, 26, ECBlockGroup{1, 108}, ECBlockGroup{0, 0}},
		ECM: {134, 24, ECBlockGroup{2, 43}, ECBlockGroup{0, 0}},
		ECQ: {134, 18, ECBlockGroup{2, 15}, ECBlockGroup{2, 16}},
		ECH: {134, 22, ECBlockGroup{2, 11}, ECBlockGroup{2, 12}},
	},
	6: {
		ECL: {172, 18, ECBlockGroup{2, 68}, ECBlockGroup{0, 0}},
		ECM: {172, 16, ECBlockGroup{4, 27}, ECBlockGroup{0, 0}},
		ECQ: {172, 24, ECBlockGroup{4, 19}, ECBlockGroup{0, 0}},
		ECH: {172, 28, ECBlockGroup{4, 15}, ECBlockGroup{0, 0}},
	},
	7: {
		ECL: {196, 20, ECBlockGroup{2, 78}, ECBlockGroup{0, 0}},
		ECM: {196, 18, ECBlockGroup{4, 31}, ECBlockGroup{0, 0}},
		ECQ: {196, 18, ECBlockGroup{2, 14}, ECBlockGroup{4, 15}},
		ECH: {196, 26, ECBlockGroup{4, 13}, ECBlockGroup{1, 14}},
	},
	8: {
		ECL: {242, 24, ECBlockGroup{2, 97}, ECBlockGroup{0, 0}},
		ECM: {242, 22, ECBlockGroup{2, 38}, ECBlockGroup{2, 39}},
		ECQ: {242, 22, ECBlockGroup{4, 18}, ECBlockGroup{2, 19}},
		ECH: {242, 26, ECBlockGroup{4, 14}, ECBlockGroup{2, 15}},
	},
	9: {
		ECL: {292, 30, ECBlockGroup{2, 116}, ECBlockGroup{0, 0}},
		ECM: {292, 22, ECBlockGroup{3, 36}, ECBlockGroup{2, 37}},
		ECQ: {292, 20, ECBlockGroup{4, 16}, ECBlockGroup{4, 17}},
		ECH: {292, 24, ECBlockGroup{4, 12}, ECBlockGroup{4, 13}},
	},
	10: {
		ECL: {346, 18, ECBlockGroup{2, 68}, ECBlockGroup{2, 69}},
		ECM: {346, 26, ECBlockGroup{4, 43}, ECBlockGroup{1, 44}},
		ECQ: {346, 24, ECBlockGroup{6, 19}, ECBlockGroup{2, 20}},
		ECH: {346, 28, ECBlockGroup{6, 15}, ECBlockGroup{2, 16}},
	},
	11: {
		ECL: {404, 20, ECBlockGroup{4, 81}, ECBlockGroup{0, 0}},
		ECM: {404, 30, ECBlockGroup{1, 50}, ECBlockGroup{4, 51}},
		ECQ: {404, 28, ECBlockGroup{4, 22}, ECBlockGroup{4, 23}},
		ECH: {404, 24, ECBlockGroup{3, 12}, ECBlockGroup{8, 13}},
	},
	12: {
		ECL: {466, 24, ECBlockGroup{2, 92}, ECBlockGroup{2, 93}},
		ECM: {466, 22, ECBlockGroup{6, 36}, ECBlockGroup{2, 37}},
		ECQ: {466, 26, ECBlockGroup{4, 20}, ECBlockGroup{6, 21}},
		ECH: {466, 28, ECBlockGroup{7, 14}, ECBlockGroup{4, 15}},
	},
	13: {
		ECL: {532, 26, ECBlockGroup{4, 107}, ECBlockGroup{0, 0}},
		ECM: {532, 22, ECBlockGroup{8, 37}, ECBlockGroup{1, 38}},
		ECQ: {532, 24, ECBlockGroup{8, 20}, ECBlockGroup{4, 21}},
		ECH: {532, 22, ECBlockGroup{12, 11}, ECBlockGroup{4, 12}},
	},
	14: {
		ECL: {581, 30, ECBlockGroup{3, 115}, ECBlockGroup{1, 116}},
		ECM: {581, 24, ECBlockGroup{4, 40}, ECBlockGroup{5, 41}},
		ECQ: {581, 20, ECBlockGroup{11, 16}, ECBlockGroup{5, 17}},
		ECH: {581, 24, ECBlockGroup{11, 12}, ECBlockGroup{5, 13}},
	},
	15: {
		ECL: {655, 22, ECBlockGroup{5, 87}, ECBlockGroup{1, 88}},
		ECM: {655, 24, ECBlockGroup{5, 41}, ECBlockGroup{5, 42}},
		ECQ: {655, 30, ECBlockGroup{5, 24}, ECBlockGroup{7, 25}},
		ECH: {655, 24, ECBlockGroup{11, 12}, ECBlockGroup{7, 13}},
	},
	16: {
		ECL: {733, 24, ECBlockGroup{5, 98}, ECBlockGroup{1, 99}},
		ECM: {733, 28, ECBlockGroup{7, 45}, ECBlockGroup{3, 46}},
		ECQ: {733, 24, ECBlockGroup{15, 19}, ECBlockGroup{2, 20}},
		ECH: {733, 30, ECBlockGroup{3, 15}, ECBlockGroup{13, 16}},
	},
	17: {
		ECL: {815, 28, ECBlockGroup{1, 107}, ECBlockGroup{5, 108}},
		ECM: {815, 28, ECBlockGroup{10, 46}, ECBlockGroup{1, 47}},
		ECQ: {815, 28, ECBlockGroup{1, 22}, ECBlockGroup{15, 23}},
		ECH: {815, 28, ECBlockGroup{2, 14}, ECBlockGroup{17, 15}},
	},
	18: {
		ECL: {901, 30, ECBlockGroup{5, 120}, ECBlockGroup{1, 121}},
		ECM: {901, 26, ECBlockGroup{9, 43}, ECBlockGroup{4, 44}},
		ECQ: {901, 28, ECBlockGroup{17, 22}, ECBlockGroup{1, 23}},
		ECH: {901, 28, ECBlockGroup{2, 14}, ECBlockGroup{19, 15}},
	},
	19: {
		ECL: {991, 28, ECBlockGroup{3, 113}, ECBlockGroup{4, 114}},
		ECM: {991, 26, ECBlockGroup{3, 44}, ECBlockGroup{11, 45}},
		ECQ: {991, 26, ECBlockGroup{17, 21}, ECBlockGroup{4, 22}},
		ECH: {991, 26, ECBlockGroup{9, 13}, ECBlockGroup{16, 14}},
	},
	20: {
		ECL: {1085, 28, ECBlockGroup{3, 107}, ECBlockGroup{5, 108}},
		ECM: {1085, 26, ECBlockGroup{3, 41}, ECBlockGroup{13, 42}},
		ECQ: {1085, 30, ECBlockGroup{15, 24}, ECBlockGroup{5, 25}},
		ECH: {1085, 28, ECBlockGroup{15, 15}, ECBlockGroup{10, 16}},
	},
	21: {
		ECL: {1156, 28, ECBlockGroup{4, 116}, ECBlockGroup{4, 117}},
		ECM: {1156, 26, ECBlockGroup{17, 42}, ECBlockGroup{0, 0}},
		ECQ: {1156, 28, ECBlockGroup{17, 22}, ECBlockGroup{6, 23}},
		ECH: {1156, 30, ECBlockGroup{19, 16}, ECBlockGroup{6, 17}},
	},
	22: {
		ECL: {1258, 28, ECBlockGroup{2, 111}, ECBlockGroup{7, 112}},
		ECM: {1258, 28, ECBlockGroup{17, 46}, ECBlockGroup{0, 0}},
		ECQ: {1258, 30, ECBlockGroup{7, 24}, ECBlockGroup{16, 25}},
		ECH: {1258, 24, ECBlockGroup{34, 13}, ECBlockGroup{0, 0}},
	},
	23: {
		ECL: {1364, 30, ECBlockGroup{4, 121}, ECBlockGroup{5, 122}},
		ECM: {1364, 28, ECBlockGroup{4, 47}, ECBlockGroup{14, 48}},
		ECQ: {1364, 30, ECBlockGroup{11, 24}, ECBlockGroup{14, 25}},
		ECH: {1364, 30, ECBlockGroup{16, 15}, ECBlockGroup{14, 16}},
	},
	24: {
		ECL: {1474, 30, ECBlockGroup{6, 117}, ECBlockGroup{4, 118}},
		ECM: {1474, 28, ECBlockGroup{6, 45}, ECBlockGroup{14, 46}},
		ECQ: {1474, 30, ECBlockGroup{11, 24}, ECBlockGroup{16, 25}},
		ECH: {1474, 30, ECBlockGroup{30, 16}, ECBlockGroup{2, 17}},
	},
	25: {
		ECL: {1588, 26, ECBlockGroup{8, 106}, ECBlockGroup{4, 107}},
		ECM: {1588, 28, ECBlockGroup{8, 47}, ECBlockGroup{13, 48}},
		ECQ: {1588, 30, ECBlockGroup{7, 24}, ECBlockGroup{22, 25}},
		ECH: {1588, 30, ECBlockGroup{22, 15}, ECBlockGroup{13, 16}},
	},
	26: {
		ECL: {1706, 28, ECBlockGroup{10, 114}, ECBlockGroup{2, 115}},
		ECM: {1706, 28, ECBlockGroup{19, 46}, ECBlockGroup{4, 47}},
		ECQ: {1706, 28, ECBlockGroup{28, 22}, ECBlockGroup{6, 23}},
		ECH: {1706, 30, ECBlockGroup{33, 16}, ECBlockGroup{4, 17}},
	},
	27: {
		ECL: {1828, 30, ECBlockGroup{8, 122}, ECBlockGroup{4, 123}},
		ECM: {1828, 28, ECBlockGroup{22, 45}, ECBlockGroup{3, 46}},
		ECQ: {1828, 30, ECBlockGroup{8, 23}, ECBlockGroup{26, 24}},
		ECH: {1828, 30, ECBlockGroup{12, 15}, ECBlockGroup{28, 16}},
	},
	28: {
		ECL: {1921, 30, ECBlockGroup{3, 117}, ECBlockGroup{10, 118}},
		ECM: {1921, 28, ECBlockGroup{3, 45}, ECBlockGroup{23, 46}},
		ECQ: {1921, 30, ECBlockGroup{4, 24}, ECBlockGroup{31, 25}},
		ECH: {1921, 30, ECBlockGroup{11, 15}, ECBlockGroup{31, 16}},
	},
	29: {
		ECL: {2051, 30, ECBlockGroup{7, 116}, ECBlockGroup{7, 117}},
		ECM: {2051, 28, ECBlockGroup{21, 45}, ECBlockGroup{7, 46}},
		ECQ: {2051, 30, ECBlockGroup{1, 23}, ECBlockGroup{37, 24}},
		ECH: {2051, 30, ECBlockGroup{19, 15}, ECBlockGroup{26, 16}},
	},
	30: {
		ECL: {2185, 30, ECBlockGroup{5, 115}, ECBlockGroup{10, 116}},
		ECM: {2185, 28, ECBlockGroup{19, 47}, ECBlockGroup{10, 48}},
		ECQ: {2185, 30, ECBlockGroup{15, 24}, ECBlockGroup{25, 25}},
		ECH: {2185, 30, ECBlockGroup{23, 15}, ECBlockGroup{25, 16}},
	},
	31: {
		ECL: {2323, 30, ECBlockGroup{13, 115}, ECBlockGroup{3, 116}},
		ECM: {2323, 28, ECBlockGroup{2, 46}, ECBlockGroup{29, 47}},
		ECQ: {2323, 30, ECBlockGroup{42, 24}, ECBlockGroup{1, 25}},
		ECH: {2323, 30, ECBlockGroup{23, 15}, ECBlockGroup{28, 16}},
	},
	32: {
		ECL: {2465, 30, ECBlockGroup{17, 115}, ECBlockGroup{0, 0}},
		ECM: {2465, 28, ECBlockGroup{10, 46}, ECBlockGroup{23, 47}},
		ECQ: {2465, 30, ECBlockGroup{10, 24}, ECBlockGroup{35, 25}},
		ECH: {2465, 30, ECBlockGroup{19, 15}, ECBlockGroup{35, 16}},
	},
	33: {
		ECL: {2611, 30, ECBlockGroup{17, 115}, ECBlockGroup{1, 116}},
		ECM: {2611, 28, ECBlockGroup{14, 46}, ECBlockGroup{21, 47}},
		ECQ: {2611, 30, ECBlockGroup{29, 24}, ECBlockGroup{19, 25}},
		ECH: {2611, 30, ECBlockGroup{11, 15}, ECBlockGroup{46, 16}},
	},
	34: {
		ECL: {2761, 30, ECBlockGroup{13, 115}, ECBlockGroup{6, 116}},
		ECM: {2761, 28, ECBlockGroup{14, 46}, ECBlockGroup{23, 47}},
		ECQ: {2761, 30, ECBlockGroup{44, 24}, ECBlockGroup{7, 25}},
		ECH: {2761, 30, ECBlockGroup{59, 16}, ECBlockGroup{1, 17}},
	},
	35: {
		ECL: {2876, 30, ECBlockGroup{12, 121}, ECBlockGroup{7, 122}},
		ECM: {2876, 28, ECBlockGroup{12, 47}, ECBlockGroup{26, 48}},
		ECQ: {2876, 30, ECBlockGroup{39, 24}, ECBlockGroup{14, 25}},
		ECH: {2876, 30, ECBlockGroup{22, 15}, ECBlockGroup{41, 16}},
	},
	36: {
		ECL: {3034, 30, ECBlockGroup{6, 121}, ECBlockGroup{14, 122}},
		ECM: {3034, 28, ECBlockGroup{6, 47}, ECBlockGroup{34, 48}},
		ECQ: {3034, 30, ECBlockGroup{46, 24}, ECBlockGroup{10, 25}},
		ECH: {3034, 30, ECBlockGroup{2, 15}, ECBlockGroup{64, 16}},
	},
	37: {
		ECL: {3196, 30, ECBlockGroup{17, 122}, ECBlockGroup{4, 123}},
		ECM: {3196, 28, ECBlockGroup{29, 46}, ECBlockGroup{14, 47}},
		ECQ: {3196, 30, ECBlockGroup{49, 24}, ECBlockGroup{10, 25}},
		ECH: {3196, 30, ECBlockGroup{24, 15}, ECBlockGroup{46, 16}},
	},
	38: {
		ECL: {3362, 30, ECBlockGroup{4, 122}, ECBlockGroup{18, 123}},
		ECM: {3362, 28, ECBlockGroup{13, 46}, ECBlockGroup{32, 47}},
		ECQ: {3362, 30, ECBlockGroup{48, 24}, ECBlockGroup{14, 25}},
		ECH: {3362, 30, ECBlockGroup{42, 15}, ECBlockGroup{32, 16}},
	},
	39: {
		ECL: {3532, 30, ECBlockGroup{20, 117}, ECBlockGroup{4, 118}},
		ECM: {3532, 28, ECBlockGroup{40, 47}, ECBlockGroup{7, 48}},
		ECQ: {3532, 30, ECBlockGroup{43, 24}, ECBlockGroup{22, 25}},
		ECH: {3532, 30, ECBlockGroup{10, 15}, ECBlockGroup{67, 16}},
	},
	40: {
		ECL: {3706, 30, ECBlockGroup{19, 118}, ECBlockGroup{6, 119}},
		ECM: {3706, 28, ECBlockGroup{18, 47}, ECBlockGroup{31, 48}},
		ECQ: {3706, 30, ECBlockGroup{34, 24}, ECBlockGroup{34, 25}},
		ECH: {3706, 30, ECBlockGroup{20, 15}, ECBlockGroup{61, 16}},
	},
}

var versionGolayCodes [41]uint32

func encodeVersionGolay(version int) uint32 {
	v := uint32(version) << 12
	rem := v
	for i := 5; i >= 0; i-- {
		if rem&(1<<(i+12)) != 0 {
			rem ^= versionGolayPoly << i
		}
	}
	return v | rem
}

func init() {
	for v := 7; v <= 40; v++ {
		versionGolayCodes[v] = encodeVersionGolay(v)
	}
}

// DimensionForVersion returns the module width/height (17 + 4*version) for version 1..40.
func DimensionForVersion(version int) int {
	if version < 1 || version > 40 {
		return 0
	}
	return 17 + 4*version
}

// AlignmentPatternCenters returns a copy of the alignment pattern center coordinates for version 1..40.
func AlignmentPatternCenters(version int) []int {
	if version < 1 || version > 40 {
		return nil
	}
	src := alignmentCentersTable[version]
	if len(src) == 0 {
		return nil
	}
	out := make([]int, len(src))
	copy(out, src)
	return out
}

// DecodeVersionInformation decodes the two 18-bit Golay(18, 6) version info blocks
// for Versions 7..40, matching within Hamming distance <= 3.
func DecodeVersionInformation(bits1, bits2 uint32) (version int, ok bool) {
	b1 := bits1 & 0x3FFFF
	b2 := bits2 & 0x3FFFF

	bestDist := 19
	bestVer := 0
	for v := 7; v <= 40; v++ {
		code := versionGolayCodes[v]
		if b1 == code || b2 == code {
			return v, true
		}
		d := bits.OnesCount32(b1 ^ code)
		if d2 := bits.OnesCount32(b2 ^ code); d2 < d {
			d = d2
		}
		if d < bestDist {
			bestDist = d
			bestVer = v
		}
	}

	if bestDist <= 3 {
		return bestVer, true
	}
	return 0, false
}

type versionLayout struct {
	fnMask    [][]bool
	bitCoords [][8][2]uint8
	maskXor   [8][]uint8
}

var (
	versionLayoutOnce  [41]sync.Once
	versionLayoutCache [41]*versionLayout
)

func getVersionLayout(version int) *versionLayout {
	if version < 1 || version > 40 {
		return nil
	}
	versionLayoutOnce[version].Do(func() {
		dim := 17 + 4*version
		fnMask := buildRawFunctionMask(version, dim)
		totalCodewords := ECBlocks[version][ECL].TotalCodewords
		bitCoords := make([][8][2]uint8, 0, totalCodewords)

		var cur [8][2]uint8
		bitCount := 0
		readingUp := true

	outer:
		for col := dim - 1; col > 0; col -= 2 {
			if col == 6 {
				col--
			}
			for i := 0; i < dim; i++ {
				r := i
				if readingUp {
					r = dim - 1 - i
				}
				for cOffset := 0; cOffset < 2; cOffset++ {
					c := col - cOffset
					if fnMask[r][c] {
						continue
					}
					cur[bitCount] = [2]uint8{uint8(r), uint8(c)}
					bitCount++
					if bitCount == 8 {
						bitCoords = append(bitCoords, cur)
						if len(bitCoords) == totalCodewords {
							break outer
						}
						bitCount = 0
					}
				}
			}
			readingUp = !readingUp
		}

		var maskXor [8][]uint8
		for m := 0; m < 8; m++ {
			xorSlice := make([]uint8, totalCodewords)
			for cw := 0; cw < totalCodewords; cw++ {
				var x uint8
				coords := bitCoords[cw]
				for b := 0; b < 8; b++ {
					x <<= 1
					if applyDataMask(m, int(coords[b][0]), int(coords[b][1])) {
						x |= 1
					}
				}
				xorSlice[cw] = x
			}
			maskXor[m] = xorSlice
		}

		versionLayoutCache[version] = &versionLayout{
			fnMask:    fnMask,
			bitCoords: bitCoords,
			maskXor:   maskXor,
		}
	})
	return versionLayoutCache[version]
}

func buildRawFunctionMask(version, dim int) [][]bool {
	mask := make([][]bool, dim)
	backing := make([]bool, dim*dim)
	for r := 0; r < dim; r++ {
		mask[r] = backing[r*dim : (r+1)*dim]
	}

	markRect := func(r0, c0, h, w int) {
		for r := r0; r < r0+h; r++ {
			row := mask[r]
			for c := c0; c < c0+w; c++ {
				row[c] = true
			}
		}
	}

	// Finder patterns + 1-module separators + adjacent format info areas.
	markRect(0, 0, 9, 9)
	markRect(0, dim-8, 9, 8)
	markRect(dim-8, 0, 8, 9)

	// Timing patterns (row 6 and col 6).
	for i := 8; i < dim-8; i++ {
		mask[6][i] = true
		mask[i][6] = true
	}

	// Alignment patterns (5x5), skipping the three corners occupied by finder patterns.
	centers := alignmentCentersTable[version]
	k := len(centers)
	for i, cy := range centers {
		for j, cx := range centers {
			if (i == 0 && j == 0) || (i == 0 && j == k-1) || (i == k-1 && j == 0) {
				continue
			}
			markRect(cy-2, cx-2, 5, 5)
		}
	}

	// Version information blocks (6x3 top-right, 3x6 bottom-left) for version >= 7.
	if version >= 7 {
		markRect(0, dim-11, 6, 3)
		markRect(dim-11, 0, 3, 6)
	}

	return mask
}

// BuildFunctionPatternMask returns a dim x dim boolean matrix where true marks
// non-data modules: finder patterns + separators, timing patterns, dark module,
// alignment patterns, format info areas, and (for version >= 7) version info areas.
func BuildFunctionPatternMask(version int) [][]bool {
	layout := getVersionLayout(version)
	if layout == nil {
		return nil
	}
	dim := len(layout.fnMask)
	out := make([][]bool, dim)
	backing := make([]bool, dim*dim)
	for r := 0; r < dim; r++ {
		row := backing[r*dim : (r+1)*dim]
		copy(row, layout.fnMask[r])
		out[r] = row
	}
	return out
}

// ExtractRawCodewords traverses the QR matrix in standard zigzag order, skipping
// function patterns, and extracts raw codewords along with per-codeword minimum
// module confidence and module bit positions [8][2]{row, col} (MSB first).
func ExtractRawCodewords(bits [][]bool, conf [][]uint8, version int) (codewords []byte, cwConf []uint8, bitPos [][8][2]int, err error) {
	dim := DimensionForVersion(version)
	if dim == 0 {
		return nil, nil, nil, fmt.Errorf("invalid QR version %d", version)
	}
	if len(bits) != dim {
		return nil, nil, nil, fmt.Errorf("bits height %d does not match version %d dimension %d", len(bits), version, dim)
	}
	for r := 0; r < dim; r++ {
		if len(bits[r]) != dim {
			return nil, nil, nil, fmt.Errorf("bits row %d width %d does not match dimension %d", r, len(bits[r]), dim)
		}
	}
	if conf != nil {
		if len(conf) != dim {
			return nil, nil, nil, fmt.Errorf("conf height %d does not match dimension %d", len(conf), dim)
		}
		for r := 0; r < dim; r++ {
			if len(conf[r]) != dim {
				return nil, nil, nil, fmt.Errorf("conf row %d width %d does not match dimension %d", r, len(conf[r]), dim)
			}
		}
	}

	layout := getVersionLayout(version)
	totalCodewords := len(layout.bitCoords)

	codewords = make([]byte, totalCodewords)
	cwConf = make([]uint8, totalCodewords)
	bitPos = make([][8][2]int, totalCodewords)

	for cw := 0; cw < totalCodewords; cw++ {
		coords := layout.bitCoords[cw]
		var b byte
		minC := uint8(255)
		var pos [8][2]int
		for i := 0; i < 8; i++ {
			r := int(coords[i][0])
			c := int(coords[i][1])
			b <<= 1
			if bits[r][c] {
				b |= 1
			}
			pos[i] = [2]int{r, c}
			if conf != nil {
				if cv := conf[r][c]; cv < minC {
					minC = cv
				}
			}
		}
		codewords[cw] = b
		cwConf[cw] = minC
		bitPos[cw] = pos
	}

	return codewords, cwConf, bitPos, nil
}
