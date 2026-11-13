package decoder

import (
	"math/bits"
	"sort"
)

const (
	// FormatMask is the 15-bit XOR mask applied to QR format information (ISO/IEC 18004).
	FormatMask uint16 = 0x5412

	formatBCHPoly uint16 = 0x537
)

// Error correction level bit encodings per ISO/IEC 18004 Table 12.
const (
	ECM = 0 // 00 -> M (~15%)
	ECL = 1 // 01 -> L (~7%)
	ECH = 2 // 10 -> H (~30%)
	ECQ = 3 // 11 -> Q (~25%)
)

// FormatInfo describes a decoded QR format information entry.
type FormatInfo struct {
	ECLevel     int
	ECName      string
	Mask        int
	HammingDist int
}

type formatTableEntry struct {
	ecLevel  int
	ecName   string
	mask     int
	unmasked uint16
	masked   uint16
}

var formatTable [32]formatTableEntry

func ecLevelName(ec int) string {
	switch ec {
	case ECL:
		return "L"
	case ECM:
		return "M"
	case ECQ:
		return "Q"
	case ECH:
		return "H"
	default:
		return ""
	}
}

func encodeFormatBCH(data uint16) uint16 {
	v := data << 10
	for i := 4; i >= 0; i-- {
		if v&(1<<(i+10)) != 0 {
			v ^= formatBCHPoly << i
		}
	}
	return (data << 10) | v
}

func init() {
	idx := 0
	for ec := 0; ec < 4; ec++ {
		name := ecLevelName(ec)
		for mask := 0; mask < 8; mask++ {
			data := uint16((ec << 3) | mask)
			unmasked := encodeFormatBCH(data)
			formatTable[idx] = formatTableEntry{
				ecLevel:  ec,
				ecName:   name,
				mask:     mask,
				unmasked: unmasked,
				masked:   unmasked ^ FormatMask,
			}
			idx++
		}
	}
}

// DecodeFormatInformation decodes the two 15-bit format information copies
// (checking both masked with FormatMask and unmasked raw bits) within Hamming distance <= 3.
func DecodeFormatInformation(bits1, bits2 uint16) (FormatInfo, bool) {
	b1 := bits1 & 0x7FFF
	b2 := bits2 & 0x7FFF

	bestDist := 16
	var best FormatInfo
	found := false

	for _, entry := range formatTable {
		if b1 == entry.masked || b2 == entry.masked || b1 == entry.unmasked || b2 == entry.unmasked {
			return FormatInfo{
				ECLevel:     entry.ecLevel,
				ECName:      entry.ecName,
				Mask:        entry.mask,
				HammingDist: 0,
			}, true
		}

		d := bits.OnesCount16(b1 ^ entry.masked)
		if d2 := bits.OnesCount16(b2 ^ entry.masked); d2 < d {
			d = d2
		}
		if d1u := bits.OnesCount16(b1 ^ entry.unmasked); d1u < d {
			d = d1u
		}
		if d2u := bits.OnesCount16(b2 ^ entry.unmasked); d2u < d {
			d = d2u
		}

		if d < bestDist {
			bestDist = d
			best = FormatInfo{
				ECLevel:     entry.ecLevel,
				ECName:      entry.ecName,
				Mask:        entry.mask,
				HammingDist: d,
			}
			found = true
		}
	}

	if found && bestDist <= 3 {
		return best, true
	}
	return FormatInfo{}, false
}

// CandidateFormats returns up to limit FormatInfo candidates sorted by minimum
// Hamming distance across bits1 and bits2.
func CandidateFormats(bits1, bits2 uint16, limit int) []FormatInfo {
	b1 := bits1 & 0x7FFF
	b2 := bits2 & 0x7FFF

	candidates := make([]FormatInfo, len(formatTable))
	for i, entry := range formatTable {
		d := bits.OnesCount16(b1 ^ entry.masked)
		if d2 := bits.OnesCount16(b2 ^ entry.masked); d2 < d {
			d = d2
		}
		du := bits.OnesCount16(b1 ^ entry.unmasked)
		if d2u := bits.OnesCount16(b2 ^ entry.unmasked); d2u < du {
			du = d2u
		}
		if du <= 3 && du < d {
			d = du
		}
		candidates[i] = FormatInfo{
			ECLevel:     entry.ecLevel,
			ECName:      entry.ecName,
			Mask:        entry.mask,
			HammingDist: d,
		}
	}

	sort.SliceStable(candidates, func(i, j int) bool {
		if candidates[i].HammingDist != candidates[j].HammingDist {
			return candidates[i].HammingDist < candidates[j].HammingDist
		}
		if candidates[i].ECLevel != candidates[j].ECLevel {
			return candidates[i].ECLevel < candidates[j].ECLevel
		}
		return candidates[i].Mask < candidates[j].Mask
	})

	if limit > 0 && limit < len(candidates) {
		return candidates[:limit]
	}
	return candidates
}
