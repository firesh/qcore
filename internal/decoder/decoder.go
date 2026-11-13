package decoder

import (
	"errors"
	"fmt"
	"math/bits"
)

var (
	errVersionMismatch = errors.New("decoder: version info mismatch")
	errFormatDecode    = errors.New("decoder: unable to decode format information")
	errRSFailed        = errors.New("decoder: RS decode failed")
	errDecodeFailed    = errors.New("decoder: decoding failed")
)

// DecodeResult contains the decoded payload and metadata from a QR matrix.
type DecodeResult struct {
	Text      string
	RawBytes  []byte
	Version   int
	ECLevel   string
	Mask      int
	Mirrored  bool
	BitErrors int
}

// DecodeMatrix decodes a binary QR module matrix (bits[row][col], true = dark)
// with optional per-module confidence values (0..255, nil defaults to 255).
// It attempts normal orientation first and falls back to mirrored (transposed).
func DecodeMatrix(bits [][]bool, conf [][]uint8) (*DecodeResult, error) {
	dim := len(bits)
	if dim < 21 || dim > 177 || (dim-17)%4 != 0 {
		return nil, fmt.Errorf("decoder: invalid QR dimension %d", dim)
	}
	for r := 0; r < dim; r++ {
		if len(bits[r]) != dim {
			return nil, fmt.Errorf("decoder: non-square matrix at row %d (len=%d, want=%d)", r, len(bits[r]), dim)
		}
	}
	if conf != nil {
		if len(conf) != dim {
			return nil, fmt.Errorf("decoder: conf height %d != dimension %d", len(conf), dim)
		}
		for r := 0; r < dim; r++ {
			if len(conf[r]) != dim {
				return nil, fmt.Errorf("decoder: conf row %d width %d != dimension %d", r, len(conf[r]), dim)
			}
		}
	}

	res, normErr := decodeOrientation(bits, conf, false)
	if normErr == nil {
		return res, nil
	}

	if resMirrored, mirErr := decodeOrientation(bits, conf, true); mirErr == nil {
		return resMirrored, nil
	}

	return nil, normErr
}

func decodeOrientation(matrix [][]bool, conf [][]uint8, mirrored bool) (*DecodeResult, error) {
	dim := len(matrix)
	version := (dim - 17) / 4

	if version >= 7 {
		v1, v2 := readVersionBitsOriented(matrix, mirrored)
		expected := versionGolayCodes[version]
		d1 := bits.OnesCount32(v1 ^ expected)
		d2 := bits.OnesCount32(v2 ^ expected)
		if d1 > 4 && d2 > 4 {
			return nil, errVersionMismatch
		}
		if decodedVer, ok := DecodeVersionInformation(v1, v2); ok && decodedVer != version {
			return nil, errVersionMismatch
		}
	}

	f1, f2 := readFormatBitsOriented(matrix, mirrored)
	maxFmtDist := 4
	if mirrored {
		maxFmtDist = 3
	}
	var candBuf [4]FormatInfo
	numCands := buildFormatCandidatesInto(f1, f2, maxFmtDist, &candBuf)
	if numCands == 0 {
		return nil, errFormatDecode
	}

	layout := getVersionLayout(version)
	var firstErr error

	for idx := 0; idx < numCands; idx++ {
		cand := candBuf[idx]
		dataBytes, bitErrors, err := decodeRSBlocksFast(matrix, conf, layout, version, cand.ECLevel, cand.Mask, mirrored)
		if err != nil {
			if idx == 0 {
				firstErr = err
			}
			continue
		}

		text, rawBytes, err := DecodeBitstream(dataBytes, version)
		if err != nil {
			if idx == 0 || firstErr == nil {
				firstErr = err
			}
			continue
		}

		return &DecodeResult{
			Text:      text,
			RawBytes:  rawBytes,
			Version:   version,
			ECLevel:   cand.ECName,
			Mask:      cand.Mask,
			Mirrored:  mirrored,
			BitErrors: bitErrors,
		}, nil
	}

	if firstErr != nil {
		return nil, firstErr
	}
	return nil, errDecodeFailed
}

// buildFormatCandidatesInto fills dst with up to 4 format candidates within maxDist,
// ordered with primary DecodeFormatInformation first (if valid) followed by lowest Hamming distance.
func buildFormatCandidatesInto(f1, f2 uint16, maxDist int, dst *[4]FormatInfo) int {
	b1 := f1 & 0x7FFF
	b2 := f2 & 0x7FFF

	primary, hasPrimary := DecodeFormatInformation(b1, b2)
	count := 0
	if hasPrimary {
		dst[0] = primary
		count = 1
		// BCH(15,5) has min distance 7; if both copies match identically at dist 0,
		// no other format entry can be within distance <= 6.
		if primary.HammingDist == 0 && b1 == b2 {
			return 1
		}
	}

	// Collect remaining entries within maxDist sorted by distance.
	var top [4]FormatInfo
	topLen := 0
	for _, entry := range formatTable {
		if hasPrimary && entry.ecLevel == primary.ECLevel && entry.mask == primary.Mask {
			continue
		}
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
		if d > maxDist {
			continue
		}
		cand := FormatInfo{
			ECLevel:     entry.ecLevel,
			ECName:      entry.ecName,
			Mask:        entry.mask,
			HammingDist: d,
		}
		// Insert into top (sorted by HammingDist, ECLevel, Mask).
		pos := topLen
		for pos > 0 {
			prev := top[pos-1]
			if prev.HammingDist < cand.HammingDist ||
				(prev.HammingDist == cand.HammingDist && (prev.ECLevel < cand.ECLevel || (prev.ECLevel == cand.ECLevel && prev.Mask <= cand.Mask))) {
				break
			}
			if pos < len(top) {
				top[pos] = prev
			}
			pos--
		}
		if pos < len(top) {
			top[pos] = cand
			if topLen < len(top) {
				topLen++
			}
		}
	}

	for i := 0; i < topLen && count < len(dst); i++ {
		dst[count] = top[i]
		count++
	}
	return count
}

func readFormatBitsOriented(matrix [][]bool, mirrored bool) (uint16, uint16) {
	dim := len(matrix)
	get := func(r, c int) bool {
		if mirrored {
			return matrix[c][r]
		}
		return matrix[r][c]
	}

	var f1 uint16
	for c := 0; c <= 5; c++ {
		f1 <<= 1
		if get(8, c) {
			f1 |= 1
		}
	}
	f1 <<= 1
	if get(8, 7) {
		f1 |= 1
	}
	f1 <<= 1
	if get(8, 8) {
		f1 |= 1
	}
	f1 <<= 1
	if get(7, 8) {
		f1 |= 1
	}
	for r := 5; r >= 0; r-- {
		f1 <<= 1
		if get(r, 8) {
			f1 |= 1
		}
	}

	var f2 uint16
	for r := dim - 1; r >= dim-7; r-- {
		f2 <<= 1
		if get(r, 8) {
			f2 |= 1
		}
	}
	for c := dim - 8; c < dim; c++ {
		f2 <<= 1
		if get(8, c) {
			f2 |= 1
		}
	}

	return f1, f2
}

func readVersionBitsOriented(matrix [][]bool, mirrored bool) (uint32, uint32) {
	dim := len(matrix)
	var v1, v2 uint32
	if !mirrored {
		for k := 0; k < 18; k++ {
			i := k / 3
			j := k % 3
			if matrix[i][dim-11+j] {
				v1 |= 1 << k
			}
			if matrix[dim-11+j][i] {
				v2 |= 1 << k
			}
		}
		return v1, v2
	}
	for k := 0; k < 18; k++ {
		i := k / 3
		j := k % 3
		if matrix[dim-11+j][i] {
			v1 |= 1 << k
		}
		if matrix[i][dim-11+j] {
			v2 |= 1 << k
		}
	}
	return v1, v2
}

func applyDataMask(mask, r, c int) bool {
	switch mask {
	case 0:
		return (r+c)%2 == 0
	case 1:
		return r%2 == 0
	case 2:
		return c%3 == 0
	case 3:
		return (r+c)%3 == 0
	case 4:
		return (r/2+c/3)%2 == 0
	case 5:
		return (r*c)%2+(r*c)%3 == 0
	case 6:
		return ((r*c)%2+(r*c)%3)%2 == 0
	case 7:
		return ((r+c)%2+(r*c)%3)%2 == 0
	default:
		return false
	}
}

func extractCodewordAt(
	matrix [][]bool,
	conf [][]uint8,
	coords *[8][2]uint8,
	maskByte uint8,
	mirrored bool,
) (byte, uint8) {
	var b byte
	minC := uint8(255)
	if !mirrored {
		if conf == nil {
			for i := 0; i < 8; i++ {
				b <<= 1
				if matrix[coords[i][0]][coords[i][1]] {
					b |= 1
				}
			}
			return b ^ maskByte, 255
		}
		for i := 0; i < 8; i++ {
			r := coords[i][0]
			c := coords[i][1]
			b <<= 1
			if matrix[r][c] {
				b |= 1
			}
			if cv := conf[r][c]; cv < minC {
				minC = cv
			}
		}
		return b ^ maskByte, minC
	}

	if conf == nil {
		for i := 0; i < 8; i++ {
			b <<= 1
			if matrix[coords[i][1]][coords[i][0]] {
				b |= 1
			}
		}
		return b ^ maskByte, 255
	}
	for i := 0; i < 8; i++ {
		r := coords[i][1]
		c := coords[i][0]
		b <<= 1
		if matrix[r][c] {
			b |= 1
		}
		if cv := conf[r][c]; cv < minC {
			minC = cv
		}
	}
	return b ^ maskByte, minC
}

func decodeRSBlocksFast(
	matrix [][]bool,
	conf [][]uint8,
	layout *versionLayout,
	version, ecLevel, mask int,
	mirrored bool,
) ([]byte, int, error) {
	spec := ECBlocks[version][ecLevel]
	numBlocks := spec.TotalBlocks()
	ecPerBlock := spec.ECPerBlock
	totalData := spec.TotalDataCodewords()
	g1Count := spec.Group1.Count
	g1Data := spec.Group1.DataCodewords
	g2Count := spec.Group2.Count
	g2Data := spec.Group2.DataCodewords

	bitCoords := layout.bitCoords
	maskXor := layout.maskXor[mask]

	var blkBuf [256]byte
	var blkConf [256]uint8
	var corrBuf [256]byte

	// Decode Block 0 first so bad grids abort immediately without extracting remaining blocks.
	b0Len := g1Data + ecPerBlock
	for i := 0; i < g1Data; i++ {
		rawIdx := i * numBlocks
		blkBuf[i], blkConf[i] = extractCodewordAt(matrix, conf, &bitCoords[rawIdx], maskXor[rawIdx], mirrored)
	}
	for j := 0; j < ecPerBlock; j++ {
		rawIdx := totalData + j*numBlocks
		blkBuf[g1Data+j], blkConf[g1Data+j] = extractCodewordAt(matrix, conf, &bitCoords[rawIdx], maskXor[rawIdx], mirrored)
	}

	var conf0 []uint8
	if conf != nil {
		conf0 = blkConf[:b0Len]
	}
	_, _, ok := decodeRSSoftBuf(blkBuf[:b0Len], conf0, ecPerBlock, &corrBuf)
	if !ok {
		return nil, 0, errRSFailed
	}

	dataBytes := make([]byte, totalData)
	copy(dataBytes[:g1Data], corrBuf[:g1Data])
	totalBitErrors := 0
	for i := 0; i < b0Len; i++ {
		if blkBuf[i] != corrBuf[i] {
			totalBitErrors += bits.OnesCount8(blkBuf[i] ^ corrBuf[i])
		}
	}

	dataOutOff := g1Data
	for b := 1; b < numBlocks; b++ {
		dLen := g1Data
		if b >= g1Count {
			dLen = g2Data
		}
		bLen := dLen + ecPerBlock

		for i := 0; i < g1Data; i++ {
			rawIdx := i*numBlocks + b
			blkBuf[i], blkConf[i] = extractCodewordAt(matrix, conf, &bitCoords[rawIdx], maskXor[rawIdx], mirrored)
		}
		for i := g1Data; i < dLen; i++ {
			rawIdx := g1Data*numBlocks + (i-g1Data)*g2Count + (b - g1Count)
			blkBuf[i], blkConf[i] = extractCodewordAt(matrix, conf, &bitCoords[rawIdx], maskXor[rawIdx], mirrored)
		}
		for j := 0; j < ecPerBlock; j++ {
			rawIdx := totalData + j*numBlocks + b
			blkBuf[dLen+j], blkConf[dLen+j] = extractCodewordAt(matrix, conf, &bitCoords[rawIdx], maskXor[rawIdx], mirrored)
		}

		var confSlice []uint8
		if conf != nil {
			confSlice = blkConf[:bLen]
		}
		_, _, ok := decodeRSSoftBuf(blkBuf[:bLen], confSlice, ecPerBlock, &corrBuf)
		if !ok {
			return nil, 0, errRSFailed
		}

		copy(dataBytes[dataOutOff:dataOutOff+dLen], corrBuf[:dLen])
		dataOutOff += dLen
		for i := 0; i < bLen; i++ {
			if blkBuf[i] != corrBuf[i] {
				totalBitErrors += bits.OnesCount8(blkBuf[i] ^ corrBuf[i])
			}
		}
	}

	return dataBytes, totalBitErrors, nil
}
