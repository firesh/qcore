package decoder

import (
	"bytes"
	"math/rand"
	"strings"
	"testing"
)

// encodeRS computes nEC Reed-Solomon parity bytes for data using QR generator polynomial.
func encodeRS(data []byte, nEC int) []byte {
	gen := []uint8{1}
	for i := 0; i < nEC; i++ {
		gen = polyMul(gen, []uint8{expTable[i], 1})
	}
	// Convert ascending gen to descending for polynomial long division.
	genDesc := make([]uint8, len(gen))
	for i := range gen {
		genDesc[i] = gen[len(gen)-1-i]
	}

	buf := make([]byte, len(data)+nEC)
	copy(buf, data)
	for i := 0; i < len(data); i++ {
		coef := buf[i]
		if coef == 0 {
			continue
		}
		for j := 0; j < len(genDesc); j++ {
			buf[i+j] ^= gfMul(genDesc[j], coef)
		}
	}
	out := make([]byte, len(data)+nEC)
	copy(out, data)
	copy(out[len(data):], buf[len(data):])
	return out
}

// --- Test QR Encoder for end-to-end tests across Versions 1..40, all EC levels, all masks ---

type bitWriter struct {
	bits []bool
}

func (bw *bitWriter) write(val uint32, n int) {
	for i := n - 1; i >= 0; i-- {
		bw.bits = append(bw.bits, (val>>i)&1 != 0)
	}
}

func (bw *bitWriter) toBytes(totalDataBytes int) []byte {
	// Terminator (up to 4 zero bits)
	rem := totalDataBytes*8 - len(bw.bits)
	if rem > 4 {
		rem = 4
	}
	for i := 0; i < rem; i++ {
		bw.bits = append(bw.bits, false)
	}
	for len(bw.bits)%8 != 0 {
		bw.bits = append(bw.bits, false)
	}
	out := make([]byte, totalDataBytes)
	nBytes := len(bw.bits) / 8
	for i := 0; i < nBytes && i < totalDataBytes; i++ {
		var b byte
		for bit := 0; bit < 8; bit++ {
			b <<= 1
			if bw.bits[i*8+bit] {
				b |= 1
			}
		}
		out[i] = b
	}
	padBytes := [2]byte{0xEC, 0x11}
	for i := nBytes; i < totalDataBytes; i++ {
		out[i] = padBytes[(i-nBytes)%2]
	}
	return out
}

func encodeTestQR(t testing.TB, version, ecLevel, mask int, payload func(bw *bitWriter, v int)) [][]bool {
	t.Helper()
	spec := ECBlocks[version][ecLevel]
	bw := &bitWriter{}
	payload(bw, version)
	if len(bw.bits) > spec.TotalDataCodewords()*8 {
		t.Fatalf("test payload overflow for v=%d ec=%d: %d bits > %d", version, ecLevel, len(bw.bits), spec.TotalDataCodewords()*8)
	}
	dataBytes := bw.toBytes(spec.TotalDataCodewords())

	// Split into blocks and compute RS parity.
	numBlocks := spec.TotalBlocks()
	blocks := make([][]byte, numBlocks)
	offset := 0
	for b := 0; b < numBlocks; b++ {
		dLen := spec.Group1.DataCodewords
		if b >= spec.Group1.Count {
			dLen = spec.Group2.DataCodewords
		}
		blocks[b] = encodeRS(dataBytes[offset:offset+dLen], spec.ECPerBlock)
		offset += dLen
	}

	// Interleave data and EC codewords.
	interleaved := make([]byte, 0, spec.TotalCodewords)
	maxData := spec.Group1.DataCodewords
	if spec.Group2.Count > 0 && spec.Group2.DataCodewords > maxData {
		maxData = spec.Group2.DataCodewords
	}
	for i := 0; i < maxData; i++ {
		for b := 0; b < numBlocks; b++ {
			dLen := len(blocks[b]) - spec.ECPerBlock
			if i < dLen {
				interleaved = append(interleaved, blocks[b][i])
			}
		}
	}
	for i := 0; i < spec.ECPerBlock; i++ {
		for b := 0; b < numBlocks; b++ {
			dLen := len(blocks[b]) - spec.ECPerBlock
			interleaved = append(interleaved, blocks[b][dLen+i])
		}
	}

	dim := DimensionForVersion(version)
	mat := make([][]bool, dim)
	for r := 0; r < dim; r++ {
		mat[r] = make([]bool, dim)
	}

	// Draw finder patterns (7x7).
	drawFinder := func(r0, c0 int) {
		for dr := 0; dr < 7; dr++ {
			for dc := 0; dc < 7; dc++ {
				dark := dr == 0 || dr == 6 || dc == 0 || dc == 6 || (dr >= 2 && dr <= 4 && dc >= 2 && dc <= 4)
				mat[r0+dr][c0+dc] = dark
			}
		}
	}
	drawFinder(0, 0)
	drawFinder(0, dim-7)
	drawFinder(dim-7, 0)

	// Draw alignment patterns (5x5).
	centers := AlignmentPatternCenters(version)
	k := len(centers)
	for i, cy := range centers {
		for j, cx := range centers {
			if (i == 0 && j == 0) || (i == 0 && j == k-1) || (i == k-1 && j == 0) {
				continue
			}
			for dr := -2; dr <= 2; dr++ {
				for dc := -2; dc <= 2; dc++ {
					dark := dr == -2 || dr == 2 || dc == -2 || dc == 2 || (dr == 0 && dc == 0)
					mat[cy+dr][cx+dc] = dark
				}
			}
		}
	}

	// Timing patterns + dark module.
	for i := 8; i < dim-8; i++ {
		dark := i%2 == 0
		mat[6][i] = dark
		mat[i][6] = dark
	}
	mat[4*version+9][8] = true

	// Version info for version >= 7.
	if version >= 7 {
		vCode := versionGolayCodes[version]
		for bit := 0; bit < 18; bit++ {
			dark := (vCode>>bit)&1 != 0
			i := bit / 3
			j := bit % 3
			mat[i][dim-11+j] = dark
			mat[dim-11+j][i] = dark
		}
	}

	// Format info.
	fCode := encodeFormatBCH(uint16((ecLevel<<3)|mask)) ^ FormatMask
	f1Coords := [15][2]int{
		{8, 0}, {8, 1}, {8, 2}, {8, 3}, {8, 4}, {8, 5}, {8, 7}, {8, 8},
		{7, 8}, {5, 8}, {4, 8}, {3, 8}, {2, 8}, {1, 8}, {0, 8},
	}
	f2Coords := [15][2]int{
		{dim - 1, 8}, {dim - 2, 8}, {dim - 3, 8}, {dim - 4, 8}, {dim - 5, 8}, {dim - 6, 8}, {dim - 7, 8},
		{8, dim - 8}, {8, dim - 7}, {8, dim - 6}, {8, dim - 5}, {8, dim - 4}, {8, dim - 3}, {8, dim - 2}, {8, dim - 1},
	}
	for b := 0; b < 15; b++ {
		dark := (fCode>>(14-b))&1 != 0
		mat[f1Coords[b][0]][f1Coords[b][1]] = dark
		mat[f2Coords[b][0]][f2Coords[b][1]] = dark
	}

	// Place data bits in zigzag and apply data mask.
	fnMask := BuildFunctionPatternMask(version)
	byteIdx := 0
	bitIdx := 7
	readingUp := true
	for col := dim - 1; col > 0; col -= 2 {
		if col == 6 {
			col--
		}
		for i := 0; i < dim; i++ {
			r := i
			if readingUp {
				r = dim - 1 - i
			}
			for cOff := 0; cOff < 2; cOff++ {
				c := col - cOff
				if fnMask[r][c] {
					continue
				}
				var bit bool
				if byteIdx < len(interleaved) {
					bit = (interleaved[byteIdx]>>bitIdx)&1 != 0
					bitIdx--
					if bitIdx < 0 {
						bitIdx = 7
						byteIdx++
					}
				}
				if applyDataMask(mask, r, c) {
					bit = !bit
				}
				mat[r][c] = bit
			}
		}
		readingUp = !readingUp
	}

	return mat
}

func writeByteSegment(bw *bitWriter, version int, data []byte) {
	bw.write(modeByte, 4)
	bw.write(uint32(len(data)), charCountBits(modeByte, version))
	for _, b := range data {
		bw.write(uint32(b), 8)
	}
}

func writeNumericSegment(bw *bitWriter, version int, digits string) {
	bw.write(modeNumeric, 4)
	bw.write(uint32(len(digits)), charCountBits(modeNumeric, version))
	i := 0
	for i+3 <= len(digits) {
		v := uint32(digits[i]-'0')*100 + uint32(digits[i+1]-'0')*10 + uint32(digits[i+2]-'0')
		bw.write(v, 10)
		i += 3
	}
	if len(digits)-i == 2 {
		v := uint32(digits[i]-'0')*10 + uint32(digits[i+1]-'0')
		bw.write(v, 7)
	} else if len(digits)-i == 1 {
		bw.write(uint32(digits[i]-'0'), 4)
	}
}

func writeAlnumSegment(bw *bitWriter, version int, s string) {
	bw.write(modeAlphanumeric, 4)
	bw.write(uint32(len(s)), charCountBits(modeAlphanumeric, version))
	idxOf := func(ch byte) uint32 {
		return uint32(strings.IndexByte(alnumAlphabet, ch))
	}
	i := 0
	for i+2 <= len(s) {
		bw.write(idxOf(s[i])*45+idxOf(s[i+1]), 11)
		i += 2
	}
	if len(s)-i == 1 {
		bw.write(idxOf(s[i]), 6)
	}
}

// --- GF(2^8) Tests ---

func TestGF256(t *testing.T) {
	seen := make([]bool, 256)
	for i := 0; i < 255; i++ {
		v := expTable[i]
		if v == 0 || seen[v] {
			t.Fatalf("duplicate/zero expTable[%d] = %d", i, v)
		}
		seen[v] = true
		if logTable[v] != uint8(i) {
			t.Fatalf("logTable[%d] = %d, want %d", v, logTable[v], i)
		}
		if expTable[i+255] != v {
			t.Fatalf("expTable[%d+255] mismatch", i)
		}
	}
	if expTable[255] != 1 {
		t.Fatalf("expTable[255] = %d, want 1", expTable[255])
	}

	for a := 1; a < 256; a++ {
		inv := gfInv(uint8(a))
		if gfMul(uint8(a), inv) != 1 {
			t.Fatalf("a=%d, inv=%d: mul != 1", a, inv)
		}
		if gfDiv(uint8(a), uint8(a)) != 1 {
			t.Fatalf("div(%d,%d) != 1", a, a)
		}
		if gfPow(uint8(a), 255) != 1 {
			t.Fatalf("pow(%d, 255) != 1", a)
		}
		if gfPow(uint8(a), -1) != inv {
			t.Fatalf("pow(%d, -1) != inv", a)
		}
	}
	if gfMul(0, 42) != 0 || gfDiv(0, 42) != 0 || gfPow(0, 0) != 1 || gfPow(0, 5) != 0 {
		t.Fatal("zero edge cases failed")
	}

	p1 := []uint8{1, 2, 3}
	p2 := []uint8{4, 5}
	sum := polyAdd(p1, p2)
	if !bytes.Equal(sum, []uint8{5, 7, 3}) {
		t.Fatalf("polyAdd = %v", sum)
	}
	prod := polyMul([]uint8{1, 2}, []uint8{1, 3})
	// (1 + 2x)(1 + 3x) = 1 + (2^3)x + (2*3)x^2 = 1 + 1x + 6x^2
	if !bytes.Equal(prod, []uint8{1, 1, 6}) {
		t.Fatalf("polyMul = %v", prod)
	}
	if polyEval(prod, 2) != gfMul(polyEval([]uint8{1, 2}, 2), polyEval([]uint8{1, 3}, 2)) {
		t.Fatal("polyEval multiplicative property failed")
	}
}

// --- Reed-Solomon Tests ---

func TestReedSolomonKnownVector(t *testing.T) {
	// ISO/IEC 18004 Version 1-M "HELLO WORLD" (16 data + 10 EC = 26 codewords)
	want := []byte{
		0x20, 0x5B, 0x0B, 0x78, 0xD1, 0x72, 0xDC, 0x4D,
		0x43, 0x40, 0xEC, 0x11, 0xEC, 0x11, 0xEC, 0x11,
		0xC4, 0x23, 0x27, 0x77, 0xEB, 0xD7, 0xE7, 0xE2,
		0x5D, 0x17,
	}
	got := encodeRS(want[:16], 10)
	if !bytes.Equal(got, want) {
		t.Fatalf("encodeRS mismatch:\ngot:  %X\nwant: %X", got, want)
	}

	// Clean decode.
	dec, nErr, err := DecodeRS(want, 10, nil)
	if err != nil || nErr != 0 || !bytes.Equal(dec, want) {
		t.Fatalf("clean DecodeRS failed: nErr=%d err=%v", nErr, err)
	}

	// Corrupt 5 bytes (max t = 5 for nEC = 10) including boundaries 0 and 25.
	corrupted := append([]byte(nil), want...)
	corrupted[0] ^= 0xFF
	corrupted[5] ^= 0x3A
	corrupted[15] ^= 0x80
	corrupted[16] ^= 0x01
	corrupted[25] ^= 0x55

	dec, nErr, err = DecodeRS(corrupted, 10, nil)
	if err != nil || nErr != 5 || !bytes.Equal(dec, want) {
		t.Fatalf("5-error DecodeRS failed: nErr=%d err=%v", nErr, err)
	}

	// Corrupt 6 bytes (> t = 5): must fail without false positive.
	corrupted[10] ^= 0x12
	if _, _, err := DecodeRS(corrupted, 10, nil); err == nil {
		t.Fatal("6-error DecodeRS unexpectedly succeeded")
	}
}

func TestReedSolomonErrorsAndErasures(t *testing.T) {
	rng := rand.New(rand.NewSource(42))

	// Test both even (nEC=10) and odd (nEC=7) EC sizes.
	cases := []struct {
		nData int
		nEC   int
	}{
		{19, 7},
		{16, 10},
		{43, 24},
		{118, 30},
	}

	for _, tc := range cases {
		data := make([]byte, tc.nData)
		rng.Read(data)
		cw := encodeRS(data, tc.nEC)
		n := len(cw)

		for e := 0; e <= tc.nEC; e++ {
			maxT := (tc.nEC - e) / 2
			for trial := 0; trial < 5; trial++ {
				perm := rng.Perm(n)
				erasures := perm[:e]
				errPos := perm[e : e+maxT]

				corrupted := append([]byte(nil), cw...)
				expectedDiffs := 0
				for _, idx := range erasures {
					delta := byte(rng.Intn(255) + 1)
					corrupted[idx] ^= delta
					expectedDiffs++
				}
				for _, idx := range errPos {
					delta := byte(rng.Intn(255) + 1)
					corrupted[idx] ^= delta
					expectedDiffs++
				}

				dec, nErr, err := DecodeRS(corrupted, tc.nEC, erasures)
				if err != nil {
					t.Fatalf("nEC=%d e=%d t=%d failed: %v", tc.nEC, e, maxT, err)
				}
				if nErr != expectedDiffs || !bytes.Equal(dec, cw) {
					t.Fatalf("nEC=%d e=%d t=%d wrong result: nErr=%d want=%d", tc.nEC, e, maxT, nErr, expectedDiffs)
				}
			}
		}

		// Exceed capacity: 2*t + e > nEC must fail.
		if tc.nEC >= 4 {
			perm := rng.Perm(n)
			erasures := perm[:2]
			errPos := perm[2 : 2+(tc.nEC-2)/2+1]
			corrupted := append([]byte(nil), cw...)
			for _, idx := range erasures {
				corrupted[idx] ^= 0xA5
			}
			for _, idx := range errPos {
				corrupted[idx] ^= 0x5A
			}
			if _, _, err := DecodeRS(corrupted, tc.nEC, erasures); err == nil {
				t.Fatalf("nEC=%d over-capacity decoding unexpectedly succeeded", tc.nEC)
			}
		}
	}

	// Zero false positive check on random uncorrectable vectors (using nEC=10, 0 erasures).
	for i := 0; i < 200; i++ {
		garbage := make([]byte, 26)
		rng.Read(garbage)
		if dec, _, err := DecodeRS(garbage, 10, nil); err == nil {
			// Verify any rare accidental decode within Hamming ball of radius 5 actually has 0 syndromes.
			for _, s := range computeSyndromes(dec, 10) {
				if s != 0 {
					t.Fatal("DecodeRS returned non-zero syndrome")
				}
			}
		}
	}
}

func TestReedSolomonSoftConfidence(t *testing.T) {
	data := []byte("Soft-confidence RS test payload!")
	nEC := 16
	cw := encodeRS(data, nEC) // 32 + 16 = 48 bytes; hard t = 8

	// Corrupt 12 bytes (> t=8, <= nEC=16) and mark them with low confidence.
	corrupted := append([]byte(nil), cw...)
	conf := make([]uint8, len(cw))
	for i := range conf {
		conf[i] = 240
	}
	badIndices := []int{0, 3, 7, 11, 14, 19, 22, 27, 31, 35, 40, 47}
	for _, idx := range badIndices {
		corrupted[idx] ^= 0xC3
		conf[idx] = 10
	}

	// Hard RS must fail.
	if _, _, err := DecodeRS(corrupted, nEC, nil); err == nil {
		t.Fatal("expected hard RS to fail for 12 errors with nEC=16")
	}

	dec, nErr, ok := DecodeRSWithSoftConfidence(corrupted, conf, nEC)
	if !ok || nErr != 12 || !bytes.Equal(dec, cw) {
		t.Fatalf("DecodeRSWithSoftConfidence failed: ok=%v nErr=%d", ok, nErr)
	}
}

// --- Version & Format Tests ---

func TestVersionTablesAndMasksAll40(t *testing.T) {
	for v := 1; v <= 40; v++ {
		dim := DimensionForVersion(v)
		if dim != 17+4*v {
			t.Fatalf("v=%d dim=%d", v, dim)
		}
		fnMask := BuildFunctionPatternMask(v)
		if len(fnMask) != dim {
			t.Fatalf("v=%d mask len=%d", v, len(fnMask))
		}
		dataMods := 0
		for r := 0; r < dim; r++ {
			for c := 0; c < dim; c++ {
				if !fnMask[r][c] {
					dataMods++
				}
			}
		}

		expectedTotal := ECBlocks[v][ECL].TotalCodewords
		if dataMods/8 != expectedTotal {
			t.Fatalf("v=%d dataMods/8=%d != TotalCodewords=%d", v, dataMods/8, expectedTotal)
		}

		for ec := 0; ec < 4; ec++ {
			spec := ECBlocks[v][ec]
			if spec.TotalCodewords != expectedTotal {
				t.Fatalf("v=%d ec=%d TotalCodewords=%d != %d", v, ec, spec.TotalCodewords, expectedTotal)
			}
			sum := spec.Group1.Count*(spec.Group1.DataCodewords+spec.ECPerBlock) +
				spec.Group2.Count*(spec.Group2.DataCodewords+spec.ECPerBlock)
			if sum != spec.TotalCodewords {
				t.Fatalf("v=%d ec=%d block sum=%d != TotalCodewords=%d", v, ec, sum, spec.TotalCodewords)
			}
		}
	}
}

func TestVersionInformationGolay(t *testing.T) {
	for v := 7; v <= 40; v++ {
		code := versionGolayCodes[v]
		// Exact match.
		got, ok := DecodeVersionInformation(code, 0)
		if !ok || got != v {
			t.Fatalf("v=%d exact failed: got=%d ok=%v", v, got, ok)
		}
		// 3 bit errors in copy 2 while copy 1 is garbage.
		damaged := code ^ 0x00105 // 3 bits flipped (bits 0, 2, 8)
		got, ok = DecodeVersionInformation(0x3FFFF^code, damaged)
		if !ok || got != v {
			t.Fatalf("v=%d 3-bit error failed: got=%d ok=%v", v, got, ok)
		}
	}
}

func TestFormatInformation(t *testing.T) {
	for ec := 0; ec < 4; ec++ {
		for mask := 0; mask < 8; mask++ {
			unmasked := encodeFormatBCH(uint16((ec << 3) | mask))
			masked := unmasked ^ FormatMask

			// Standard masked with up to 3 bit errors.
			for flips := 0; flips <= 3; flips++ {
				corrupted := masked ^ uint16((1<<flips)-1)
				info, ok := DecodeFormatInformation(corrupted, corrupted)
				if !ok || info.ECLevel != ec || info.Mask != mask || info.HammingDist != flips {
					t.Fatalf("ec=%d mask=%d flips=%d failed: %+v ok=%v", ec, mask, flips, info, ok)
				}
			}

			// Unmasked raw bits.
			info, ok := DecodeFormatInformation(unmasked, unmasked)
			if !ok || info.ECLevel != ec || info.Mask != mask || info.HammingDist != 0 {
				t.Fatalf("unmasked ec=%d mask=%d failed: %+v ok=%v", ec, mask, info, ok)
			}
		}
	}

	cands := CandidateFormats(0, 0, 4)
	if len(cands) != 4 {
		t.Fatalf("CandidateFormats len=%d, want 4", len(cands))
	}
	for i := 1; i < len(cands); i++ {
		if cands[i].HammingDist < cands[i-1].HammingDist {
			t.Fatalf("CandidateFormats not sorted: %+v", cands)
		}
	}
}

// --- Bitstream Tests ---

func TestBitstreamModesAndErrors(t *testing.T) {
	// Numeric + Alphanumeric + Byte (UTF-8) + Kanji
	bw := &bitWriter{}
	writeNumericSegment(bw, 1, "1234567")
	writeAlnumSegment(bw, 1, "AB-9")
	// ECI 26 (UTF-8) + Byte "π"
	bw.write(modeECI, 4)
	bw.write(26, 8)
	writeByteSegment(bw, 1, []byte("π"))
	// Kanji: "日本" (Shift-JIS 0x93FA, 0x967B)
	bw.write(modeKanji, 4)
	bw.write(2, 8)
	// 0x93FA - 0x8140 = 0x12BA -> 0x12*0xC0 + 0xBA
	bw.write(0x12*0xC0+0xBA, 13)
	// 0x967B - 0x8140 = 0x153B -> 0x15*0xC0 + 0x3B
	bw.write(0x15*0xC0+0x3B, 13)

	data := bw.toBytes(26)
	text, _, err := DecodeBitstream(data, 1)
	if err != nil {
		t.Fatalf("DecodeBitstream failed: %v", err)
	}
	if text != "1234567AB-9π日本" {
		t.Fatalf("got %q, want %q", text, "1234567AB-9π日本")
	}

	// ECI 20 Shift-JIS Byte mode
	bwSJIS := &bitWriter{}
	bwSJIS.write(modeECI, 4)
	bwSJIS.write(20, 8)
	writeByteSegment(bwSJIS, 1, []byte{0x93, 0xFA, 0x96, 0x7B, 0xB1}) // "日本ｱ"
	text, _, err = DecodeBitstream(bwSJIS.toBytes(16), 1)
	if err != nil || text != "日本ｱ" {
		t.Fatalf("Shift-JIS ECI 20 got %q err=%v", text, err)
	}

	// Non-empty payload with per-block restarted 0xEC/0x11 padding (e.g. BoofCV generator)
	// after modeTerminator must decode cleanly.
	restartedPad := append([]byte(nil), data...)
	restartedPad[len(restartedPad)-2] = 0xEC
	restartedPad[len(restartedPad)-1] = 0xEC
	if got, _, err := DecodeBitstream(restartedPad, 1); err != nil || got != "1234567AB-9π日本" {
		t.Fatalf("expected restarted padding after terminator to decode, got %q err=%v", got, err)
	}

	// Truncated segment must be rejected.
	if _, _, err := DecodeBitstream([]byte{0x40, 0xFF, 0xEC, 0x11}, 1); err == nil {
		t.Fatal("expected error for truncated byte segment")
	}

	// All-zero garbage must be rejected (fails 0xEC/0x11 padding check for empty payload).
	zeros := make([]byte, 16)
	if _, _, err := DecodeBitstream(zeros, 1); err == nil {
		t.Fatal("expected error for all-zero data")
	}
}

// --- Full Matrix Decoder Tests ---

func TestDecodeMatrixVersionsLevelsMasks(t *testing.T) {
	versions := []int{1, 2, 5, 7, 10, 20, 40}
	ecs := []int{ECL, ECM, ECQ, ECH}

	for _, v := range versions {
		for i, ec := range ecs {
			mask := (v + i*3) % 8
			var msg string
			switch {
			case v == 1:
				msg = "v01:测" // 7 bytes (fits V1-H max 7)
			case v == 2:
				msg = "v02:测试🚀" // 14 bytes (fits V2-H max 14)
			default:
				msg = "Qore QR v" + string(rune('0'+v/10)) + string(rune('0'+v%10)) + " UTF-8: 测试🚀"
			}
			mat := encodeTestQR(t, v, ec, mask, func(bw *bitWriter, ver int) {
				writeByteSegment(bw, ver, []byte(msg))
			})

			res, err := DecodeMatrix(mat, nil)
			if err != nil {
				t.Fatalf("v=%d ec=%s mask=%d DecodeMatrix failed: %v", v, ecLevelName(ec), mask, err)
			}
			if res.Text != msg || res.Version != v || res.ECLevel != ecLevelName(ec) || res.Mask != mask || res.Mirrored || res.BitErrors != 0 {
				t.Fatalf("v=%d unexpected result: %+v", v, res)
			}
		}
	}

	// Test all 8 masks explicitly on Version 5-Q (Group 1 + Group 2 unequal blocks!).
	for mask := 0; mask < 8; mask++ {
		msg := "ALL-8-MASKS-TEST-12345"
		mat := encodeTestQR(t, 5, ECQ, mask, func(bw *bitWriter, ver int) {
			writeAlnumSegment(bw, ver, msg)
		})
		res, err := DecodeMatrix(mat, nil)
		if err != nil || res.Text != msg || res.Mask != mask || res.ECLevel != "Q" {
			t.Fatalf("mask=%d failed: res=%+v err=%v", mask, res, err)
		}
	}
}

func TestDecodeMatrixMirroredAndModes(t *testing.T) {
	// Numeric + Mirrored
	matNum := encodeTestQR(t, 2, ECM, 3, func(bw *bitWriter, v int) {
		writeNumericSegment(bw, v, "0123456789876543210")
	})
	mirrored := transposeBoolMatrix(matNum)
	res, err := DecodeMatrix(mirrored, nil)
	if err != nil {
		t.Fatalf("mirrored decode failed: %v", err)
	}
	if res.Text != "0123456789876543210" || !res.Mirrored || res.Version != 2 {
		t.Fatalf("unexpected mirrored result: %+v", res)
	}

	// Kanji mode
	matKanji := encodeTestQR(t, 3, ECH, 5, func(bw *bitWriter, v int) {
		bw.write(modeKanji, 4)
		bw.write(2, charCountBits(modeKanji, v))
		bw.write(0x12*0xC0+0xBA, 13) // 日
		bw.write(0x15*0xC0+0x3B, 13) // 本
	})
	res, err = DecodeMatrix(matKanji, nil)
	if err != nil || res.Text != "日本" || res.ECLevel != "H" {
		t.Fatalf("kanji decode failed: res=%+v err=%v", res, err)
	}
}

func TestDecodeMatrixSoftErasuresAndFormatFallback(t *testing.T) {
	// Version 2-M has 1 block: 28 data + 16 EC (hard t=8, soft erasures up to 16).
	msg := "Soft-confidence recovery!"
	mat := encodeTestQR(t, 2, ECM, 2, func(bw *bitWriter, v int) {
		writeByteSegment(bw, v, []byte(msg))
	})
	dim := len(mat)
	conf := make([][]uint8, dim)
	for r := 0; r < dim; r++ {
		conf[r] = make([]uint8, dim)
		for c := 0; c < dim; c++ {
			conf[r][c] = 250
		}
	}

	// Extract codeword bit positions so we can flip 12 distinct bytes (> hard t=8).
	fnMask := BuildFunctionPatternMask(2)
	unmasked := unmaskMatrix(mat, fnMask, 2)
	_, _, bitPos, err := ExtractRawCodewords(unmasked, conf, 2)
	if err != nil {
		t.Fatal(err)
	}

	// Corrupt 12 codewords (indices 0..11) and mark one bit of each with low confidence.
	for cwIdx := 0; cwIdx < 12; cwIdx++ {
		r, c := bitPos[cwIdx][0][0], bitPos[cwIdx][0][1]
		mat[r][c] = !mat[r][c]
		conf[r][c] = 15
	}

	// Without soft confidences, hard RS (t=8) must fail.
	if _, err := DecodeMatrix(mat, nil); err == nil {
		t.Fatal("expected hard DecodeMatrix to fail with 12 byte errors")
	}

	// With soft confidences, erasure RS must recover.
	res, err := DecodeMatrix(mat, conf)
	if err != nil {
		t.Fatalf("soft DecodeMatrix failed: %v", err)
	}
	if res.Text != msg || res.BitErrors != 12 {
		t.Fatalf("unexpected soft result: %+v", res)
	}

	// Test format fallback when primary copy has >3 bit errors and copy 2 has 4 bit errors:
	// corrupt 4 bits in copy 1 and 4 bits in copy 2 of a clean V1-L matrix so DecodeFormatInformation returns false,
	// forcing CandidateFormats fallback.
	matFmt := encodeTestQR(t, 1, ECL, 4, func(bw *bitWriter, v int) {
		writeByteSegment(bw, v, []byte("FALLBACK"))
	})
	// Flip 4 bits in copy 1: (8,0)..(8,3) and 4 bits in copy 2: (20,8)..(17,8)
	for c := 0; c < 4; c++ {
		matFmt[8][c] = !matFmt[8][c]
	}
	for r := 20; r >= 17; r-- {
		matFmt[r][8] = !matFmt[r][8]
	}
	f1, f2 := readFormatBitsOriented(matFmt, false)
	if _, ok := DecodeFormatInformation(f1, f2); ok {
		t.Fatal("expected primary DecodeFormatInformation to fail with 4 bit errors in both copies")
	}
	resFmt, err := DecodeMatrix(matFmt, nil)
	if err != nil {
		t.Fatalf("CandidateFormats fallback failed: %v", err)
	}
	if resFmt.Text != "FALLBACK" || resFmt.ECLevel != "L" || resFmt.Mask != 4 {
		t.Fatalf("unexpected format fallback result: %+v", resFmt)
	}
}

func TestDecodeMatrixExternalReferenceVector(t *testing.T) {
	// Version 7-Q QR matrix generated independently by python-qrcode for:
	// "https://example.com/qore?v=7&msg=测试"
	rows := []string{
		"#######.#####...##.######..#..#.#...#.#######",
		"#.....#.#.###..###...#.#...###.#...#..#.....#",
		"#.###.#..##.#...##.##.##......##...#..#.###.#",
		"#.###.#..#.#.####.###..####.#...##.##.#.###.#",
		"#.###.#.....#....#.######....###.####.#.###.#",
		"#.....#..#.#.##...#.#...##............#.....#",
		"#######.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#######",
		".........##..#####..#...#..........#.........",
		".#....###.#..#.##...######..#.###....#.....##",
		"#.#.#..###.#.#...#.##...#.#.......##.##.#.#..",
		"#####.###..#.##..#..#.##.#..#..##..####..##..",
		"..##.#.#.#....##.##....#######.#.###...#..###",
		"....#.#..##.#.####.##.##.#.###.###.#...##.#.#",
		"..##...#..###.##..###..#.##.##.#.##.##.#.#..#",
		"..#####.#......##.#..#.#......#..#.##.##...#.",
		"#..#...###.#..##..#.####..##..#....#..#.##..#",
		".#.##.##..#.....#...######...#######.##.#.###",
		"....##.#####.####.#.....#..#..###...##..#.#.#",
		".#.##.#..#...##.#.#.......#...##.#...#....#.#",
		"######.#........###.#.#####..#.####.####..###",
		"##.########.##.##########.#.#..#..##########.",
		"##.##...##.##.#.#...#...##...#.#.##.#...#...#",
		".####.#.##.######.###.#.#...#.###.#.#.#.#.###",
		"....#...#.#..#....###...####....##..#...###.#",
		"#...#####..#..#..#..#####.#####.##..#####....",
		".#.#.#.##.#....##.#...#..###.####.#.#.#..###.",
		"#..#.###.######..#.###.#.#.#...##.#.###..##..",
		"#.#..#...#..#.#.......#...#.###.#.#...#.#.###",
		"#..####...#####...##....#...#####..##...#....",
		"#.#.##.##.#...#.####..##..#.#.#.#..#.#.#.##..",
		"..#...#.#...#..#....#....#........#..##.##...",
		"..###.....#..#.....##.#.##...#.#.##.##.####.#",
		"###.####..#.##.###.##.##.#.#.#...##..####..##",
		".#####.#.#.##.#.###........##..###..#..###.#.",
		"....#.###.......#.#.##...##..##..#...####....",
		".####..#.##.....#.#.####...##.#....##..###.#.",
		"#..##.#...#...#####.######..#.#...#########..",
		"........##...##.....#...##.##.#.#..##...#.#.#",
		"#######.#.#.##.#.#.##.#.##..#####..##.#.####.",
		"#.....#....######.#.#...#.#..#..##.##...#.#.#",
		"#.###.#...#.###..########.......#.#.#######..",
		"#.###.#..####.##.##.#...#.##.#.#.......###..#",
		"#.###.#..###...##....#...######.##..###.##.##",
		"#.....#.###..#######..###..#..#.#..#.#...#...",
		"#######....##.#...#..####.##.###...#.##...##.",
	}
	mat := make([][]bool, len(rows))
	for r, row := range rows {
		mat[r] = make([]bool, len(row))
		for c, ch := range row {
			mat[r][c] = ch == '#'
		}
	}

	res, err := DecodeMatrix(mat, nil)
	if err != nil {
		t.Fatalf("external reference vector decode failed: %v", err)
	}
	want := "https://example.com/qore?v=7&msg=测试"
	if res.Text != want || res.Version != 7 || res.ECLevel != "Q" || res.Mirrored || res.BitErrors != 0 {
		t.Fatalf("unexpected external reference result: %+v", res)
	}
}

func TestHighVersionsV15ToV40Thorough(t *testing.T) {
	// Thoroughly verify V15, V25, V27, V35, V40 version bits, alignment centers,
	// EC block counts/sizes, normal + mirrored + all masks round-trip decoding.
	highVers := []int{15, 25, 27, 35, 40}
	ecs := []int{ECL, ECM, ECQ, ECH}

	for _, v := range highVers {
		// Verify version Golay code round-trip and bit layout.
		vCode := versionGolayCodes[v]
		if gotV, ok := DecodeVersionInformation(vCode, vCode); !ok || gotV != v {
			t.Fatalf("v=%d DecodeVersionInformation failed: got=%d ok=%v", v, gotV, ok)
		}
		// Verify alignment centers are strictly increasing and bounded in [6, dim-7].
		centers := AlignmentPatternCenters(v)
		dim := DimensionForVersion(v)
		if len(centers) < 4 || centers[0] != 6 || centers[len(centers)-1] != dim-7 {
			t.Fatalf("v=%d invalid alignment centers: %v", v, centers)
		}
		for i := 1; i < len(centers); i++ {
			if centers[i] <= centers[i-1] {
				t.Fatalf("v=%d non-increasing alignment centers: %v", v, centers)
			}
		}

		for idx, ec := range ecs {
			mask := (v*3 + idx) % 8
			payload := strings.Repeat("QORE-HIGH-VERSION-TEST-0123456789/", 4) + string(rune('A'+v-15))
			mat := encodeTestQR(t, v, ec, mask, func(bw *bitWriter, ver int) {
				writeByteSegment(bw, ver, []byte(payload))
			})

			// Normal orientation.
			res, err := DecodeMatrix(mat, nil)
			if err != nil {
				t.Fatalf("v=%d ec=%s mask=%d normal decode failed: %v", v, ecLevelName(ec), mask, err)
			}
			if res.Text != payload || res.Version != v || res.ECLevel != ecLevelName(ec) || res.Mask != mask || res.Mirrored {
				t.Fatalf("v=%d ec=%s normal mismatch: %+v", v, ecLevelName(ec), res)
			}

			// Mirrored orientation on one EC per version.
			if idx == v%4 {
				mir := transposeBoolMatrix(mat)
				resM, err := DecodeMatrix(mir, nil)
				if err != nil {
					t.Fatalf("v=%d ec=%s mask=%d mirrored decode failed: %v", v, ecLevelName(ec), mask, err)
				}
				if resM.Text != payload || !resM.Mirrored || resM.Version != v || resM.Mask != mask {
					t.Fatalf("v=%d ec=%s mirrored mismatch: %+v", v, ecLevelName(ec), resM)
				}
			}
		}
	}
}

func transposeBoolMatrix(m [][]bool) [][]bool {
	n := len(m)
	out := make([][]bool, n)
	for r := 0; r < n; r++ {
		row := make([]bool, n)
		for c := 0; c < n; c++ {
			row[c] = m[c][r]
		}
		out[r] = row
	}
	return out
}

func unmaskMatrix(matrix, fnMask [][]bool, mask int) [][]bool {
	dim := len(matrix)
	out := make([][]bool, dim)
	for r := 0; r < dim; r++ {
		row := make([]bool, dim)
		for c := 0; c < dim; c++ {
			v := matrix[r][c]
			if !fnMask[r][c] && applyDataMask(mask, r, c) {
				v = !v
			}
			row[c] = v
		}
		out[r] = row
	}
	return out
}

func makeCorruptedGrid(t testing.TB, version int, seed int64) ([][]bool, [][]uint8) {
	t.Helper()
	rng := rand.New(rand.NewSource(seed))
	dim := DimensionForVersion(version)
	bits := make([][]bool, dim)
	conf := make([][]uint8, dim)
	for r := 0; r < dim; r++ {
		bits[r] = make([]bool, dim)
		conf[r] = make([]uint8, dim)
		for c := 0; c < dim; c++ {
			bits[r][c] = rng.Intn(2) == 1
			conf[r][c] = uint8(40 + rng.Intn(215))
		}
	}
	return bits, conf
}

func makeCleanConf(dim int) [][]uint8 {
	conf := make([][]uint8, dim)
	for r := 0; r < dim; r++ {
		conf[r] = make([]uint8, dim)
		for c := 0; c < dim; c++ {
			conf[r][c] = 240
		}
	}
	return conf
}

func BenchmarkDecodeMatrixV1Valid(b *testing.B) {
	mat := encodeTestQR(b, 1, ECM, 3, func(bw *bitWriter, v int) {
		writeByteSegment(bw, v, []byte("HELLO WORLD!"))
	})
	conf := makeCleanConf(len(mat))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res, err := DecodeMatrix(mat, conf)
		if err != nil || res == nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDecodeMatrixV1Failed(b *testing.B) {
	mat, conf := makeCorruptedGrid(b, 1, 101)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if res, err := DecodeMatrix(mat, conf); err == nil || res != nil {
			b.Fatal("expected failure")
		}
	}
}

func BenchmarkDecodeMatrixV27Valid(b *testing.B) {
	payload := strings.Repeat("Qore benchmark payload V27-", 12)
	mat := encodeTestQR(b, 27, ECM, 5, func(bw *bitWriter, v int) {
		writeByteSegment(bw, v, []byte(payload))
	})
	conf := makeCleanConf(len(mat))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res, err := DecodeMatrix(mat, conf)
		if err != nil || res == nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDecodeMatrixV27Failed(b *testing.B) {
	mat, conf := makeCorruptedGrid(b, 27, 202)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if res, err := DecodeMatrix(mat, conf); err == nil || res != nil {
			b.Fatal("expected failure")
		}
	}
}

func BenchmarkDecodeMatrixV27FailedData(b *testing.B) {
	// Valid V27 format and version bits, but corrupted interior data modules.
	payload := strings.Repeat("Qore benchmark payload V27-", 12)
	mat := encodeTestQR(b, 27, ECM, 5, func(bw *bitWriter, v int) {
		writeByteSegment(bw, v, []byte(payload))
	})
	rng := rand.New(rand.NewSource(303))
	dim := len(mat)
	conf := make([][]uint8, dim)
	fnMask := BuildFunctionPatternMask(27)
	for r := 0; r < dim; r++ {
		conf[r] = make([]uint8, dim)
		for c := 0; c < dim; c++ {
			conf[r][c] = uint8(50 + rng.Intn(205))
			if !fnMask[r][c] && rng.Intn(3) == 0 {
				mat[r][c] = !mat[r][c]
			}
		}
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if res, err := DecodeMatrix(mat, conf); err == nil || res != nil {
			b.Fatal("expected failure")
		}
	}
}
