package decoder

import (
	"errors"
	"fmt"
)

var (
	errInvalidRSParams   = errors.New("rs: invalid block parameters")
	errTooManyErasures   = errors.New("rs: too many erasures")
	errUncorrectable     = errors.New("rs: uncorrectable errors")
	errSyndromeCheckFail = errors.New("rs: post-correction syndrome verification failed")
)

// computeSyndromesInto evaluates R(alpha^j) for j = 0..numECCodewords-1 into dst,
// returning true if any syndrome is non-zero.
func computeSyndromesInto(codewords []byte, numECCodewords int, dst []uint8) bool {
	var anyNonZero uint8
	for j := 0; j < numECCodewords; j++ {
		var s uint8
		for _, cw := range codewords {
			if s != 0 {
				s = expTable[int(logTable[s])+j]
			}
			s ^= cw
		}
		dst[j] = s
		anyNonZero |= s
	}
	return anyNonZero != 0
}

func computeSyndromes(codewords []byte, numECCodewords int) []uint8 {
	syndromes := make([]uint8, numECCodewords)
	computeSyndromesInto(codewords, numECCodewords, syndromes)
	return syndromes
}

func polyDegree(p []uint8) int {
	for i := len(p) - 1; i >= 0; i-- {
		if p[i] != 0 {
			return i
		}
	}
	return -1
}

// mulLinearInPlace multiplies polynomial p (of current length curLen) by (1 + xCoef*x) in place,
// returning the new length (capped at maxLen if maxLen > 0).
func mulLinearInPlace(p []uint8, curLen int, xCoef uint8, maxLen int) int {
	newLen := curLen + 1
	if maxLen > 0 && newLen > maxLen {
		newLen = maxLen
	}
	if newLen > curLen {
		p[newLen-1] = 0
	}
	for i := newLen - 1; i >= 1; i-- {
		prev := p[i-1]
		if prev != 0 {
			p[i] ^= gfMul(prev, xCoef)
		}
	}
	return newLen
}

func evalPolyLog(p []uint8, deg int, logX int) uint8 {
	if deg < 0 {
		return 0
	}
	acc := p[deg]
	for i := deg - 1; i >= 0; i-- {
		if acc != 0 {
			acc = expTable[int(logTable[acc])+logX]
		}
		acc ^= p[i]
	}
	return acc
}

// decodeRSWithSyndromes runs Berlekamp-Massey, root finding, Forney, and post-correction
// syndrome verification on out (which must already contain the received codewords).
// syndromes, gamma, modSyn must be valid for the e erasures in erasures[:e] and erased[0..n-1].
func decodeRSWithSyndromes(
	out []byte,
	nEC int,
	syndromes []uint8,
	gamma []uint8,
	modSyn []uint8,
	erased []bool,
	erasures []int,
) (int, error) {
	n := len(out)
	e := len(erasures)
	M := nEC - e
	U := modSyn[e:nEC]

	var cBuf, bBuf, tBuf [256]uint8
	cBuf[0] = 1
	bBuf[0] = 1
	cLen := 1
	bLen := 1
	L := 0
	m := 1
	b := uint8(1)

	for k := 0; k < M; k++ {
		d := U[k]
		limit := L
		if cLen-1 < limit {
			limit = cLen - 1
		}
		for i := 1; i <= limit; i++ {
			ci := cBuf[i]
			ui := U[k-i]
			if ci != 0 && ui != 0 {
				d ^= expTable[int(logTable[ci])+int(logTable[ui])]
			}
		}
		if d == 0 {
			m++
			continue
		}

		copy(tBuf[:cLen], cBuf[:cLen])
		tLen := cLen

		scaleLog := int(logTable[gfDiv(d, b)])
		need := bLen + m
		if need > cLen {
			for i := cLen; i < need; i++ {
				cBuf[i] = 0
			}
			cLen = need
		}
		for i := 0; i < bLen; i++ {
			bv := bBuf[i]
			if bv != 0 {
				cBuf[i+m] ^= expTable[int(logTable[bv])+scaleLog]
			}
		}

		if 2*L <= k {
			L = k + 1 - L
			copy(bBuf[:tLen], tBuf[:tLen])
			bLen = tLen
			b = d
			m = 1
		} else {
			m++
		}
	}

	C := cBuf[:cLen]
	degErr := polyDegree(C)
	if degErr != L || 2*L > M || C[0] == 0 {
		return 0, errUncorrectable
	}

	// Find the L unknown error positions (roots of C(x) in [0, n) not already erased).
	var posBuf [256]int
	posCount := 0
	copy(posBuf[:e], erasures)
	posCount = e

	switch L {
	case 0:
		// No unknown errors; only erasures need Forney magnitudes.
	case 1:
		// C[0] + C[1]*xInv == 0 => X = C[1]/C[0]
		X := gfDiv(C[1], C[0])
		pow := int(logTable[X])
		idx := n - 1 - pow
		if idx < 0 || idx >= n || erased[idx] {
			return 0, errUncorrectable
		}
		posBuf[posCount] = idx
		posCount++
	default:
		// Incremental Chien search from idx = n-1 (pow = 0, xInv = 1) down to idx = 0 (pow = n-1).
		var terms [256]uint8
		copy(terms[:L+1], C[:L+1])
		unknownFound := 0
		for idx := n - 1; idx >= 0; idx-- {
			if !erased[idx] {
				var sum uint8
				for i := 0; i <= L; i++ {
					sum ^= terms[i]
				}
				if sum == 0 {
					posBuf[posCount] = idx
					posCount++
					unknownFound++
					if unknownFound == L {
						break
					}
				}
			}
			if idx > 0 {
				for i := 1; i <= L; i++ {
					if ti := terms[i]; ti != 0 {
						terms[i] = expTable[int(logTable[ti])+(255-i)]
					}
				}
			}
		}
		if unknownFound != L {
			return 0, errUncorrectable
		}
	}

	degLambda := e + L
	if degLambda == 0 {
		return 0, nil
	}

	// Combined locator Lambda(x) = Gamma(x) * C(x) in lamBuf.
	var lamBuf [256]uint8
	for i := 0; i <= e; i++ {
		gi := gamma[i]
		if gi == 0 {
			continue
		}
		gLog := int(logTable[gi])
		for j := 0; j <= L; j++ {
			cj := C[j]
			if cj != 0 {
				lamBuf[i+j] ^= expTable[gLog+int(logTable[cj])]
			}
		}
	}

	// Error evaluator Omega(x) = S(x) * Lambda(x) mod x^nEC.
	var omegaBuf [256]uint8
	for i := 0; i < nEC; i++ {
		si := syndromes[i]
		if si == 0 {
			continue
		}
		sLog := int(logTable[si])
		maxJ := nEC - 1 - i
		if degLambda < maxJ {
			maxJ = degLambda
		}
		for j := 0; j <= maxJ; j++ {
			lj := lamBuf[j]
			if lj != 0 {
				omegaBuf[i+j] ^= expTable[sLog+int(logTable[lj])]
			}
		}
	}
	degOmega := polyDegree(omegaBuf[:nEC])

	// Formal derivative Lambda'(x) in GF(2^8) (odd-degree terms only).
	var lpBuf [256]uint8
	for i := 1; i <= degLambda; i += 2 {
		lpBuf[i-1] = lamBuf[i]
	}
	degLP := polyDegree(lpBuf[:degLambda])

	// Forney's formula: e_k = X_k * Omega(X_k^{-1}) / Lambda'(X_k^{-1}).
	numErrors := 0
	for k := 0; k < posCount; k++ {
		idx := posBuf[k]
		pow := n - 1 - idx
		X := expTable[pow]
		logXInv := (255 - pow) % 255

		denom := evalPolyLog(lpBuf[:], degLP, logXInv)
		if denom == 0 {
			return 0, errUncorrectable
		}
		num := evalPolyLog(omegaBuf[:], degOmega, logXInv)
		if num != 0 {
			errMag := gfMul(X, gfDiv(num, denom))
			if errMag != 0 {
				out[idx] ^= errMag
				numErrors++
			}
		}
	}

	// Verify all syndromes are zero after correction to guarantee zero false positives.
	var postBuf [256]uint8
	if computeSyndromesInto(out, nEC, postBuf[:nEC]) {
		return 0, errSyndromeCheckFail
	}

	return numErrors, nil
}

// DecodeRS decodes a QR Reed-Solomon block with optional known erasure indices.
// codewords[0] corresponds to x^{n-1} and codewords[n-1] to x^0, matching ISO/IEC 18004.
// Supports 2*t + len(erasures) <= numECCodewords.
func DecodeRS(codewords []byte, numECCodewords int, erasures []int) (corrected []byte, numErrors int, err error) {
	n := len(codewords)
	if n == 0 || n > 255 || numECCodewords <= 0 || numECCodewords >= n {
		return nil, 0, errInvalidRSParams
	}

	var erasedBuf [256]bool
	var uniqueBuf [256]int
	e := 0
	for _, idx := range erasures {
		if idx < 0 || idx >= n {
			return nil, 0, fmt.Errorf("%w: erasure index %d out of [0, %d)", errInvalidRSParams, idx, n)
		}
		if !erasedBuf[idx] {
			erasedBuf[idx] = true
			uniqueBuf[e] = idx
			e++
		}
	}

	if e > numECCodewords {
		return nil, 0, errTooManyErasures
	}

	out := make([]byte, n)
	copy(out, codewords)

	var synBuf [256]uint8
	if !computeSyndromesInto(out, numECCodewords, synBuf[:numECCodewords]) {
		return out, 0, nil
	}

	var gammaBuf [256]uint8
	var modSynBuf [256]uint8
	gammaBuf[0] = 1
	gammaLen := 1
	copy(modSynBuf[:numECCodewords], synBuf[:numECCodewords])

	for i := 0; i < e; i++ {
		X := expTable[n-1-uniqueBuf[i]]
		gammaLen = mulLinearInPlace(gammaBuf[:], gammaLen, X, 0)
		mulLinearInPlace(modSynBuf[:], numECCodewords, X, numECCodewords)
	}

	nErr, err := decodeRSWithSyndromes(
		out,
		numECCodewords,
		synBuf[:numECCodewords],
		gammaBuf[:gammaLen],
		modSynBuf[:numECCodewords],
		erasedBuf[:n],
		uniqueBuf[:e],
	)
	if err != nil {
		return nil, 0, err
	}
	return out, nErr, nil
}

// decodeRSSoftBuf runs hard RS followed by incremental soft-confidence erasure trials
// using caller-supplied stack scratch dstOut, performing zero heap allocations on failure.
func decodeRSSoftBuf(codewords []byte, confidences []uint8, numECCodewords int, dstOut *[256]byte) (int, int, bool) {
	n := len(codewords)
	if n == 0 || n > 255 || numECCodewords <= 0 || numECCodewords >= n {
		return 0, 0, false
	}

	var synBuf [256]uint8
	if !computeSyndromesInto(codewords, numECCodewords, synBuf[:numECCodewords]) {
		copy(dstOut[:n], codewords)
		return n, 0, true
	}

	var gammaBuf [256]uint8
	var modSynBuf [256]uint8
	var erasedBuf [256]bool

	gammaBuf[0] = 1
	copy(modSynBuf[:numECCodewords], synBuf[:numECCodewords])
	copy(dstOut[:n], codewords)

	if nErr, err := decodeRSWithSyndromes(
		dstOut[:n],
		numECCodewords,
		synBuf[:numECCodewords],
		gammaBuf[:1],
		modSynBuf[:numECCodewords],
		erasedBuf[:n],
		nil,
	); err == nil {
		return n, nErr, true
	}

	if len(confidences) != n {
		return 0, 0, false
	}

	minConf, maxConf := confidences[0], confidences[0]
	for _, c := range confidences[1:] {
		if c < minConf {
			minConf = c
		}
		if c > maxConf {
			maxConf = c
		}
	}
	if minConf == maxConf {
		return 0, 0, false
	}

	// Insertion sort indices [0..n-1] by ascending confidence (zero heap alloc).
	var idxBuf [256]int
	for i := 0; i < n; i++ {
		idxBuf[i] = i
	}
	for i := 1; i < n; i++ {
		key := idxBuf[i]
		kc := confidences[key]
		j := i - 1
		for j >= 0 {
			jc := confidences[idxBuf[j]]
			if jc < kc || (jc == kc && idxBuf[j] < key) {
				break
			}
			idxBuf[j+1] = idxBuf[j]
			j--
		}
		idxBuf[j+1] = key
	}

	// Cap erasures at numECCodewords - 2 so at least 2 spare syndromes always verify the result,
	// avoiding 100% false positives at e == numECCodewords on garbage blocks.
	gammaLen := 1
	for e := 2; e <= numECCodewords-2; e += 2 {
		idxA := idxBuf[e-2]
		idxB := idxBuf[e-1]
		erasedBuf[idxA] = true
		erasedBuf[idxB] = true

		XA := expTable[n-1-idxA]
		XB := expTable[n-1-idxB]
		gammaLen = mulLinearInPlace(gammaBuf[:], gammaLen, XA, 0)
		gammaLen = mulLinearInPlace(gammaBuf[:], gammaLen, XB, 0)
		mulLinearInPlace(modSynBuf[:], numECCodewords, XA, numECCodewords)
		mulLinearInPlace(modSynBuf[:], numECCodewords, XB, numECCodewords)

		copy(dstOut[:n], codewords)
		if nErr, err := decodeRSWithSyndromes(
			dstOut[:n],
			numECCodewords,
			synBuf[:numECCodewords],
			gammaBuf[:gammaLen],
			modSynBuf[:numECCodewords],
			erasedBuf[:n],
			idxBuf[:e],
		); err == nil {
			return n, nErr, true
		}
	}

	return 0, 0, false
}

// DecodeRSWithSoftConfidence attempts standard hard-decision RS decoding first.
// If that fails and confidences are available, it marks the e = 2, 4, 6, ...
// lowest-confidence byte positions as erasures up to numECCodewords-2.
func DecodeRSWithSoftConfidence(codewords []byte, confidences []uint8, numECCodewords int) ([]byte, int, bool) {
	var outBuf [256]byte
	n, nErr, ok := decodeRSSoftBuf(codewords, confidences, numECCodewords, &outBuf)
	if !ok {
		return nil, 0, false
	}
	out := make([]byte, n)
	copy(out, outBuf[:n])
	return out, nErr, true
}
