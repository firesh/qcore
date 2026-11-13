package main

import (
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/firesh/qcore"
	"github.com/firesh/qcore/internal/geom"
)

type syntheticEntry struct {
	Filename string `json:"filename"`
	Expected any    `json:"expected"`
	DataName string `json:"data_name"`
	Scenario string `json:"scenario"`
}

func toExpectedList(v any) []string {
	switch x := v.(type) {
	case string:
		return []string{x}
	case []any:
		out := make([]string, 0, len(x))
		for _, item := range x {
			if s, ok := item.(string); ok {
				out = append(out, s)
			}
		}
		return out
	default:
		return nil
	}
}

func loadRawImage(path string) (image.Image, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	return img, err
}

func runSynthetic(dir string, outDir string) {
	manifestPath := filepath.Join(dir, "manifest.json")
	raw, err := os.ReadFile(manifestPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "synthetic manifest error: %v\n", err)
		return
	}
	var manifest []syntheticEntry
	if err := json.Unmarshal(raw, &manifest); err != nil {
		fmt.Fprintf(os.Stderr, "synthetic parse error: %v\n", err)
		return
	}

	opts := &qore.Options{
		Multi: true,
	}

	scenarioCorrect := map[string]int{}
	scenarioTotal := map[string]int{}
	catCorrect := map[string]int{}
	catTotal := map[string]int{}
	var failRows [][]string

	totalCorrect := 0
	var totalNs int64

	for _, entry := range manifest {
		imgPath := filepath.Join(dir, entry.Filename)
		img, err := loadRawImage(imgPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "load %s: %v\n", entry.Filename, err)
			continue
		}
		expected := toExpectedList(entry.Expected)
		sort.Strings(expected)

		start := time.Now()
		res, err := qore.DecodeAll(img, opts)
		elapsed := time.Since(start)
		totalNs += elapsed.Nanoseconds()

		decodedSet := map[string]bool{}
		var decodedList []string
		if err == nil {
			for _, r := range res {
				if r.Decoded && r.Text != "" {
					decodedSet[r.Text] = true
					decodedList = append(decodedList, r.Text)
				}
			}
		}
		found := 0
		for _, exp := range expected {
			if decodedSet[exp] {
				found++
			}
		}
		ok := found == len(expected) && len(expected) > 0
		if ok {
			totalCorrect++
			scenarioCorrect[entry.Scenario]++
		} else {
			failRows = append(failRows, []string{
				entry.Filename,
				entry.Scenario,
				entry.DataName,
				fmt.Sprintf("%d/%d", found, len(expected)),
				strings.Join(decodedList, " | "),
				fmt.Sprintf("%.2f", float64(elapsed.Microseconds())/1000.0),
			})
		}
		scenarioTotal[entry.Scenario]++
		cat := strings.SplitN(entry.Scenario, "_", 2)[0][:1]
		if ok {
			catCorrect[cat]++
		}
		catTotal[cat]++
	}

	total := len(manifest)
	totalMs := float64(totalNs) / 1e6
	avgMs := totalMs / float64(max(1, total))
	fmt.Printf("\n=== QORE Synthetic Benchmark ===\n")
	fmt.Printf("Total: %d/%d (%.2f%%), Total Time: %.1f ms, Avg Time: %.2f ms\n",
		totalCorrect, total, float64(totalCorrect)/float64(max(1, total))*100.0, totalMs, avgMs)

	cats := make([]string, 0, len(catTotal))
	for k := range catTotal {
		cats = append(cats, k)
	}
	sort.Strings(cats)
	fmt.Println("\nBy Category:")
	for _, c := range cats {
		fmt.Printf("  %s: %d/%d (%.1f%%)\n", c, catCorrect[c], catTotal[c],
			float64(catCorrect[c])/float64(max(1, catTotal[c]))*100.0)
	}

	scenarios := make([]string, 0, len(scenarioTotal))
	for k := range scenarioTotal {
		scenarios = append(scenarios, k)
	}
	sort.Strings(scenarios)
	fmt.Println("\nIncomplete Scenarios (< 100%):")
	for _, s := range scenarios {
		if scenarioCorrect[s] < scenarioTotal[s] {
			fmt.Printf("  %-26s %2d/%2d\n", s, scenarioCorrect[s], scenarioTotal[s])
		}
	}

	if outDir != "" {
		_ = os.MkdirAll(outDir, 0o755)
		f, err := os.Create(filepath.Join(outDir, "qore_synthetic_failures.csv"))
		if err == nil {
			w := csv.NewWriter(f)
			_ = w.Write([]string{"filename", "scenario", "data_name", "found", "decoded", "ms"})
			for _, r := range failRows {
				_ = w.Write(r)
			}
			w.Flush()
			f.Close()
		}
	}
}

func parseGTQuads(txtPath string) ([]geom.Quad, error) {
	raw, err := os.ReadFile(txtPath)
	if err != nil {
		return nil, err
	}
	var lines []string
	for _, l := range strings.Split(string(raw), "\n") {
		l = strings.TrimSpace(l)
		if l == "" || strings.HasPrefix(l, "#") {
			continue
		}
		lines = append(lines, l)
	}
	if len(lines) == 0 {
		return nil, nil
	}
	var quads []geom.Quad
	if lines[0] == "SETS" {
		for _, line := range lines[1:] {
			fields := strings.Fields(line)
			if len(fields) != 8 {
				continue
			}
			var vals [8]float64
			for i := 0; i < 8; i++ {
				vals[i], _ = strconv.ParseFloat(fields[i], 64)
			}
			quads = append(quads, geom.Quad{
				{X: vals[0], Y: vals[1]},
				{X: vals[2], Y: vals[3]},
				{X: vals[4], Y: vals[5]},
				{X: vals[6], Y: vals[7]},
			})
		}
	} else {
		var pts []geom.PointF
		for _, line := range lines {
			fields := strings.Fields(line)
			if len(fields) != 2 {
				continue
			}
			x, _ := strconv.ParseFloat(fields[0], 64)
			y, _ := strconv.ParseFloat(fields[1], 64)
			pts = append(pts, geom.PointF{X: x, Y: y})
		}
		for i := 0; i+3 < len(pts); i += 4 {
			quads = append(quads, geom.Quad{pts[i], pts[i+1], pts[i+2], pts[i+3]})
		}
	}
	return quads, nil
}

func matchQuads(pred []geom.Quad, gt []geom.Quad) int {
	if len(pred) == 0 || len(gt) == 0 {
		return 0
	}
	type pair struct {
		pi, gi int
		iou    float64
	}
	var cands []pair
	for pi, pq := range pred {
		pc := pq.Center()
		for gi, gq := range gt {
			iou := geom.PolygonIoU(pq, gq)
			gc := gq.Center()
			mutualCenter := gq.ContainsPoint(pc) && pq.ContainsPoint(gc)
			if iou >= 0.3 || mutualCenter {
				cands = append(cands, pair{pi: pi, gi: gi, iou: iou})
			}
		}
	}
	sort.Slice(cands, func(i, j int) bool {
		return cands[i].iou > cands[j].iou
	})
	usedP := make([]bool, len(pred))
	usedG := make([]bool, len(gt))
	matched := 0
	for _, c := range cands {
		if !usedP[c.pi] && !usedG[c.gi] {
			usedP[c.pi] = true
			usedG[c.gi] = true
			matched++
		}
	}
	return matched
}

func runBoofCV(dir string) {
	boofDir := dir
	if _, err := os.Stat(filepath.Join(dir, "boofcv-v3")); err == nil {
		boofDir = filepath.Join(dir, "boofcv-v3")
	}

	opts := &qore.Options{
		Multi:                      true,
		ReturnUnreadableDetections: true,
	}

	// 1. Decoding subset
	decDir := filepath.Join(boofDir, "decoding")
	decEntries, err := os.ReadDir(decDir)
	if err == nil {
		decTotal := 0
		decNormOK := 0
		decRawOK := 0
		var decNs int64
		for _, e := range decEntries {
			if e.IsDir() {
				continue
			}
			ext := strings.ToLower(filepath.Ext(e.Name()))
			if ext != ".png" && ext != ".jpg" && ext != ".jpeg" {
				continue
			}
			stem := strings.TrimSuffix(e.Name(), ext)
			gtPath := filepath.Join(decDir, stem+".txt")
			gtBytes, err := os.ReadFile(gtPath)
			if err != nil || stem == "readme" {
				continue
			}
			img, err := loadRawImage(filepath.Join(decDir, e.Name()))
			if err != nil {
				continue
			}
			decTotal++
			start := time.Now()
			res, _ := qore.DecodeAll(img, opts)
			decNs += time.Since(start).Nanoseconds()

			gtStr := string(gtBytes)
			gtNorm := strings.TrimSpace(strings.ReplaceAll(gtStr, "\r\n", "\n"))
			matchedNorm, matchedRaw := false, false
			for _, r := range res {
				if !r.Decoded {
					continue
				}
				if r.Text == gtStr {
					matchedRaw = true
				}
				if strings.TrimSpace(strings.ReplaceAll(r.Text, "\r\n", "\n")) == gtNorm {
					matchedNorm = true
				}
			}
			if matchedNorm {
				decNormOK++
			} else {
				fmt.Printf("  [decoding FAIL] %s\n", e.Name())
			}
			if matchedRaw {
				decRawOK++
			}
		}
		fmt.Printf("\n=== QORE BoofCV Decoding ===\n")
		fmt.Printf("Norm: %d/%d (%.1f%%), Raw: %d/%d, Total: %.1f ms, Avg: %.2f ms\n",
			decNormOK, decTotal, float64(decNormOK)/float64(max(1, decTotal))*100.0,
			decRawOK, decTotal, float64(decNs)/1e6, float64(decNs)/1e6/float64(max(1, decTotal)))
	}

	// 2. Detection subset
	detDir := filepath.Join(boofDir, "detection")
	cats, err := os.ReadDir(detDir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "boofcv detection dir error: %v\n", err)
		return
	}
	var catNames []string
	for _, c := range cats {
		if c.IsDir() {
			catNames = append(catNames, c.Name())
		}
	}
	sort.Strings(catNames)

	fmt.Printf("\n=== QORE BoofCV Detection ===\n")
	fmt.Printf("%-15s %5s %7s %13s %13s %9s\n", "Category", "Imgs", "GT", "Decoded", "Detected", "AvgMs")

	totalImgs := 0
	totalGT := 0
	totalDecoded := 0
	totalDetected := 0
	var totalNs int64

	for _, cat := range catNames {
		cDir := filepath.Join(detDir, cat)
		entries, _ := os.ReadDir(cDir)
		var imgs []string
		for _, e := range entries {
			if e.IsDir() {
				continue
			}
			ext := strings.ToLower(filepath.Ext(e.Name()))
			if ext == ".png" || ext == ".jpg" || ext == ".jpeg" {
				imgs = append(imgs, e.Name())
			}
		}
		sort.Strings(imgs)

		cImgs := 0
		cGT := 0
		cDecoded := 0
		cDetected := 0
		var cNs int64

		for _, name := range imgs {
			ext := filepath.Ext(name)
			stem := strings.TrimSuffix(name, ext)
			gtQuads, err := parseGTQuads(filepath.Join(cDir, stem+".txt"))
			if err != nil {
				continue
			}
			img, err := loadRawImage(filepath.Join(cDir, name))
			if err != nil {
				continue
			}
			cImgs++
			cGT += len(gtQuads)

			start := time.Now()
			res, _ := qore.DecodeAll(img, opts)
			cNs += time.Since(start).Nanoseconds()

			decCount := 0
			var predQuads []geom.Quad
			for _, r := range res {
				predQuads = append(predQuads, geom.Quad{
					{X: r.Points[0].X, Y: r.Points[0].Y},
					{X: r.Points[1].X, Y: r.Points[1].Y},
					{X: r.Points[2].X, Y: r.Points[2].Y},
					{X: r.Points[3].X, Y: r.Points[3].Y},
				})
				if r.Decoded {
					decCount++
				}
			}
			if decCount > len(gtQuads) {
				decCount = len(gtQuads)
			}
			cDecoded += decCount
			cDetected += matchQuads(predQuads, gtQuads)
		}

		totalImgs += cImgs
		totalGT += cGT
		totalDecoded += cDecoded
		totalDetected += cDetected
		totalNs += cNs

		fmt.Printf("%-15s %5d %7d %6d (%5.1f%%) %6d (%5.1f%%) %8.2f\n",
			cat, cImgs, cGT,
			cDecoded, float64(cDecoded)/float64(max(1, cGT))*100.0,
			cDetected, float64(cDetected)/float64(max(1, cGT))*100.0,
			float64(cNs)/1e6/float64(max(1, cImgs)))
	}

	fmt.Printf("%-15s %5d %7d %6d (%5.1f%%) %6d (%5.1f%%) %8.2f (Total %.1fms)\n",
		"TOTAL", totalImgs, totalGT,
		totalDecoded, float64(totalDecoded)/float64(max(1, totalGT))*100.0,
		totalDetected, float64(totalDetected)/float64(max(1, totalGT))*100.0,
		float64(totalNs)/1e6/float64(max(1, totalImgs)),
		float64(totalNs)/1e6)
}

func main() {
	syntheticDir := flag.String("synthetic", "", "Path to synthetic test_images directory")
	boofcvDir := flag.String("boofcv", "", "Path to boofcv-v3 directory")
	outDir := flag.String("out", "benchmark/results", "Output directory for diagnostics")
	flag.Parse()

	if *syntheticDir != "" {
		runSynthetic(*syntheticDir, *outDir)
	}
	if *boofcvDir != "" {
		runBoofCV(*boofcvDir)
	}
}
