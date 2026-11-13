package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/firesh/qcore"
)

type cliOutput struct {
	File      string        `json:"file"`
	Results   []qore.Result `json:"results"`
	ElapsedMs float64       `json:"elapsed_ms"`
	Error     string        `json:"error,omitempty"`
}

func main() {
	jsonOut := flag.Bool("json", false, "Output JSON")
	multi := flag.Bool("multi", true, "Detect multiple QR codes")
	raw := flag.Bool("raw", false, "Ignore EXIF orientation")
	unreadable := flag.Bool("unreadable", false, "Also output unreadable detected QR quads")
	batchStdin := flag.Bool("batch-stdin", false, "Read image paths line-by-line from stdin and emit JSONL")
	flag.Parse()

	opts := &qore.Options{
		Multi:                      *multi,
		ReturnUnreadableDetections: *unreadable,
	}

	decodeOne := func(path string) ([]qore.Result, float64, error) {
		var img image.Image
		var err error
		if *raw {
			f, openErr := os.Open(path)
			if openErr != nil {
				return nil, 0, openErr
			}
			img, _, err = image.Decode(f)
			f.Close()
		} else {
			start := time.Now()
			res, decErr := qore.DecodeFile(path, opts)
			return res, float64(time.Since(start).Microseconds()) / 1000.0, decErr
		}
		if err != nil {
			return nil, 0, err
		}
		start := time.Now()
		res, decErr := qore.DecodeAll(img, opts)
		elapsed := float64(time.Since(start).Microseconds()) / 1000.0
		return res, elapsed, decErr
	}

	if *batchStdin {
		scanner := bufio.NewScanner(os.Stdin)
		scanner.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
		enc := json.NewEncoder(os.Stdout)
		for scanner.Scan() {
			line := strings.TrimSpace(scanner.Text())
			if line == "" {
				continue
			}
			res, elapsed, err := decodeOne(line)
			out := cliOutput{
				File:      line,
				Results:   res,
				ElapsedMs: elapsed,
			}
			if err != nil {
				out.Error = err.Error()
			}
			_ = enc.Encode(out)
		}
		return
	}

	args := flag.Args()
	if len(args) == 0 {
		fmt.Fprintln(os.Stderr, "usage: qore [flags] <image-or-dir>...")
		os.Exit(2)
	}

	var files []string
	for _, arg := range args {
		info, err := os.Stat(arg)
		if err != nil {
			fmt.Fprintf(os.Stderr, "error: %v\n", err)
			os.Exit(1)
		}
		if info.IsDir() {
			_ = filepath.WalkDir(arg, func(p string, d os.DirEntry, err error) error {
				if err != nil || d.IsDir() {
					return nil
				}
				ext := strings.ToLower(filepath.Ext(p))
				if ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".gif" {
					files = append(files, p)
				}
				return nil
			})
		} else {
			files = append(files, arg)
		}
	}

	var allOut []cliOutput
	for _, f := range files {
		res, elapsed, err := decodeOne(f)
		if *jsonOut {
			out := cliOutput{File: f, Results: res, ElapsedMs: elapsed}
			if err != nil {
				out.Error = err.Error()
			}
			allOut = append(allOut, out)
			continue
		}
		if err != nil {
			fmt.Fprintf(os.Stderr, "%s: error: %v\n", f, err)
			continue
		}
		if len(res) == 0 {
			fmt.Printf("%s: (none) [%.2fms]\n", f, elapsed)
			continue
		}
		for i, r := range res {
			status := "DECODED"
			if !r.Decoded {
				status = "DETECTED"
			}
			fmt.Printf("%s [%d] %s (%.2fms): %s\n", f, i, status, elapsed, r.Text)
		}
	}

	if *jsonOut {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		_ = enc.Encode(allOut)
	}
}
