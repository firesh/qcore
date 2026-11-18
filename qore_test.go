package qore

import (
	"encoding/json"
	"image"
	"image/color"
	"os"
	"path/filepath"
	"testing"
)

func TestDecodeSyntheticSample(t *testing.T) {
	sample := filepath.Join("benchmark", "synthetic", "test_images", "url__A01_doc_center.png")
	if _, err := os.Stat(sample); err != nil {
		t.Skip("synthetic benchmark images not present:", err)
	}
	res, err := DecodeFile(sample, nil)
	if err != nil {
		t.Fatalf("DecodeFile error: %v", err)
	}
	if len(res) != 1 || !res[0].Decoded || res[0].Text != "https://example.com/product/12345" {
		t.Fatalf("unexpected result: %+v", res)
	}
}

func TestDecodeBarrelDistortion(t *testing.T) {
	const filename = "dense_600__C10_barrel.png"
	dir := filepath.Join("benchmark", "synthetic", "test_images")
	sample := filepath.Join(dir, filename)
	if _, err := os.Stat(sample); os.IsNotExist(err) {
		t.Skip("synthetic benchmark image not present:", err)
	}
	raw, err := os.ReadFile(filepath.Join(dir, "manifest.json"))
	if err != nil {
		t.Fatal(err)
	}
	var entries []struct {
		Filename string
		Expected []string
	}
	if err := json.Unmarshal(raw, &entries); err != nil {
		t.Fatal(err)
	}
	var want string
	for _, entry := range entries {
		if entry.Filename == filename {
			if len(entry.Expected) != 1 {
				t.Fatal("expected one payload in synthetic manifest")
			}
			want = entry.Expected[0]
			break
		}
	}
	if want == "" {
		t.Fatal("missing expected payload in synthetic manifest")
	}
	for _, multi := range []bool{false, true} {
		results, err := DecodeRawFile(sample, &Options{Multi: multi})
		if err != nil {
			t.Fatal(err)
		}
		if len(results) != 1 || !results[0].Decoded || results[0].Text != want {
			t.Errorf("multi=%t: expected one matching QR code, got %d results", multi, len(results))
		}
	}
}

func TestDecodeNilAndBlank(t *testing.T) {
	if _, err := Decode(nil, nil); err == nil {
		t.Fatal("expected error for nil image")
	}
	blank := image.NewGray(image.Rect(0, 0, 100, 100))
	for i := range blank.Pix {
		blank.Pix[i] = 255
	}
	if _, err := Decode(blank, nil); err != ErrNotFound {
		t.Fatalf("expected ErrNotFound, got %v", err)
	}
	all, err := DecodeAll(blank, &Options{Multi: true})
	if err != nil {
		t.Fatalf("DecodeAll blank error: %v", err)
	}
	if len(all) != 0 {
		t.Fatalf("expected 0 results on blank, got %d", len(all))
	}

	rgba := image.NewRGBA(image.Rect(0, 0, 32, 32))
	rgba.SetRGBA(10, 10, color.RGBA{0, 0, 0, 255})
	all, err = DecodeAll(rgba, nil)
	if err != nil {
		t.Fatalf("DecodeAll rgba error: %v", err)
	}
	if len(all) != 0 {
		t.Fatalf("expected 0 results on noise dot, got %d", len(all))
	}
}

func TestIssue1(t *testing.T) {
	for _, opts := range []*Options{nil, {Multi: true}} {
		res, err := DecodeFile("testdata/issue1.png", opts)
		if err != nil || len(res) != 1 || res[0].Text != "https://example.com" {
			t.Fatalf("opts=%+v results=%+v err=%v", opts, res, err)
		}
	}
}
