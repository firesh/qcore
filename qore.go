// Package qore implements a high-performance, pure-Go QR code detection and decoding engine.
package qore

import (
	"errors"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"os"

	"github.com/firesh/qcore/internal/detector"
	"github.com/firesh/qcore/internal/imageproc"
)

// ErrNotFound is returned by Decode when no QR code could be decoded.
var ErrNotFound = errors.New("qore: no QR code found")

// Point represents a 2D pixel coordinate in the image.
type Point struct {
	X float64 `json:"x"`
	Y float64 `json:"y"`
}

// Result holds a detected or decoded QR code symbol.
type Result struct {
	Text        string   `json:"text"`
	RawBytes    []byte   `json:"raw_bytes,omitempty"`
	Points      [4]Point `json:"points"` // TL, TR, BR, BL
	Version     int      `json:"version,omitempty"`
	ECLevel     string   `json:"ec_level,omitempty"`
	Orientation float64  `json:"orientation"` // Degrees in (-180, 180] along TL -> TR edge
	Decoded     bool     `json:"decoded"`
}

// Options configures QR detection and decoding behavior.
type Options struct {
	Multi                      bool
	PureBarcode                bool
	MaxResults                 int
	ReturnUnreadableDetections bool
}

// DefaultOptions returns recommended default decoding options.
func DefaultOptions() Options {
	return Options{
		Multi: true,
	}
}

func toDetectorOpts(opts *Options, multiDefault bool) detector.Options {
	if opts == nil {
		return detector.Options{
			Multi: multiDefault,
		}
	}
	return detector.Options{
		Multi:                      opts.Multi,
		PureBarcode:                opts.PureBarcode,
		MaxResults:                 opts.MaxResults,
		ReturnUnreadableDetections: opts.ReturnUnreadableDetections,
	}
}

func convertResults(in []detector.Result) []Result {
	out := make([]Result, len(in))
	for i, r := range in {
		out[i] = Result{
			Text:        r.Text,
			RawBytes:    r.RawBytes,
			Version:     r.Version,
			ECLevel:     r.ECLevel,
			Orientation: r.Orientation,
			Decoded:     r.Decoded,
			Points: [4]Point{
				{X: r.Quad[0].X, Y: r.Quad[0].Y},
				{X: r.Quad[1].X, Y: r.Quad[1].Y},
				{X: r.Quad[2].X, Y: r.Quad[2].Y},
				{X: r.Quad[3].X, Y: r.Quad[3].Y},
			},
		}
	}
	return out
}

// Decode detects and decodes the first readable QR code in img.
// If opts is nil, Multi defaults to false.
func Decode(img image.Image, opts *Options) (*Result, error) {
	if img == nil {
		return nil, errors.New("qore: nil image")
	}
	dOpts := toDetectorOpts(opts, false)
	if opts == nil {
		dOpts.MaxResults = 1
	}
	res := detector.DetectAndDecodeImage(img, dOpts)
	for _, r := range res {
		if r.Decoded {
			conv := convertResults([]detector.Result{r})
			return &conv[0], nil
		}
	}
	return nil, ErrNotFound
}

// DecodeAll detects and decodes all QR codes in img.
func DecodeAll(img image.Image, opts *Options) ([]Result, error) {
	if img == nil {
		return nil, errors.New("qore: nil image")
	}
	dOpts := toDetectorOpts(opts, true)
	res := detector.DetectAndDecodeImage(img, dOpts)
	return convertResults(res), nil
}

// DecodeFile reads an image from disk (applying JPEG EXIF orientation) and decodes all QR codes.
func DecodeFile(path string, opts *Options) ([]Result, error) {
	img, err := imageproc.ReadImageWithOrientation(path)
	if err != nil {
		return nil, err
	}
	return DecodeAll(img, opts)
}

// DecodeRawFile reads an image from disk without applying EXIF orientation transforms.
func DecodeRawFile(path string, opts *Options) ([]Result, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	if err != nil {
		return nil, err
	}
	return DecodeAll(img, opts)
}
