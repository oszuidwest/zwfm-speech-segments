// Package kaldi reads the parts of Kaldi's binary format that the BBC
// speech segmenter model uses: tokens, basic types, and float/double
// vectors and matrices (kaldi base/io-funcs.cc, matrix/kaldi-matrix.cc).
package kaldi

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"strconv"
	"strings"
)

// Reader reads one Kaldi binary object stream.
type Reader struct {
	r *bufio.Reader
}

// Open opens a binary Kaldi file and checks its "\0B" header. The caller
// closes the returned file when done with the Reader.
func Open(path string) (*Reader, *os.File, error) {
	f, err := os.Open(path) //nolint:gosec // callers pass model files from our own cache directory
	if err != nil {
		return nil, nil, err
	}
	r := &Reader{r: bufio.NewReaderSize(f, 1<<20)}
	var head [2]byte
	if _, err := io.ReadFull(r.r, head[:]); err != nil || head != [2]byte{0, 'B'} {
		_ = f.Close()
		return nil, nil, fmt.Errorf("%s: not a binary Kaldi file", path)
	}
	return r, f, nil
}

// Token reads a whitespace-terminated token such as "<Plda>" or "FM".
func (k *Reader) Token() (string, error) {
	var b []byte
	for {
		c, err := k.r.ReadByte()
		if err != nil {
			return "", err
		}
		if c == ' ' || c == '\n' || c == '\t' || c == '\r' {
			if len(b) > 0 {
				return string(b), nil
			}
			continue
		}
		b = append(b, c)
	}
}

// Expect reads a token and fails if it is not want.
func (k *Reader) Expect(want string) error {
	got, err := k.Token()
	if err != nil {
		return err
	}
	if got != want {
		return fmt.Errorf("kaldi: expected %q, got %q", want, got)
	}
	return nil
}

// SkipTo reads tokens until it has read want.
func (k *Reader) SkipTo(want string) error {
	for {
		got, err := k.Token()
		if err != nil {
			return fmt.Errorf("kaldi: %q not found: %w", want, err)
		}
		if got == want {
			return nil
		}
	}
}

// Peek returns the next n bytes without consuming them, so a caller can
// tell a matrix ("FM"), vector ("FV"), bool ('T'/'F') or number apart.
func (k *Reader) Peek(n int) ([]byte, error) { return k.r.Peek(n) }

// Int32 reads a binary basic-type int32: a size byte (4) and 4 bytes.
func (k *Reader) Int32() (int32, error) {
	size, err := k.r.ReadByte()
	if err != nil {
		return 0, err
	}
	if size != 4 {
		return 0, fmt.Errorf("kaldi: int32 size byte %d", size)
	}
	var v int32
	err = binary.Read(k.r, binary.LittleEndian, &v)
	return v, err
}

// Float reads a binary basic-type float or double (size byte 4 or 8).
func (k *Reader) Float() (float64, error) {
	size, err := k.r.ReadByte()
	if err != nil {
		return 0, err
	}
	switch size {
	case 4:
		var v float32
		err = binary.Read(k.r, binary.LittleEndian, &v)
		return float64(v), err
	case 8:
		var v float64
		err = binary.Read(k.r, binary.LittleEndian, &v)
		return v, err
	}
	return 0, fmt.Errorf("kaldi: float size byte %d", size)
}

// Bool reads a binary bool, written as the character 'T' or 'F'.
func (k *Reader) Bool() (bool, error) {
	c, err := k.r.ReadByte()
	if err != nil {
		return false, err
	}
	switch c {
	case 'T':
		return true, nil
	case 'F':
		return false, nil
	}
	return false, fmt.Errorf("kaldi: bool byte %q", c)
}

// Vector reads an "FV" (float) or "DV" (double) vector.
func (k *Reader) Vector() ([]float64, error) {
	tok, err := k.Token()
	if err != nil {
		return nil, err
	}
	n, err := k.Int32()
	if err != nil {
		return nil, err
	}
	return k.values(tok, "FV", "DV", int(n))
}

// Matrix is a dense row-major matrix.
type Matrix struct {
	Rows, Cols int
	Data       []float64
}

// Row returns row i as a slice into Data.
func (m Matrix) Row(i int) []float64 { return m.Data[i*m.Cols : (i+1)*m.Cols] }

// Matrix reads an "FM" (float) or "DM" (double) matrix.
func (k *Reader) Matrix() (Matrix, error) {
	tok, err := k.Token()
	if err != nil {
		return Matrix{}, err
	}
	rows, err := k.Int32()
	if err != nil {
		return Matrix{}, err
	}
	cols, err := k.Int32()
	if err != nil {
		return Matrix{}, err
	}
	data, err := k.values(tok, "FM", "DM", int(rows)*int(cols))
	return Matrix{Rows: int(rows), Cols: int(cols), Data: data}, err
}

// values reads n little-endian elements, as float32 if tok is single and as
// float64 if tok is double.
func (k *Reader) values(tok, single, double string, n int) ([]float64, error) {
	out := make([]float64, n)
	switch tok {
	case single:
		buf := make([]byte, 4*n)
		if _, err := io.ReadFull(k.r, buf); err != nil {
			return nil, err
		}
		for i := range out {
			out[i] = float64(math.Float32frombits(binary.LittleEndian.Uint32(buf[4*i:])))
		}
	case double:
		buf := make([]byte, 8*n)
		if _, err := io.ReadFull(k.r, buf); err != nil {
			return nil, err
		}
		for i := range out {
			out[i] = math.Float64frombits(binary.LittleEndian.Uint64(buf[8*i:]))
		}
	default:
		return nil, fmt.Errorf("kaldi: expected %s or %s, got %q", single, double, tok)
	}
	return out, nil
}

// ReadVector reads a text vector file such as mean.vec ("[ 1.1 0.7 ... ]").
func ReadVector(path string) ([]float64, error) {
	data, err := os.ReadFile(path) //nolint:gosec // callers pass model files from our own cache directory
	if err != nil {
		return nil, err
	}
	fields := strings.Fields(string(data))
	if len(fields) < 2 || fields[0] != "[" || fields[len(fields)-1] != "]" {
		return nil, fmt.Errorf("%s: not a text Kaldi vector", path)
	}
	out := make([]float64, 0, len(fields)-2)
	for _, f := range fields[1 : len(fields)-1] {
		v, err := strconv.ParseFloat(f, 64)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", path, err)
		}
		out = append(out, v)
	}
	return out, nil
}

// ReadMatrix reads a file that holds a single matrix, such as transform.mat.
func ReadMatrix(path string) (Matrix, error) {
	r, f, err := Open(path)
	if err != nil {
		return Matrix{}, err
	}
	defer func() { _ = f.Close() }()
	return r.Matrix()
}
