// Package model fetches the BBC speech segmenter model once and keeps it in
// the user cache directory. The files come from github.com/bbc/bbc-speech-segmenter
// (Apache-2.0) at a fixed commit and are checked against their SHA-256.
package model

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"time"
)

// commit is the bbc-speech-segmenter commit the model files are taken from.
const commit = "a6a1de29c41f882cae4c0ab84db6f54f1dbf0773"

// files maps each model file under recipe/model to its SHA-256.
var files = map[string]string{
	"final.raw":      "215c6977f0e77bf4de2274fcf5f6eec9a79140dafaab6720c18379243eadce3a",
	"plda":           "2b96094b6a26d852226af73abe06a857a046f1533812c077c0b26b9f6456209b",
	"transform.mat":  "f40afabf71d3e75842811dfbd00720926fc16cd6eabb188b1b6b6bac60f57f43",
	"mean.vec":       "86a53f2902839babf856e46f5b245b69bc2f7f00b65a0e8e9e06d152882712b7",
	"min_chunk_size": "64aeb9975f234becd55bb4635e6e2f2da7a6b7bf0a896f0c07763bdfbfb31420",
	"max_chunk_size": "876e13f4e07bb39705302c01f445ffd2d2c3b180a207e4d959d6b671c67da09b",
}

// Dir returns the directory that holds the model files, downloading any
// file that is missing or does not match its hash.
func Dir() (string, error) {
	cache, err := os.UserCacheDir()
	if err != nil {
		return "", err
	}
	dir := filepath.Join(cache, "zwfm-speech-segments", "bbc-speech-segmenter-"+commit[:12])
	if err := os.MkdirAll(dir, 0o755); err != nil { //nolint:gosec // the cache holds public model files only
		return "", err
	}
	for name, sum := range files {
		path := filepath.Join(dir, name)
		if hashOf(path) == sum {
			continue
		}
		url := "https://raw.githubusercontent.com/bbc/bbc-speech-segmenter/" + commit + "/recipe/model/" + name
		if err := download(url, path, sum); err != nil {
			return "", err
		}
	}
	return dir, nil
}

// hashOf returns the hex SHA-256 of a file, or "" if it cannot be read.
func hashOf(path string) string {
	f, err := os.Open(path) //nolint:gosec // path is a fixed model file name inside our cache directory
	if err != nil {
		return ""
	}
	defer func() { _ = f.Close() }()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return ""
	}
	return hex.EncodeToString(h.Sum(nil))
}

// client bounds the whole download, so a stalled connection fails instead
// of hanging; the largest file is 33 MB.
var client = &http.Client{Timeout: 10 * time.Minute}

// download writes url to path only if its SHA-256 matches.
func download(url, path, sum string) error {
	resp, err := client.Get(url)
	if err != nil {
		return err
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("%s: %s", url, resp.Status)
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), ".download-*")
	if err != nil {
		return err
	}
	// Clean up on failure; after a successful rename this finds nothing.
	defer func() { _ = os.Remove(tmp.Name()) }()
	hash := sha256.New()
	_, err = io.Copy(io.MultiWriter(tmp, hash), resp.Body)
	if cerr := tmp.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		return fmt.Errorf("%s: %w", url, err)
	}
	if got := hex.EncodeToString(hash.Sum(nil)); got != sum {
		return fmt.Errorf("%s: sha256 %s, want %s", url, got, sum)
	}
	return os.Rename(tmp.Name(), path)
}
