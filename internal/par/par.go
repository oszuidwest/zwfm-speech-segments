// Package par runs the iterations of a loop in parallel on all CPUs.
package par

import (
	"runtime"
	"sync"
	"sync/atomic"
)

// For runs fn(0), ..., fn(n-1) on up to GOMAXPROCS goroutines, handing out
// indexes in order, and returns when all calls have returned.
func For(n int, fn func(i int)) {
	var next atomic.Int64
	var wg sync.WaitGroup
	for range min(runtime.GOMAXPROCS(0), n) {
		wg.Go(func() {
			for i := int(next.Add(1)) - 1; i < n; i = int(next.Add(1)) - 1 {
				fn(i)
			}
		})
	}
	wg.Wait()
}
