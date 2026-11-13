package imageproc

import (
	"runtime"
	"sync"
)

func workerCount(n int, enable bool) int {
	if !enable || n <= 1 {
		return 1
	}
	workers := runtime.GOMAXPROCS(0)
	if workers <= 1 {
		return 1
	}
	if workers > 8 {
		workers = 8
	}
	if workers > n {
		workers = n
	}
	return workers
}

func parallelFor(n int, enable bool, fn func(start, end int)) {
	workers := workerCount(n, enable)
	if workers == 1 {
		fn(0, n)
		return
	}
	var wg sync.WaitGroup
	wg.Add(workers - 1)
	for w := 0; w < workers-1; w++ {
		start := (w * n) / workers
		end := ((w + 1) * n) / workers
		go func(s, e int) {
			defer wg.Done()
			fn(s, e)
		}(start, end)
	}
	lastStart := ((workers - 1) * n) / workers
	fn(lastStart, n)
	wg.Wait()
}

func parallelIndexed(n int, enable bool, fn func(worker, start, end int)) int {
	workers := workerCount(n, enable)
	if workers == 1 {
		fn(0, 0, n)
		return 1
	}
	var wg sync.WaitGroup
	wg.Add(workers - 1)
	for w := 0; w < workers-1; w++ {
		start := (w * n) / workers
		end := ((w + 1) * n) / workers
		go func(idx, s, e int) {
			defer wg.Done()
			fn(idx, s, e)
		}(w, start, end)
	}
	lastStart := ((workers - 1) * n) / workers
	fn(workers-1, lastStart, n)
	wg.Wait()
	return workers
}
